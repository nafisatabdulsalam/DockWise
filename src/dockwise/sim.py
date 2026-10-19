from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from dockwise.config import DockWiseConfig
from dockwise.energy import OneSidedReserve, RecursiveLeastSquares, RobotPlant, StationPlant, route_features
from dockwise.scenario import clone_scenario, load_routes
from dockwise.types import ObservationView, Phase, Route, Scenario, StationFlows, StepEvents, Task


class FastSimulator:
    """Causal, fixed-decision-interval backend with finer physical integration."""

    def __init__(self, config: DockWiseConfig, scenario: Scenario, filter_mode: str = "joint"):
        self.config = config
        self.scenario = clone_scenario(scenario)
        self.filter_mode = filter_mode
        self.routes = load_routes()
        self.robot_plant = RobotPlant(config.robot)
        self.station = StationPlant(config.station)
        self.predictor = RecursiveLeastSquares(config.estimator)
        self.robot_reserve = OneSidedReserve(config.estimator, "robot")
        self.solar_reserve = OneSidedReserve(config.estimator, "solar")
        self.episode_duration_s = config.simulation.episode_hours * 3600.0
        self.k = config.tasks.candidate_count
        self.return_action = self.k
        self.wait_action = self.k + 1
        self.charge_action = self.k + 2
        self._reset_state()

    @property
    def action_count(self) -> int:
        return self.k + 3

    def _reset_state(self) -> None:
        cfg = self.config
        self.time_s = 0.0
        self.robot_capacity_wh = cfg.robot.nominal_capacity_wh * self.scenario.capacity_multiplier
        self.robot_energy_wh = cfg.robot.initial_soc * self.robot_capacity_wh
        self.robot_floor_wh = cfg.robot.operating_floor_soc * self.robot_capacity_wh
        self.robot_ceiling_wh = cfg.robot.operating_ceiling_soc * self.robot_capacity_wh
        self.phase = Phase.IDLE
        self.phase_remaining_s = 0.0
        self.active_task: Task | None = None
        self.active_route: Route | None = None
        self.active_start_energy_wh: float | None = None
        self.at_dock = True
        self.terminated = False
        self.truncated = False
        self.last_charge_requested = False
        self.last_invalid_action = False
        self._last_solar_issue_hour = -1
        self._pending_solar_forecasts: list[tuple[int, int, float]] = []
        self.metrics: dict[str, Any] = {
            "scenario_id": self.scenario.scenario_id,
            "policy": "",
            "arrivals_total": len(self.scenario.tasks),
            "dispatched": 0,
            "completed": 0,
            "on_time_completed": 0,
            "tardy_completed": 0,
            "newly_missed": 0,
            "robot_floor_violation": False,
            "station_floor_violation": False,
            "physical_depletion": False,
            "failed_return": False,
            "robot_energy_consumed_wh": 0.0,
            "robot_charge_received_wh": 0.0,
            "station_charge_time_s": 0.0,
            "station_starvation_wait_s": 0.0,
            "pv_generated_wh": 0.0,
            "pv_curtailed_wh": 0.0,
            "station_storage_loss_wh": 0.0,
            "charger_loss_wh": 0.0,
            "filter_candidate_opportunities": 0,
            "filter_candidate_exclusions": 0,
            "invalid_actions": 0,
            "decision_count": 0,
            "selected_distance_m": 0.0,
            "selected_payload_kg": 0.0,
        }
        self._update_solar_reserve()

    def _arrived_pending(self) -> list[Task]:
        return sorted(
            (
                task
                for task in self.scenario.tasks
                if task.arrival_s <= self.time_s and task.dispatched_s is None
            ),
            key=lambda task: (task.deadline_s, task.arrival_s, task.task_id),
        )

    def candidates(self) -> tuple[Task | None, ...]:
        tasks = self._arrived_pending()[: self.k]
        return tuple([*tasks, *([None] * (self.k - len(tasks)))])

    def _feature(self, task: Task) -> np.ndarray:
        return route_features(self.routes[task.destination_id], self.config.tasks, self.config.robot.speed_mps)

    def predicted_task_energy_wh(self, task: Task) -> float:
        return self.predictor.predict(self._feature(task))

    @property
    def adaptive_robot_margin_wh(self) -> float:
        return self.robot_reserve.margin_wh

    @property
    def adaptive_solar_margin_wh(self) -> float:
        return self.solar_reserve.margin_wh

    @property
    def filter_robot_margin_wh(self) -> float:
        if self.filter_mode in {"adaptive", "robot_adaptive", "joint"}:
            return self.adaptive_robot_margin_wh
        if self.filter_mode in {"expected", "unshielded"}:
            return 0.0
        return self.config.estimator.fixed_robot_margin_wh

    @property
    def filter_solar_margin_wh(self) -> float:
        if self.filter_mode in {"adaptive", "solar_adaptive", "joint"}:
            return self.adaptive_solar_margin_wh
        if self.filter_mode in {"expected", "unshielded"}:
            return 0.0
        return self.config.estimator.fixed_solar_margin_wh

    def task_energy_feasible(self, task: Task) -> bool:
        if self.filter_mode == "unshielded":
            return True
        required = self.predicted_task_energy_wh(task) + self.filter_robot_margin_wh
        return self._observed_robot_energy_wh() - self.robot_floor_wh >= required

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.action_count, dtype=bool)
        candidates = self.candidates()
        if self.active_task is None and self.at_dock:
            for index, task in enumerate(candidates):
                if task is not None:
                    self.metrics["filter_candidate_opportunities"] += 1
                    mask[index] = self.task_energy_feasible(task)
                    if not mask[index]:
                        self.metrics["filter_candidate_exclusions"] += 1
            robot_soc = self.robot_energy_wh / max(self.robot_capacity_wh, 1e-9)
            can_receive = robot_soc < self.config.robot.operating_ceiling_soc - 1e-6
            current_supply = self.scenario.solar_w(self.time_s) > self.config.station.idle_power_w
            stored_supply = self.station.energy_wh > 1e-6
            mask[self.charge_action] = bool(can_receive and (current_supply or stored_supply))
        elif self.active_task is None and not self.at_dock:
            mask[self.return_action] = True
        # WAIT doubles as CONTINUE and is the guaranteed valid fallback.
        mask[self.wait_action] = True
        return mask

    def _observed_robot_energy_wh(self) -> float:
        seed = int(self.scenario.metadata.get("task_seed", 0))
        deterministic_noise = np.sin(self.time_s / 173.0 + seed * 0.19)
        noise_wh = deterministic_noise * self.config.simulation.sensor_noise_fraction * self.config.robot.nominal_capacity_wh
        return min(self.config.robot.nominal_capacity_wh, max(0.0, self.robot_energy_wh + noise_wh))

    def _position(self) -> tuple[float, float]:
        dock_x, dock_y = 1.0, 1.0
        if self.active_route is None or self.phase in (Phase.IDLE, Phase.PICKUP):
            return dock_x, dock_y
        route = self.active_route
        travel_duration = route.distance_m / self.config.robot.speed_mps
        if self.phase == Phase.OUTBOUND:
            progress = 1.0 - self.phase_remaining_s / max(travel_duration, 1.0)
        elif self.phase == Phase.RETURN:
            progress = self.phase_remaining_s / max(travel_duration, 1.0)
        else:
            progress = 1.0
        return dock_x + progress * (route.x_m - dock_x), dock_y + progress * (route.y_m - dock_y)

    def observe(self) -> ObservationView:
        candidates = self.candidates()
        observed_energy = self._observed_robot_energy_wh()
        position_x, position_y = self._position()
        local_hour = (self.time_s / 3600.0) % 24.0
        phases = list(Phase)
        phase_one_hot = [float(self.phase == item) for item in phases]
        pending = self._arrived_pending()
        overdue = sum(task.deadline_s < self.time_s for task in pending)
        min_slack_h = min(((task.deadline_s - self.time_s) / 3600.0 for task in pending), default=0.0)
        forecasts = self.scenario.causal_forecast_w(
            self.time_s, self.config.simulation.solar_forecast_horizon_h
        )
        global_features = [
            observed_energy / self.config.robot.nominal_capacity_wh,
            self.robot_energy_wh / max(self.robot_capacity_wh, 1e-9),
            position_x / 20.0,
            position_y / 20.0,
            float(self.at_dock),
            *phase_one_hot,
            self.station.energy_wh / self.config.station.capacity_wh,
            self.scenario.solar_w(self.time_s) / max(self.config.station.pv_peak_w, 1e-9),
            np.sin(2.0 * np.pi * local_hour / 24.0),
            np.cos(2.0 * np.pi * local_hour / 24.0),
            max(0.0, self.episode_duration_s - self.time_s) / self.episode_duration_s,
            min(len(pending), 50) / 50.0,
            min(overdue, 50) / 50.0,
            np.clip(min_slack_h, -24.0, 24.0) / 24.0,
            min(self.predictor.updates, 100) / 100.0,
            self.robot_reserve.mean_error_wh / self.config.robot.nominal_capacity_wh,
            self.adaptive_robot_margin_wh / self.config.robot.nominal_capacity_wh,
            self.adaptive_solar_margin_wh / self.config.station.capacity_wh,
            self.filter_robot_margin_wh / self.config.robot.nominal_capacity_wh,
            self.filter_solar_margin_wh / self.config.station.capacity_wh,
        ]
        global_features.extend((forecasts / max(self.config.station.pv_peak_w, 1e-9)).tolist())
        candidate_features: list[float] = []
        for task in candidates:
            if task is None:
                candidate_features.extend([0.0] * 8)
                continue
            route = self.routes[task.destination_id]
            candidate_features.extend(
                [
                    1.0,
                    np.clip((task.deadline_s - self.time_s) / 3600.0, -24.0, 24.0) / 24.0,
                    route.distance_m / 40.0,
                    task.nominal_service_s / 1800.0,
                    self.predicted_task_energy_wh(task) / self.config.robot.nominal_capacity_wh,
                    task.destination_id / max(len(self.routes) - 1, 1),
                    np.clip((self.time_s - task.arrival_s) / 3600.0, 0.0, 24.0) / 24.0,
                    route.turns / 10.0,
                ]
            )
        vector = np.asarray([*global_features, *candidate_features], dtype=np.float32)
        return ObservationView(
            vector=vector,
            candidate_task_ids=tuple(task.task_id if task else None for task in candidates),
            candidate_tasks=candidates,
            robot_energy_estimate_wh=observed_energy,
            station_energy_wh=self.station.energy_wh,
            robot_margin_wh=self.filter_robot_margin_wh,
            solar_margin_wh=self.filter_solar_margin_wh,
            phase=self.phase,
        )

    def _dispatch(self, task: Task) -> None:
        task.dispatched_s = self.time_s
        self.active_task = task
        self.active_route = self.routes[task.destination_id]
        self.active_start_energy_wh = self.robot_energy_wh
        self.phase = Phase.PICKUP
        self.phase_remaining_s = self.config.tasks.pickup_time_s
        self.at_dock = True
        self.metrics["dispatched"] += 1
        self.metrics["selected_distance_m"] += self.active_route.distance_m
        self.metrics["selected_payload_kg"] += task.payload_kg

    def _finish_phase(self, events: StepEvents) -> None:
        if self.active_task is None or self.active_route is None:
            self.phase = Phase.IDLE
            self.at_dock = True
            return
        if self.phase == Phase.PICKUP:
            self.phase = Phase.OUTBOUND
            self.phase_remaining_s = self.active_route.distance_m / self.config.robot.speed_mps
            self.at_dock = False
        elif self.phase == Phase.OUTBOUND:
            self.phase = Phase.UNLOAD
            self.phase_remaining_s = self.config.tasks.unload_time_s
            self.at_dock = False
        elif self.phase == Phase.UNLOAD:
            self.active_task.completion_s = self.time_s
            self.metrics["completed"] += 1
            if self.time_s <= self.active_task.deadline_s + 1e-9:
                events.on_time_completed += 1
                self.metrics["on_time_completed"] += 1
            else:
                events.tardy_completed += 1
                self.metrics["tardy_completed"] += 1
            self.phase = Phase.RETURN
            self.phase_remaining_s = self.active_route.distance_m / self.config.robot.speed_mps
            self.at_dock = False
        elif self.phase == Phase.RETURN:
            measured_wh = max(0.0, (self.active_start_energy_wh or self.robot_energy_wh) - self.robot_energy_wh)
            residual = self.predictor.update(self._feature(self.active_task), measured_wh)
            self.robot_reserve.update(residual)
            self.phase = Phase.IDLE
            self.phase_remaining_s = 0.0
            self.active_task = None
            self.active_route = None
            self.active_start_energy_wh = None
            self.at_dock = True

    def _update_solar_reserve(self) -> None:
        current_hour = int(self.time_s // 3600)
        still_pending: list[tuple[int, int, float]] = []
        for issue_hour, horizon, forecast_wh in self._pending_solar_forecasts:
            if issue_hour + horizon <= current_hour:
                actual_wh = sum(
                    self.scenario.solar_w((issue_hour + lead) * 3600.0)
                    for lead in range(1, horizon + 1)
                )
                self.solar_reserve.update(forecast_wh - actual_wh)
            else:
                still_pending.append((issue_hour, horizon, forecast_wh))
        self._pending_solar_forecasts = still_pending
        if current_hour > self._last_solar_issue_hour:
            horizon = self.config.simulation.scheduler_horizon_h
            forecast_wh = float(self.scenario.causal_forecast_w(self.time_s, horizon).sum())
            self._pending_solar_forecasts.append((current_hour, horizon, forecast_wh))
            self._last_solar_issue_hour = current_hour

    @staticmethod
    def _add_flows(target: dict[str, Any], flows: StationFlows) -> None:
        target["pv_generated_wh"] += flows.pv_wh
        target["pv_curtailed_wh"] += flows.curtailed_wh
        target["station_storage_loss_wh"] += flows.storage_loss_wh
        target["charger_loss_wh"] += flows.charger_loss_wh
        target["robot_charge_received_wh"] += flows.charger_output_wh

    def _integrate(self, duration_s: float, charge_requested: bool, events: StepEvents) -> None:
        remaining = min(duration_s, self.episode_duration_s - self.time_s)
        while remaining > 1e-9 and not self.terminated:
            step_s = min(self.config.simulation.physics_interval_s, remaining)
            if self.active_task is not None:
                step_s = min(step_s, max(self.phase_remaining_s, 1e-9))
            local_hour = int(self.time_s // 3600) % 24
            daytime = self.config.tasks.workday_start_hour <= local_hour < self.config.tasks.workday_end_hour
            robot_soc = self.robot_energy_wh / max(self.robot_capacity_wh, 1e-9)
            request_w = 0.0
            if charge_requested and self.active_task is None and self.at_dock:
                request_w = self.station.available_robot_charge_w(robot_soc)
                self.metrics["station_charge_time_s"] += step_s
            flows = self.station.step(step_s, self.scenario.solar_w(self.time_s), request_w)
            self._add_flows(self.metrics, flows)
            if request_w > 0 and flows.unmet_bus_wh > 1e-12:
                self.metrics["station_starvation_wait_s"] += step_s
            plant_phase = Phase.CHARGING if request_w > 0 else self.phase
            payload = self.active_task.payload_kg if self.active_task else 0.0
            power_w = self.robot_plant.power_w(
                plant_phase,
                self.active_route,
                payload,
                crr_multiplier=self.scenario.crr_multiplier,
                auxiliary_multiplier=self.scenario.auxiliary_multiplier,
                daytime=daytime,
            )
            consumed_wh = power_w * step_s / 3600.0
            unbounded_energy = self.robot_energy_wh + flows.charger_output_wh - consumed_wh
            self.metrics["robot_energy_consumed_wh"] += consumed_wh
            if unbounded_energy < self.robot_floor_wh - 1e-12:
                events.robot_floor_violation = True
                self.metrics["robot_floor_violation"] = True
            if self.station.energy_wh < self.station.floor_wh - 1e-12:
                events.station_floor_violation = True
                self.metrics["station_floor_violation"] = True
            if unbounded_energy <= 0.0:
                events.physical_depletion = True
                events.failed_return = bool(not self.at_dock or self.active_task is not None)
                self.metrics["physical_depletion"] = True
                self.metrics["failed_return"] = events.failed_return
                self.terminated = True
            self.robot_energy_wh = min(self.robot_ceiling_wh, max(0.0, unbounded_energy))
            self.time_s += step_s
            remaining -= step_s
            if self.active_task is not None:
                self.phase_remaining_s -= step_s
                if self.phase_remaining_s <= 1e-8:
                    self._finish_phase(events)
            for task in self.scenario.tasks:
                if (
                    task.arrival_s <= self.time_s
                    and task.completion_s is None
                    and not task.missed_recorded
                    and task.deadline_s < self.time_s
                ):
                    task.missed_recorded = True
                    events.newly_missed += 1
                    self.metrics["newly_missed"] += 1
            self._update_solar_reserve()
        if self.time_s >= self.episode_duration_s - 1e-9 and not self.terminated:
            self.truncated = True

    def step(self, action: int) -> tuple[ObservationView, float, bool, bool, dict]:
        if self.terminated or self.truncated:
            raise RuntimeError("Cannot step a completed simulation")
        self.metrics["decision_count"] += 1
        events = StepEvents()
        masks = self.action_masks()
        self.last_invalid_action = not (0 <= int(action) < self.action_count and masks[int(action)])
        if self.last_invalid_action:
            self.metrics["invalid_actions"] += 1
            action = self.wait_action
        charge_requested = False
        if self.active_task is None:
            if 0 <= action < self.k:
                task = self.candidates()[action]
                if task is not None:
                    self._dispatch(task)
            elif action == self.charge_action and self.at_dock:
                charge_requested = True
            elif action == self.return_action and not self.at_dock:
                self.phase = Phase.RETURN
        self.last_charge_requested = charge_requested
        self._integrate(self.config.simulation.decision_interval_s, charge_requested, events)
        reward = (
            self.config.rewards.on_time_completion * events.on_time_completed
            + self.config.rewards.tardy_completion * events.tardy_completed
            + self.config.rewards.deadline_miss * events.newly_missed
            + self.config.rewards.elapsed_hour * self.config.simulation.decision_interval_s / 3600.0
        )
        if events.robot_floor_violation:
            reward += self.config.rewards.floor_violation
        if events.physical_depletion:
            reward += self.config.rewards.physical_depletion
        if self.terminated or self.truncated:
            unfinished = len(self.scenario.tasks) - int(self.metrics["completed"])
            reward += self.config.rewards.terminal_unfinished * unfinished
        info = {"events": asdict(events), "metrics": self.results() if (self.terminated or self.truncated) else None}
        return self.observe(), float(reward), self.terminated, self.truncated, info

    def results(self) -> dict[str, Any]:
        result = dict(self.metrics)
        total = max(int(result["arrivals_total"]), 1)
        completed = max(int(result["completed"]), 1)
        result.update(
            {
                "episode_time_s": self.time_s,
                "on_time_rate": result["on_time_completed"] / total,
                "completion_rate": result["completed"] / total,
                "deadline_missed": total - result["on_time_completed"],
                "episode_risk": bool(result["robot_floor_violation"] or result["failed_return"]),
                "robot_energy_per_completed_wh": result["robot_energy_consumed_wh"] / completed,
                "pv_utilisation": 1.0 - result["pv_curtailed_wh"] / max(result["pv_generated_wh"], 1e-12),
                "terminal_robot_energy_wh": self.robot_energy_wh,
                "terminal_station_energy_wh": self.station.energy_wh,
                "predictor_updates": self.predictor.updates,
                "robot_margin_final_wh": self.adaptive_robot_margin_wh,
                "solar_margin_final_wh": self.adaptive_solar_margin_wh,
                "filter_exclusion_rate": result["filter_candidate_exclusions"]
                / max(result["filter_candidate_opportunities"], 1),
                "mean_selected_distance_m": result["selected_distance_m"] / max(result["dispatched"], 1),
                "mean_selected_payload_kg": result["selected_payload_kg"] / max(result["dispatched"], 1),
                "weather_source": self.scenario.source,
                "forecast_bias": self.scenario.forecast_bias,
                "crr_multiplier": self.scenario.crr_multiplier,
                "auxiliary_multiplier": self.scenario.auxiliary_multiplier,
                "capacity_multiplier": self.scenario.capacity_multiplier,
                "demand_mode": self.config.tasks.demand_mode,
                "arrival_rate": self.config.tasks.arrival_rate_per_hour,
                "deadline_multiplier": self.config.tasks.deadline_multiplier,
            }
        )
        return result

