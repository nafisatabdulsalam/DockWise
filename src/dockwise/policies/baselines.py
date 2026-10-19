from __future__ import annotations

import itertools

import numpy as np

from dockwise.sim import FastSimulator
from dockwise.types import ObservationView, Task


class ThresholdPolicy:
    def __init__(self, threshold_soc: float):
        self.threshold_soc = threshold_soc
        self.name = f"threshold_{int(threshold_soc * 100)}"

    def act(self, observation: ObservationView, masks: np.ndarray, backend: FastSimulator) -> int:
        if backend.active_task is not None:
            return backend.wait_action
        observed_soc = observation.robot_energy_estimate_wh / backend.config.robot.nominal_capacity_wh
        feasible = [index for index in range(backend.k) if masks[index]]
        if observed_soc >= self.threshold_soc and feasible:
            return feasible[0]  # candidates are earliest-deadline first
        if masks[backend.charge_action]:
            return backend.charge_action
        return backend.wait_action


class ExpectedValuePolicy:
    name = "expected_value"

    def act(self, observation: ObservationView, masks: np.ndarray, backend: FastSimulator) -> int:
        if backend.active_task is not None:
            return backend.wait_action
        feasible = [index for index in range(backend.k) if masks[index]]
        if feasible:
            return feasible[0]
        if masks[backend.charge_action]:
            return backend.charge_action
        return backend.wait_action


class ModelBasedPolicy:
    """Bounded receding-horizon sequence search with lexicographic service objectives."""

    def __init__(self, name: str, search_depth: int = 3):
        self.name = name
        self.search_depth = search_depth

    def _projected_station_supply_wh(self, backend: FastSimulator) -> float:
        horizon = backend.config.simulation.scheduler_horizon_h
        solar = float(backend.scenario.causal_forecast_w(backend.time_s, horizon).sum())
        solar = max(0.0, solar - backend.filter_solar_margin_wh)
        stored_bus = max(0.0, backend.station.energy_wh - backend.station.floor_wh)
        stored_bus *= backend.config.station.storage_discharge_efficiency
        return solar + stored_bus

    def _score_sequence(self, backend: FastSimulator, tasks: tuple[Task, ...]) -> tuple:
        cursor = backend.time_s
        energy = backend._observed_robot_energy_wh()
        on_time = 0
        tardiness_s = 0.0
        completed = 0
        for task in tasks:
            predicted = backend.predicted_task_energy_wh(task) + backend.filter_robot_margin_wh
            if energy - backend.robot_floor_wh < predicted:
                break
            delivery_duration = (
                backend.config.tasks.pickup_time_s
                + backend.routes[task.destination_id].distance_m / backend.config.robot.speed_mps
                + backend.config.tasks.unload_time_s
            )
            cursor += delivery_duration
            if cursor <= task.deadline_s:
                on_time += 1
            else:
                tardiness_s += cursor - task.deadline_s
            cursor += backend.routes[task.destination_id].distance_m / backend.config.robot.speed_mps
            energy -= predicted
            completed += 1
            if cursor - backend.time_s > backend.config.simulation.scheduler_horizon_h * 3600:
                break
        first_id = tasks[0].task_id if tasks else 10**9
        return (on_time, -tardiness_s, completed, energy, -first_id)

    def act(self, observation: ObservationView, masks: np.ndarray, backend: FastSimulator) -> int:
        if backend.active_task is not None:
            return backend.wait_action
        candidates = [(index, task) for index, task in enumerate(observation.candidate_tasks) if task and masks[index]]
        if candidates:
            best_action = candidates[0][0]
            best_score: tuple | None = None
            depth = min(self.search_depth, len(candidates))
            for length in range(1, depth + 1):
                for sequence in itertools.permutations(candidates, length):
                    score = self._score_sequence(backend, tuple(item[1] for item in sequence))
                    if best_score is None or score > best_score:
                        best_score = score
                        best_action = sequence[0][0]
            # Preserve recharge headroom if the projected supply cannot restore the next reserve.
            first_task = observation.candidate_tasks[best_action]
            assert first_task is not None
            after = observation.robot_energy_estimate_wh - (
                backend.predicted_task_energy_wh(first_task) + backend.filter_robot_margin_wh
            )
            recharge_need = max(0.0, 0.50 * backend.robot_capacity_wh - after)
            projected_robot_charge = self._projected_station_supply_wh(backend) * backend.config.station.charger_efficiency
            if recharge_need > projected_robot_charge and masks[backend.charge_action]:
                return backend.charge_action
            return best_action
        if masks[backend.charge_action]:
            return backend.charge_action
        return backend.wait_action

