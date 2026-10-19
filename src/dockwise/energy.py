from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from dockwise.config import EstimatorConfig, RobotConfig, StationConfig, TaskConfig
from dockwise.types import Phase, Route, StationFlows


GRAVITY_MPS2 = 9.80665


def route_features(route: Route, task_config: TaskConfig, speed_mps: float) -> np.ndarray:
    """Observable delivery-plus-return features; payload is intentionally absent."""
    travel_duration_s = 2.0 * route.distance_m / speed_mps
    total_duration_min = (task_config.pickup_time_s + task_config.unload_time_s + travel_duration_s) / 60.0
    return np.asarray(
        [
            1.0,
            2.0 * route.distance_m,
            total_duration_min,
            2.0 * route.turns,
            2.0 * route.acceleration_events,
        ],
        dtype=np.float64,
    )


class RobotPlant:
    """True plant. It includes mass-dependent effects unavailable to the predictor."""

    def __init__(self, config: RobotConfig):
        self.config = config

    def power_w(
        self,
        phase: Phase,
        route: Route | None,
        payload_kg: float,
        *,
        crr_multiplier: float,
        auxiliary_multiplier: float,
        daytime: bool,
    ) -> float:
        cfg = self.config
        if phase in (Phase.OUTBOUND, Phase.RETURN) and route is not None:
            carried_payload = payload_kg if phase == Phase.OUTBOUND else 0.0
            mass = cfg.base_mass_kg + carried_payload
            duration_s = route.distance_m / cfg.speed_mps
            rolling_w = (
                cfg.rolling_resistance
                * crr_multiplier
                * mass
                * GRAVITY_MPS2
                * cfg.speed_mps
                / cfg.drivetrain_efficiency
            )
            acceleration_w = (
                0.5
                * mass
                * cfg.speed_mps**2
                * route.acceleration_events
                / max(duration_s, 1.0)
                / cfg.drivetrain_efficiency
            )
            turn_w = (
                cfg.turn_energy_j_per_kg_rad
                * mass
                * route.turns
                * (np.pi / 2.0)
                / max(duration_s, 1.0)
                / cfg.drivetrain_efficiency
            )
            return float(
                rolling_w
                + acceleration_w
                + turn_w
                + cfg.auxiliary_power_w * auxiliary_multiplier
            )
        if phase in (Phase.PICKUP, Phase.UNLOAD):
            return cfg.handling_power_w * auxiliary_multiplier
        if phase == Phase.CHARGING:
            return cfg.charge_aux_power_w * auxiliary_multiplier
        if daytime:
            return cfg.idle_power_w * auxiliary_multiplier
        return cfg.overnight_power_w * auxiliary_multiplier


class StationPlant:
    """DC-bus station model with explicit, non-overlapping conversion boundaries."""

    def __init__(self, config: StationConfig):
        self.config = config
        self.energy_wh = config.initial_soc * config.capacity_wh
        self.floor_violation_count = 0

    @property
    def floor_wh(self) -> float:
        return self.config.floor_soc * self.config.capacity_wh

    @property
    def ceiling_wh(self) -> float:
        return self.config.ceiling_soc * self.config.capacity_wh

    def available_robot_charge_w(self, robot_soc: float) -> float:
        cfg = self.config
        if robot_soc <= cfg.taper_start_soc:
            return cfg.charger_output_limit_w
        span = max(0.90 - cfg.taper_start_soc, 1e-9)
        fraction = min(1.0, max(0.0, (robot_soc - cfg.taper_start_soc) / span))
        return float(cfg.charger_output_limit_w - fraction * (cfg.charger_output_limit_w - cfg.taper_end_power_w))

    def step(self, dt_s: float, pv_w: float, requested_robot_charge_w: float) -> StationFlows:
        cfg = self.config
        flows = StationFlows()
        flows.pv_wh = max(0.0, pv_w) * dt_s / 3600.0
        requested_output_wh = max(0.0, min(requested_robot_charge_w, cfg.charger_output_limit_w)) * dt_s / 3600.0
        requested_charger_bus_wh = requested_output_wh / cfg.charger_efficiency
        flows.idle_wh = cfg.idle_power_w * dt_s / 3600.0
        total_bus_demand = flows.idle_wh + requested_charger_bus_wh

        if flows.pv_wh >= total_bus_demand:
            surplus_bus_wh = flows.pv_wh - total_bus_demand
            room_stored_wh = max(0.0, self.ceiling_wh - self.energy_wh)
            flows.station_charge_input_wh = min(
                surplus_bus_wh,
                room_stored_wh / cfg.storage_charge_efficiency,
            )
            flows.station_charge_stored_wh = flows.station_charge_input_wh * cfg.storage_charge_efficiency
            flows.storage_loss_wh = flows.station_charge_input_wh - flows.station_charge_stored_wh
            self.energy_wh += flows.station_charge_stored_wh
            flows.curtailed_wh = surplus_bus_wh - flows.station_charge_input_wh
            charger_bus_supplied_wh = requested_charger_bus_wh
        else:
            deficit_bus_wh = total_bus_demand - flows.pv_wh
            possible_bus_wh = self.energy_wh * cfg.storage_discharge_efficiency
            flows.station_discharge_bus_wh = min(deficit_bus_wh, possible_bus_wh)
            flows.station_withdrawn_wh = flows.station_discharge_bus_wh / cfg.storage_discharge_efficiency
            flows.storage_loss_wh = flows.station_withdrawn_wh - flows.station_discharge_bus_wh
            self.energy_wh -= flows.station_withdrawn_wh
            supplied_bus_wh = flows.pv_wh + flows.station_discharge_bus_wh
            flows.unmet_bus_wh = max(0.0, total_bus_demand - supplied_bus_wh)
            charger_bus_supplied_wh = max(0.0, supplied_bus_wh - flows.idle_wh)

        charger_bus_supplied_wh = min(charger_bus_supplied_wh, requested_charger_bus_wh)
        flows.charger_output_wh = charger_bus_supplied_wh * cfg.charger_efficiency
        flows.charger_loss_wh = charger_bus_supplied_wh - flows.charger_output_wh
        if self.energy_wh < self.floor_wh - 1e-12:
            self.floor_violation_count += 1
        self.energy_wh = min(self.ceiling_wh, max(0.0, self.energy_wh))
        if flows.station_charge_stored_wh > 0 and flows.station_withdrawn_wh > 0:
            raise RuntimeError("Station charged and discharged in the same integration step")
        return flows


class RecursiveLeastSquares:
    def __init__(self, estimator_config: EstimatorConfig):
        # Warm coefficients are assumptions, expressed in Wh per feature unit.
        self.theta = np.asarray([0.04, 0.0030, 0.070, 0.005, 0.003], dtype=np.float64)
        self.covariance = np.eye(len(self.theta), dtype=np.float64) * estimator_config.rls_initial_covariance
        self.forgetting = estimator_config.rls_forgetting_factor
        self.updates = 0

    def predict(self, features: np.ndarray) -> float:
        return max(0.01, float(features @ self.theta))

    def update(self, features: np.ndarray, observed_wh: float) -> float:
        prediction = self.predict(features)
        px = self.covariance @ features
        gain = px / (self.forgetting + float(features @ px))
        self.theta += gain * (observed_wh - float(features @ self.theta))
        self.covariance = (self.covariance - np.outer(gain, features) @ self.covariance) / self.forgetting
        self.updates += 1
        return observed_wh - prediction


class OneSidedReserve:
    def __init__(self, config: EstimatorConfig, kind: str):
        self.config = config
        self.kind = kind
        self.errors: deque[float] = deque(maxlen=config.residual_window)
        # Warm-start residuals are conservative calibration assumptions, not observations.
        floor = config.robot_margin_floor_wh if kind == "robot" else config.solar_margin_floor_wh
        self.warm_errors = (floor, 1.25 * floor, 1.5 * floor, 2.0 * floor)
        self._cached_margin: float | None = None

    def update(self, signed_overprediction_error_wh: float) -> None:
        self.errors.append(max(0.0, float(signed_overprediction_error_wh)))
        self._cached_margin = None

    @property
    def margin_wh(self) -> float:
        if self._cached_margin is not None:
            return self._cached_margin
        floor = self.config.robot_margin_floor_wh if self.kind == "robot" else self.config.solar_margin_floor_wh
        samples = list(self.errors)
        if len(samples) < self.config.minimum_residual_samples:
            samples = [*self.warm_errors, *samples]
        if not samples:
            self._cached_margin = floor
            return floor
        quantile = float(np.quantile(samples, self.config.reserve_quantile, method="higher"))
        # Small-sample addition shrinks to zero once the configured minimum is reached.
        deficit = max(0, self.config.minimum_residual_samples - len(self.errors))
        small_sample = floor * deficit / max(self.config.minimum_residual_samples, 1)
        self._cached_margin = max(floor, quantile + small_sample)
        return self._cached_margin

    @property
    def mean_error_wh(self) -> float:
        return float(np.mean(self.errors)) if self.errors else 0.0

