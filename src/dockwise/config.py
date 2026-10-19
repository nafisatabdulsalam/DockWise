from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass
class RobotConfig:
    base_mass_kg: float = 1.8
    nominal_capacity_wh: float = 19.98
    initial_soc: float = 0.80
    operating_floor_soc: float = 0.10
    operating_ceiling_soc: float = 0.90
    physical_depletion_soc: float = 0.0
    max_speed_mps: float = 0.26
    speed_mps: float = 0.20
    rolling_resistance: float = 0.025
    drivetrain_efficiency: float = 0.55
    auxiliary_power_w: float = 4.5
    idle_power_w: float = 3.2
    overnight_power_w: float = 1.2
    handling_power_w: float = 4.0
    charge_aux_power_w: float = 1.0
    turn_energy_j_per_kg_rad: float = 0.12
    battery_capacity_multiplier: float = 1.0


@dataclass
class StationConfig:
    pv_peak_w: float = 30.0
    capacity_wh: float = 100.0
    initial_soc: float = 0.50
    floor_soc: float = 0.10
    ceiling_soc: float = 0.90
    charger_output_limit_w: float = 10.0
    charger_efficiency: float = 0.90
    storage_charge_efficiency: float = 0.95
    storage_discharge_efficiency: float = 0.95
    idle_power_w: float = 0.15
    taper_start_soc: float = 0.80
    taper_end_power_w: float = 2.0


@dataclass
class TaskConfig:
    arrival_rate_per_hour: float = 4.0
    workday_start_hour: int = 8
    workday_end_hour: int = 16
    pickup_time_s: float = 45.0
    unload_time_s: float = 45.0
    deadline_multiplier: float = 2.0
    payloads_kg: tuple[float, ...] = (0.0, 1.0, 3.0)
    candidate_count: int = 6
    demand_mode: str = "changing"


@dataclass
class SimulationConfig:
    episode_hours: int = 72
    decision_interval_s: float = 60.0
    physics_interval_s: float = 10.0
    solar_forecast_horizon_h: int = 4
    scheduler_horizon_h: int = 2
    route_speed_mps: float = 0.20
    local_timezone: str = "Africa/Lagos"
    sensor_noise_fraction: float = 0.002


@dataclass
class EstimatorConfig:
    rls_forgetting_factor: float = 0.995
    rls_initial_covariance: float = 25.0
    residual_window: int = 50
    reserve_quantile: float = 0.95
    robot_margin_floor_wh: float = 0.08
    solar_margin_floor_wh: float = 0.10
    fixed_robot_margin_wh: float = 0.22
    fixed_solar_margin_wh: float = 0.50
    minimum_residual_samples: int = 8


@dataclass
class TrainingConfig:
    benchmark_steps: int = 2_048
    search_steps: int = 8_192
    final_steps: int = 20_000
    seeds: tuple[int, ...] = (11, 23, 37, 53, 71)
    learning_rate: float = 3.0e-4
    n_steps: int = 512
    batch_size: int = 128
    gamma: float = 0.995
    gae_lambda: float = 0.95
    network_width: int = 64


@dataclass
class RewardConfig:
    on_time_completion: float = 10.0
    tardy_completion: float = 1.0
    deadline_miss: float = -4.0
    floor_violation: float = -25.0
    physical_depletion: float = -75.0
    elapsed_hour: float = -0.02
    terminal_unfinished: float = -3.0


@dataclass
class DockWiseConfig:
    robot: RobotConfig = field(default_factory=RobotConfig)
    station: StationConfig = field(default_factory=StationConfig)
    tasks: TaskConfig = field(default_factory=TaskConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _update_dataclass(instance: T, values: dict[str, Any]) -> T:
    valid = {item.name for item in fields(instance)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Unknown configuration keys for {type(instance).__name__}: {sorted(unknown)}")
    for name, value in values.items():
        current = getattr(instance, name)
        if is_dataclass(current):
            _update_dataclass(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(instance, name, tuple(value))
        else:
            setattr(instance, name, value)
    return instance


def load_config(path: str | Path | None = None) -> DockWiseConfig:
    config = DockWiseConfig()
    if path is None:
        return config
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _update_dataclass(config, raw)

