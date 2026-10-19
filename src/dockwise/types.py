from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Phase(str, Enum):
    IDLE = "idle"
    PICKUP = "pickup"
    OUTBOUND = "outbound"
    UNLOAD = "unload"
    RETURN = "return"
    CHARGING = "charging"


@dataclass(frozen=True)
class Route:
    destination_id: int
    name: str
    x_m: float
    y_m: float
    distance_m: float
    turns: int
    acceleration_events: int


@dataclass
class Task:
    task_id: int
    arrival_s: float
    deadline_s: float
    destination_id: int
    payload_kg: float
    nominal_service_s: float
    completion_s: float | None = None
    dispatched_s: float | None = None
    missed_recorded: bool = False


@dataclass
class Scenario:
    scenario_id: str
    start_timestamp: str
    tasks: list[Task]
    solar_hourly_w: np.ndarray
    context_hours: int = 24
    forecast_bias: float = 0.0
    crr_multiplier: float = 1.0
    auxiliary_multiplier: float = 1.0
    capacity_multiplier: float = 1.0
    source: str = "synthetic"
    metadata: dict[str, Any] = field(default_factory=dict)

    def solar_w(self, episode_time_s: float) -> float:
        hour = self.context_hours + int(max(0.0, episode_time_s) // 3600)
        return float(self.solar_hourly_w[min(hour, len(self.solar_hourly_w) - 1)])

    def causal_forecast_w(self, origin_s: float, horizon_hours: int) -> np.ndarray:
        """Previous-day same-hour forecast; every source hour precedes origin."""
        origin_hour = int(max(0.0, origin_s) // 3600)
        values: list[float] = []
        for lead in range(1, horizon_hours + 1):
            target_hour = origin_hour + lead
            source_index = self.context_hours + target_hour - 24
            if source_index < 0:
                source_index = 0
            value = float(self.solar_hourly_w[min(source_index, len(self.solar_hourly_w) - 1)])
            values.append(value * (1.0 + self.forecast_bias))
        return np.asarray(values, dtype=np.float64)


@dataclass
class StationFlows:
    pv_wh: float = 0.0
    charger_output_wh: float = 0.0
    charger_loss_wh: float = 0.0
    station_charge_input_wh: float = 0.0
    station_charge_stored_wh: float = 0.0
    station_withdrawn_wh: float = 0.0
    station_discharge_bus_wh: float = 0.0
    storage_loss_wh: float = 0.0
    idle_wh: float = 0.0
    curtailed_wh: float = 0.0
    unmet_bus_wh: float = 0.0


@dataclass
class StepEvents:
    on_time_completed: int = 0
    tardy_completed: int = 0
    newly_missed: int = 0
    robot_floor_violation: bool = False
    station_floor_violation: bool = False
    physical_depletion: bool = False
    failed_return: bool = False


@dataclass
class ObservationView:
    vector: np.ndarray
    candidate_task_ids: tuple[int | None, ...]
    candidate_tasks: tuple[Task | None, ...]
    robot_energy_estimate_wh: float
    station_energy_wh: float
    robot_margin_wh: float
    solar_margin_wh: float
    phase: Phase

