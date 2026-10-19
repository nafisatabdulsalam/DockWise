from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from dockwise.config import DockWiseConfig
from dockwise.types import Route, Scenario, Task


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_routes(path: str | Path | None = None) -> dict[int, Route]:
    route_path = Path(path) if path else PROJECT_ROOT / "data" / "maps" / "routes.json"
    with route_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(item["destination_id"]): Route(**item) for item in raw["routes"]}


def _synthetic_solar(seed: int, hours: int, peak_w: float) -> np.ndarray:
    """Hourly, explicitly synthetic PV with autocorrelated daily cloud attenuation."""
    rng = np.random.default_rng(seed)
    result = np.zeros(hours, dtype=np.float64)
    cloud = 0.75
    for hour_index in range(hours):
        local_hour = hour_index % 24
        cloud = float(np.clip(0.82 * cloud + 0.18 * rng.beta(5.0, 2.0), 0.12, 1.0))
        if 6 <= local_hour <= 18:
            clear = math.sin(math.pi * (local_hour - 6) / 12.0) ** 1.35
            result[hour_index] = peak_w * clear * cloud
    return result


class WeatherRepository:
    """Read the immutable processed PVGIS cache and provide contiguous blocks."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.suffix == ".parquet":
            self.frame = pd.read_parquet(self.path)
        else:
            self.frame = pd.read_csv(self.path)
        self.frame["timestamp_local"] = pd.to_datetime(self.frame["timestamp_local"])
        self.frame = self.frame.sort_values("timestamp_local").reset_index(drop=True)

    def available_years(self) -> tuple[int, ...]:
        return tuple(sorted(int(year) for year in self.frame["timestamp_local"].dt.year.unique()))

    def block(self, start: str | datetime, episode_hours: int, context_hours: int = 24) -> tuple[np.ndarray, dict]:
        start_ts = pd.Timestamp(start)
        first = start_ts - pd.Timedelta(hours=context_hours)
        last = start_ts + pd.Timedelta(hours=episode_hours + 4)
        selection = self.frame[
            (self.frame["timestamp_local"] >= first) & (self.frame["timestamp_local"] < last)
        ].copy()
        expected = context_hours + episode_hours + 4
        if len(selection) != expected:
            raise ValueError(f"Weather block {start_ts} needs {expected} consecutive hours; found {len(selection)}")
        if selection["timestamp_local"].diff().dropna().ne(pd.Timedelta(hours=1)).any():
            raise ValueError(f"Weather block {start_ts} contains a timestamp gap")
        metadata = {
            "weather_start": str(start_ts),
            "weather_year": int(start_ts.year),
            "weather_file": str(self.path),
            "low_solar_48h": bool(selection.iloc[context_hours : context_hours + 48]["low_solar"].any())
            if "low_solar" in selection
            else False,
        }
        return selection["pv_power_w"].to_numpy(dtype=np.float64), metadata


class ScenarioFactory:
    def __init__(self, config: DockWiseConfig, routes: dict[int, Route] | None = None):
        self.config = config
        self.routes = routes or load_routes()

    def tasks(self, seed: int) -> list[Task]:
        cfg = self.config.tasks
        sim = self.config.simulation
        rng = np.random.default_rng(seed)
        tasks: list[Task] = []
        task_id = 0
        for day in range(math.ceil(sim.episode_hours / 24)):
            window_start = day * 86400 + cfg.workday_start_hour * 3600
            window_end = min(
                day * 86400 + cfg.workday_end_hour * 3600,
                sim.episode_hours * 3600,
            )
            cursor = float(window_start)
            while cfg.arrival_rate_per_hour > 0:
                cursor += float(rng.exponential(3600.0 / cfg.arrival_rate_per_hour))
                if cursor >= window_end:
                    break
                destination_id = int(rng.integers(0, len(self.routes)))
                route = self.routes[destination_id]
                payload = self._payload_for_arrival(rng, cursor, task_id)
                outbound_s = route.distance_m / self.config.robot.speed_mps
                nominal_service_s = cfg.pickup_time_s + outbound_s + cfg.unload_time_s + outbound_s
                delivery_nominal_s = cfg.pickup_time_s + outbound_s + cfg.unload_time_s
                deadline_s = cursor + cfg.deadline_multiplier * delivery_nominal_s
                tasks.append(
                    Task(
                        task_id=task_id,
                        arrival_s=cursor,
                        deadline_s=deadline_s,
                        destination_id=destination_id,
                        payload_kg=payload,
                        nominal_service_s=nominal_service_s,
                    )
                )
                task_id += 1
        return tasks

    def _payload_for_arrival(self, rng: np.random.Generator, arrival_s: float, task_id: int) -> float:
        payloads = np.asarray(self.config.tasks.payloads_kg, dtype=np.float64)
        if self.config.tasks.demand_mode == "stable":
            probabilities = np.asarray([0.40, 0.40, 0.20])[: len(payloads)]
        else:
            # Clock-time regimes are exogenous and cannot shift with policy completion count.
            local_hour = int(arrival_s // 3600) % 24
            day = int(arrival_s // 86400)
            heavy_period = ((local_hour >= 12) ^ (day % 2 == 1))
            probabilities = np.asarray([0.15, 0.35, 0.50] if heavy_period else [0.65, 0.25, 0.10])[: len(payloads)]
        probabilities = probabilities / probabilities.sum()
        return float(rng.choice(payloads, p=probabilities))

    def create(
        self,
        seed: int,
        *,
        start: str = "2020-01-01 00:00:00+01:00",
        weather: WeatherRepository | None = None,
        forecast_bias: float = 0.0,
        crr_multiplier: float = 1.0,
        auxiliary_multiplier: float = 1.0,
        capacity_multiplier: float = 1.0,
        scenario_id: str | None = None,
    ) -> Scenario:
        hours = self.config.simulation.episode_hours
        if weather is None:
            solar = _synthetic_solar(
                seed + 100_003,
                24 + math.ceil(hours) + 4,
                self.config.station.pv_peak_w,
            )
            metadata = {"label": "synthetic solar; historical conclusions pending"}
            source = "synthetic"
        else:
            solar, metadata = weather.block(start, hours, 24)
            solar = solar * (self.config.station.pv_peak_w / 30.0)
            metadata["pvgis_reference_peak_w"] = 30.0
            metadata["scenario_pv_peak_w"] = self.config.station.pv_peak_w
            source = "PVGIS"
        return Scenario(
            scenario_id=scenario_id or f"scenario-{seed}-{pd.Timestamp(start).strftime('%Y%m%d')}",
            start_timestamp=str(start),
            tasks=self.tasks(seed),
            solar_hourly_w=solar,
            forecast_bias=forecast_bias,
            crr_multiplier=crr_multiplier,
            auxiliary_multiplier=auxiliary_multiplier,
            capacity_multiplier=capacity_multiplier,
            source=source,
            metadata={"task_seed": seed, **metadata},
        )


def clone_scenario(scenario: Scenario) -> Scenario:
    """Policies receive independent mutable task records but identical exogenous streams."""
    return replace(scenario, tasks=[replace(task) for task in scenario.tasks], solar_hourly_w=scenario.solar_hourly_w.copy())


def nonoverlapping_starts(
    weather: WeatherRepository,
    years: Iterable[int],
    count: int,
    episode_hours: int = 72,
) -> list[str]:
    frame = weather.frame
    candidates = frame[
        frame["timestamp_local"].dt.year.isin(list(years))
        & frame["timestamp_local"].dt.hour.eq(0)
    ]["timestamp_local"].tolist()
    usable: list[pd.Timestamp] = []
    last: pd.Timestamp | None = None
    for candidate in candidates:
        if last is None or candidate >= last + pd.Timedelta(hours=episode_hours):
            try:
                weather.block(candidate, episode_hours)
            except ValueError:
                continue
            usable.append(candidate)
            last = candidate
    if len(usable) < count:
        raise ValueError(f"Only {len(usable)} nonoverlapping weather blocks available; requested {count}")
    # Deterministic, season-spanning selection over the chronological candidate list.
    indices = np.linspace(0, len(usable) - 1, count, dtype=int)
    return [str(usable[index]) for index in indices]

