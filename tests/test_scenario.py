from __future__ import annotations

import numpy as np

from dockwise.config import DockWiseConfig
from dockwise.scenario import ScenarioFactory
from dockwise.types import Scenario


def test_scenario_generation_is_repeatable_and_policy_independent() -> None:
    factory = ScenarioFactory(DockWiseConfig())
    first = factory.create(42)
    second = factory.create(42)
    assert [(t.arrival_s, t.deadline_s, t.destination_id, t.payload_kg) for t in first.tasks] == [
        (t.arrival_s, t.deadline_s, t.destination_id, t.payload_kg) for t in second.tasks
    ]
    assert np.array_equal(first.solar_hourly_w, second.solar_hourly_w)


def test_previous_day_forecast_is_causal() -> None:
    solar = np.arange(24 + 72 + 4, dtype=float)
    scenario = Scenario("causal", "2022-01-01", [], solar, context_hours=24)
    origin_s = 30 * 3600.0
    forecast = scenario.causal_forecast_w(origin_s, 4)
    expected_indices = [24 + 30 + lead - 24 for lead in range(1, 5)]
    assert np.array_equal(forecast, solar[expected_indices])
    assert all(index < 24 + int(origin_s // 3600) for index in expected_indices)


def test_future_arrivals_not_in_observation() -> None:
    from dockwise.sim import FastSimulator

    config = DockWiseConfig()
    scenario = ScenarioFactory(config).create(8)
    backend = FastSimulator(config, scenario)
    assert all(value is None for value in backend.observe().candidate_task_ids)

