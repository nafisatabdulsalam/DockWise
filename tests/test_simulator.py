from __future__ import annotations

from copy import deepcopy

import numpy as np

from dockwise.config import DockWiseConfig
from dockwise.env import DockWiseEnv
from dockwise.scenario import ScenarioFactory
from dockwise.sim import FastSimulator


def test_mask_has_fallback_and_does_not_use_hidden_payload() -> None:
    config = DockWiseConfig()
    first = ScenarioFactory(config).create(12)
    arrival = first.tasks[0].arrival_s
    backend_a = FastSimulator(config, first, "joint")
    backend_a.time_s = arrival
    observation_a = backend_a.observe().vector
    masks_a = backend_a.action_masks()
    second = deepcopy(first)
    second.tasks[0].payload_kg = 99.0
    backend_b = FastSimulator(config, second, "joint")
    backend_b.time_s = arrival
    assert np.array_equal(observation_a, backend_b.observe().vector)
    assert np.array_equal(masks_a, backend_b.action_masks())
    assert masks_a[backend_a.wait_action]


def test_gymnasium_truncation_and_denominator() -> None:
    config = DockWiseConfig()
    config.simulation.episode_hours = 0.05
    scenario = ScenarioFactory(config).create(4)
    env = DockWiseEnv(config, filter_mode="fixed", fixed_scenario=scenario)
    observation, _ = env.reset(seed=1)
    terminated = truncated = False
    while not (terminated or truncated):
        observation, _, terminated, truncated, info = env.step(env.backend.wait_action)
    assert not terminated
    assert truncated
    assert info["metrics"]["arrivals_total"] == len(scenario.tasks)


def test_physical_depletion_is_termination_and_logs_before_clip() -> None:
    config = DockWiseConfig()
    config.simulation.episode_hours = 1
    config.robot.initial_soc = 0.001
    scenario = ScenarioFactory(config).create(1)
    backend = FastSimulator(config, scenario, "unshielded")
    _, _, terminated, truncated, _ = backend.step(backend.wait_action)
    assert terminated and not truncated
    assert backend.metrics["robot_floor_violation"]
    assert backend.metrics["physical_depletion"]
    assert backend.robot_energy_wh == 0.0


def test_active_task_is_noninterruptible() -> None:
    config = DockWiseConfig()
    scenario = ScenarioFactory(config).create(19)
    backend = FastSimulator(config, scenario, "fixed")
    backend.time_s = scenario.tasks[0].arrival_s
    masks = backend.action_masks()
    action = int(np.flatnonzero(masks[: backend.k])[0])
    backend.step(action)
    assert backend.active_task is not None
    active_id = backend.active_task.task_id
    backend.step(backend.charge_action)  # invalid while active; falls back to continue
    assert backend.active_task is not None
    assert backend.active_task.task_id == active_id
    assert backend.metrics["invalid_actions"] == 1

