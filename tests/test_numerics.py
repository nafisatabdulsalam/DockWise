from __future__ import annotations

from copy import deepcopy

from dockwise.config import DockWiseConfig
from dockwise.policies.baselines import ExpectedValuePolicy
from dockwise.runner import EpisodeRun, run_episode
from dockwise.scenario import ScenarioFactory


def test_timestep_convergence_representative_episode() -> None:
    base = DockWiseConfig()
    base.simulation.episode_hours = 4
    base.tasks.workday_start_hour = 0
    base.tasks.workday_end_hour = 4
    scenario = ScenarioFactory(base).create(123)
    fine = deepcopy(base)
    fine.simulation.physics_interval_s = 2.0
    coarse = deepcopy(base)
    coarse.simulation.physics_interval_s = 10.0
    policy = ExpectedValuePolicy()
    fine_result = run_episode(EpisodeRun(policy, "expected", scenario), fine)
    coarse_result = run_episode(EpisodeRun(policy, "expected", scenario), coarse)
    assert abs(fine_result["robot_energy_consumed_wh"] - coarse_result["robot_energy_consumed_wh"]) < 0.02
    assert fine_result["on_time_completed"] == coarse_result["on_time_completed"]

