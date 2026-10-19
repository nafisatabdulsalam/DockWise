from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dockwise.config import DockWiseConfig
from dockwise.scenario import ScenarioFactory
from dockwise.sim import FastSimulator
from dockwise.types import Scenario


ScenarioProvider = Callable[[int], Scenario]


class DockWiseEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        config: DockWiseConfig,
        *,
        filter_mode: str,
        scenario_provider: ScenarioProvider | None = None,
        fixed_scenario: Scenario | None = None,
        domain_randomization: bool = True,
    ):
        super().__init__()
        self.config = config
        self.filter_mode = filter_mode
        self.scenario_provider = scenario_provider
        self.fixed_scenario = fixed_scenario
        self.domain_randomization = domain_randomization
        self.factory = ScenarioFactory(config)
        initial_scenario = fixed_scenario or self.factory.create(0)
        self.backend = FastSimulator(config, initial_scenario, filter_mode)
        observation_size = self.backend.observe().vector.shape[0]
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(observation_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.backend.action_count)

    def _sample_scenario(self, episode_seed: int) -> Scenario:
        if self.fixed_scenario is not None:
            return self.fixed_scenario
        if self.scenario_provider is not None:
            return self.scenario_provider(episode_seed)
        kwargs: dict[str, float] = {}
        if self.domain_randomization:
            kwargs = {
                "crr_multiplier": float(self.np_random.uniform(0.85, 1.25)),
                "auxiliary_multiplier": float(self.np_random.uniform(0.85, 1.35)),
                "capacity_multiplier": float(self.np_random.uniform(0.90, 1.0)),
                "forecast_bias": float(self.np_random.uniform(-0.05, 0.15)),
            }
        return self.factory.create(episode_seed, **kwargs)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        if options and "scenario_seed" in options:
            episode_seed = int(options["scenario_seed"])
        scenario = self._sample_scenario(episode_seed)
        self.backend = FastSimulator(self.config, scenario, self.filter_mode)
        observation = self.backend.observe().vector
        return observation, {"scenario_id": scenario.scenario_id, "scenario_seed": episode_seed}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.backend.step(int(action))
        return observation.vector, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self.backend.action_masks()

    def render(self) -> str:
        metrics = self.backend.results()
        return (
            f"t={self.backend.time_s / 3600:.2f}h phase={self.backend.phase.value} "
            f"robot={self.backend.robot_energy_wh:.2f}Wh station={self.backend.station.energy_wh:.2f}Wh "
            f"on_time={metrics['on_time_completed']}/{metrics['arrivals_total']}"
        )

