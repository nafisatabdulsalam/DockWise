from __future__ import annotations

from gymnasium.utils.env_checker import check_env

from dockwise.config import DockWiseConfig
from dockwise.env import DockWiseEnv


def test_gymnasium_contract() -> None:
    config = DockWiseConfig()
    config.simulation.episode_hours = 0.05
    env = DockWiseEnv(config, filter_mode="fixed")
    check_env(env, skip_render_check=True)

