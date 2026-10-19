from __future__ import annotations

from typing import Protocol

import numpy as np

from dockwise.types import ObservationView


class SchedulingBackend(Protocol):
    """Shared high-level interface implemented by fast simulation and ROS bridge."""

    @property
    def action_count(self) -> int: ...

    def observe(self) -> ObservationView: ...

    def action_masks(self) -> np.ndarray: ...

    def step(self, action: int) -> tuple[ObservationView, float, bool, bool, dict]: ...

