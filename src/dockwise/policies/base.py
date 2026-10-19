from __future__ import annotations

from typing import Protocol

import numpy as np

from dockwise.sim import FastSimulator
from dockwise.types import ObservationView


class Policy(Protocol):
    name: str

    def act(self, observation: ObservationView, masks: np.ndarray, backend: FastSimulator) -> int: ...

