from __future__ import annotations

from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from dockwise.sim import FastSimulator
from dockwise.types import ObservationView


class MaskablePPOPolicy:
    def __init__(self, model_path: str | Path, name: str, deterministic: bool = True):
        self.model_path = Path(model_path)
        self.name = name
        self.deterministic = deterministic
        self.model = MaskablePPO.load(self.model_path, device="cpu")

    def act(self, observation: ObservationView, masks: np.ndarray, backend: FastSimulator) -> int:
        action, _ = self.model.predict(
            observation.vector,
            action_masks=masks,
            deterministic=self.deterministic,
        )
        return int(action)

