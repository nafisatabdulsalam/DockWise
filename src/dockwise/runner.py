from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from dockwise.config import DockWiseConfig
from dockwise.policies.base import Policy
from dockwise.sim import FastSimulator
from dockwise.types import Scenario


@dataclass
class EpisodeRun:
    policy: Policy
    filter_mode: str
    scenario: Scenario
    condition: str = "primary"
    training_seed: int | None = None


def run_episode(run: EpisodeRun, config: DockWiseConfig) -> dict[str, Any]:
    backend = FastSimulator(config, run.scenario, run.filter_mode)
    inference_ns = 0
    decisions = 0
    observation = backend.observe()
    while not (backend.terminated or backend.truncated):
        masks = backend.action_masks()
        started = time.perf_counter_ns()
        action = run.policy.act(observation, masks, backend)
        inference_ns += time.perf_counter_ns() - started
        decisions += 1
        observation, _, _, _, _ = backend.step(action)
    result = backend.results()
    result.update(
        {
            "policy": run.policy.name,
            "filter_mode": run.filter_mode,
            "condition": run.condition,
            "training_seed": run.training_seed,
            "task_seed": run.scenario.metadata.get("task_seed"),
            "weather_start": run.scenario.metadata.get("weather_start", run.scenario.start_timestamp),
            "weather_year": run.scenario.metadata.get("weather_year"),
            "low_solar_48h": run.scenario.metadata.get("low_solar_48h", False),
            "inference_time_ms": inference_ns / 1e6,
            "mean_inference_us": inference_ns / max(decisions, 1) / 1e3,
        }
    )
    return result


class CompletionManifest:
    """Crash-safe per-run checkpointing; completed run IDs are never repeated."""

    def __init__(self, path: str | Path, metadata: dict[str, Any]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                self.data = json.load(handle)
        else:
            self.data = {"metadata": metadata, "completed": [], "errors": {}, "excluded": {}}
            self.save()

    def contains(self, run_id: str) -> bool:
        return run_id in self.data["completed"]

    def complete(self, run_id: str) -> None:
        if run_id not in self.data["completed"]:
            self.data["completed"].append(run_id)
        self.data["errors"].pop(run_id, None)
        self.save()

    def error(self, run_id: str, message: str) -> None:
        self.data["errors"][run_id] = message
        self.save()

    def save(self) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)


def append_result(path: str | Path, result: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([result])
    if output.exists():
        prior = pd.read_parquet(output)
        new = pd.concat([prior, new], ignore_index=True)
    temporary = output.with_suffix(".tmp.parquet")
    new.to_parquet(temporary, index=False)
    temporary.replace(output)

