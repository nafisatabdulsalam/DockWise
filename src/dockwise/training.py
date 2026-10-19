from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from dockwise.config import DockWiseConfig
from dockwise.env import DockWiseEnv
from dockwise.scenario import ScenarioFactory, WeatherRepository


class LearningCurveCallback(BaseCallback):
    def __init__(self, output_path: Path, interval: int = 512):
        super().__init__()
        self.output_path = output_path
        self.interval = interval
        self.rows: list[dict[str, float]] = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.interval == 0:
            row = {"timesteps": float(self.num_timesteps)}
            for key in ("value_loss", "policy_gradient_loss", "entropy_loss", "approx_kl"):
                value = self.logger.name_to_value.get(f"train/{key}")
                if value is not None:
                    row[key] = float(value)
            self.rows.append(row)
            pd.DataFrame(self.rows).to_csv(self.output_path, index=False)
        return True


def _historical_provider(
    config: DockWiseConfig,
    repository: WeatherRepository,
    years: tuple[int, ...],
    randomize: bool,
):
    frame = repository.frame
    starts = frame[
        frame["timestamp_local"].dt.year.isin(years)
        & frame["timestamp_local"].dt.hour.eq(0)
    ]["timestamp_local"].tolist()
    factory = ScenarioFactory(config)

    def provider(seed: int):
        rng = np.random.default_rng(seed)
        start = starts[int(rng.integers(0, len(starts)))]
        kwargs: dict[str, float] = {}
        if randomize:
            kwargs = {
                "crr_multiplier": float(rng.uniform(0.85, 1.25)),
                "auxiliary_multiplier": float(rng.uniform(0.85, 1.35)),
                "capacity_multiplier": float(rng.uniform(0.90, 1.0)),
                "forecast_bias": float(rng.uniform(-0.05, 0.15)),
            }
        return factory.create(seed, start=str(start), weather=repository, **kwargs)

    return provider


def make_training_env(
    config: DockWiseConfig,
    variant: str,
    weather_path: str | Path | None,
) -> DockWiseEnv:
    provider = None
    if weather_path and Path(weather_path).exists():
        provider = _historical_provider(config, WeatherRepository(weather_path), (2019, 2020), True)
    return DockWiseEnv(
        config,
        filter_mode=variant,
        scenario_provider=provider,
        domain_randomization=provider is None,
    )


def train_one(
    config: DockWiseConfig,
    *,
    variant: str,
    seed: int,
    total_steps: int,
    output_dir: str | Path,
    weather_path: str | Path | None = None,
    resume: bool = True,
    learning_rate: float | None = None,
    network_width: int | None = None,
) -> dict[str, Any]:
    if variant not in {"fixed", "adaptive", "joint"}:
        raise ValueError("RL variant must be fixed or adaptive/joint")
    normalized_variant = "joint" if variant == "adaptive" else variant
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"ppo_{'adaptive' if normalized_variant == 'joint' else normalized_variant}_seed{seed}"
    model_path = output / stem
    metadata_path = output / f"{stem}.metadata.json"
    curve_path = output / f"{stem}.learning.csv"
    env = make_training_env(config, normalized_variant, weather_path)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    lr = learning_rate or config.training.learning_rate
    width = network_width or config.training.network_width
    resumed = bool(resume and model_path.with_suffix(".zip").exists())
    if resumed:
        model = MaskablePPO.load(model_path, env=env, device="cpu")
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=lr,
            n_steps=config.training.n_steps,
            batch_size=config.training.batch_size,
            gamma=config.training.gamma,
            gae_lambda=config.training.gae_lambda,
            policy_kwargs={"net_arch": [width, width]},
            seed=seed,
            device="cpu",
            verbose=0,
        )
    curve = LearningCurveCallback(curve_path)
    checkpoint = CheckpointCallback(
        save_freq=max(config.training.n_steps * 4, 1),
        save_path=str(output / "intermediate"),
        name_prefix=stem,
    )
    started = time.perf_counter()
    model.learn(
        total_timesteps=total_steps,
        callback=CallbackList([curve, checkpoint]),
        reset_num_timesteps=not resumed,
        progress_bar=False,
    )
    wall_s = time.perf_counter() - started
    model.save(model_path)
    metadata = {
        "variant": normalized_variant,
        "seed": seed,
        "requested_additional_steps": total_steps,
        "model_num_timesteps": int(model.num_timesteps),
        "wall_clock_s": wall_s,
        "resumed": resumed,
        "learning_rate": lr,
        "network_width": width,
        "n_steps": config.training.n_steps,
        "batch_size": config.training.batch_size,
        "gamma": config.training.gamma,
        "gae_lambda": config.training.gae_lambda,
        "normalisation": "none; observations are dimensionless bounded features",
        "weather": str(weather_path) if weather_path else "synthetic-labelled",
        "device": "cpu",
        "torch_version": torch.__version__,
        "checkpoint_contains_optimizer_state": True,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    env.close()
    return metadata


def benchmark_training(
    config: DockWiseConfig,
    output_dir: str | Path,
    weather_path: str | Path | None = None,
) -> dict[str, Any]:
    metadata = train_one(
        config,
        variant="fixed",
        seed=3,
        total_steps=config.training.benchmark_steps,
        output_dir=Path(output_dir) / "benchmark",
        weather_path=weather_path,
        resume=False,
    )
    metadata["steps_per_second"] = metadata["requested_additional_steps"] / max(metadata["wall_clock_s"], 1e-9)
    path = Path(output_dir) / "benchmark.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def train_final_models(
    config: DockWiseConfig,
    output_dir: str | Path,
    weather_path: str | Path | None = None,
    steps: int | None = None,
    seeds: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    results = []
    for variant in ("fixed", "joint"):
        for seed in seeds or config.training.seeds:
            results.append(
                train_one(
                    config,
                    variant=variant,
                    seed=seed,
                    total_steps=steps or config.training.final_steps,
                    output_dir=output_dir,
                    weather_path=weather_path,
                    resume=True,
                )
            )
    return results

