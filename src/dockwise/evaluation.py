from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from dockwise.config import DockWiseConfig
from dockwise.policies.baselines import ExpectedValuePolicy, ModelBasedPolicy, ThresholdPolicy
from dockwise.runner import CompletionManifest, EpisodeRun, append_result, run_episode
from dockwise.scenario import ScenarioFactory, WeatherRepository, nonoverlapping_starts


def baseline_policies() -> list[tuple[Any, str]]:
    return [
        *((ThresholdPolicy(value), "fixed") for value in (0.10, 0.20, 0.30, 0.40)),
        (ExpectedValuePolicy(), "expected"),
        (ModelBasedPolicy("model_fixed"), "fixed"),
        (ModelBasedPolicy("robot_adaptive"), "robot_adaptive"),
        (ModelBasedPolicy("solar_adaptive"), "solar_adaptive"),
        (ModelBasedPolicy("joint_adaptive"), "joint"),
    ]


def rl_policies(checkpoint_dir: str | Path) -> list[tuple[Any, str, int]]:
    from dockwise.policies.rl import MaskablePPOPolicy

    checkpoint_dir = Path(checkpoint_dir)
    found: list[tuple[Any, str, int]] = []
    for variant, mode in (("fixed", "fixed"), ("adaptive", "joint")):
        for path in sorted(checkpoint_dir.glob(f"ppo_{variant}_seed*.zip")):
            seed = int(path.stem.rsplit("seed", 1)[1])
            found.append((MaskablePPOPolicy(path, f"ppo_{variant}"), mode, seed))
    return found


def condition_config(base: DockWiseConfig, condition: str) -> tuple[DockWiseConfig, dict[str, float]]:
    cfg = deepcopy(base)
    kwargs: dict[str, float] = {}
    if condition == "crr_130":
        kwargs["crr_multiplier"] = 1.30
    elif condition == "aux_150":
        kwargs["auxiliary_multiplier"] = 1.50
    elif condition == "capacity_080":
        kwargs["capacity_multiplier"] = 0.80
    elif condition == "solar_bias_15":
        kwargs["forecast_bias"] = 0.15
    elif condition == "solar_bias_30":
        kwargs["forecast_bias"] = 0.30
    elif condition == "stable_abundant":
        cfg.tasks.demand_mode = "stable"
        cfg.station.pv_peak_w = 60.0
    elif condition == "stable_low_solar":
        cfg.tasks.demand_mode = "stable"
        cfg.station.pv_peak_w = 15.0
    elif condition == "changing_low_solar":
        cfg.tasks.demand_mode = "changing"
        cfg.station.pv_peak_w = 15.0
    elif condition.startswith("arrival_"):
        cfg.tasks.arrival_rate_per_hour = float(condition.split("_", 1)[1])
    elif condition.startswith("deadline_"):
        cfg.tasks.deadline_multiplier = float(condition.split("_", 1)[1])
    elif condition.startswith("pv_"):
        cfg.station.pv_peak_w = float(condition.split("_", 1)[1])
    elif condition.startswith("storage_"):
        cfg.station.capacity_wh = float(condition.split("_", 1)[1])
        cfg.station.initial_soc = 0.50
    elif condition.startswith("window_"):
        cfg.estimator.residual_window = int(condition.split("_", 1)[1])
    elif condition.startswith("horizon_"):
        cfg.simulation.scheduler_horizon_h = int(condition.split("_", 1)[1])
    elif condition != "primary":
        raise ValueError(f"Unknown condition: {condition}")
    return cfg, kwargs


def suite_conditions(suite: str) -> list[str]:
    if suite == "pilot":
        return ["primary", "crr_130", "solar_bias_30", "stable_abundant"]
    if suite == "main":
        return ["primary"]
    if suite == "stress":
        return [
            "crr_130",
            "aux_150",
            "capacity_080",
            "solar_bias_15",
            "solar_bias_30",
            "stable_low_solar",
            "changing_low_solar",
            "arrival_2",
            "arrival_4",
            "arrival_6",
            "deadline_2",
            "deadline_4",
            "stable_abundant",
        ]
    if suite == "sensitivity":
        return [
            *(f"pv_{value}" for value in (15, 30, 60)),
            *(f"storage_{value}" for value in (50, 100, 200)),
            *(f"window_{value}" for value in (20, 50, 100)),
            *(f"horizon_{value}" for value in (1, 2, 4)),
        ]
    if suite == "full":
        return list(dict.fromkeys([*suite_conditions("main"), *suite_conditions("stress"), *suite_conditions("sensitivity")]))
    raise ValueError(f"Unknown suite {suite}")


def evaluate_suite(
    config: DockWiseConfig,
    *,
    suite: str,
    output_path: str | Path,
    manifest_path: str | Path,
    weather_path: str | Path | None,
    checkpoint_dir: str | Path | None,
    weather_blocks: int,
    task_seeds: int,
    include_rl: bool,
) -> dict[str, Any]:
    repository = WeatherRepository(weather_path) if weather_path and Path(weather_path).exists() else None
    if repository:
        starts = nonoverlapping_starts(repository, (2022, 2023), weather_blocks, config.simulation.episode_hours)
    else:
        starts = [f"2022-01-{1 + 3 * index:02d} 00:00:00+01:00" for index in range(weather_blocks)]
    policies: list[tuple[Any, str, int | None]] = [(policy, mode, None) for policy, mode in baseline_policies()]
    if include_rl and checkpoint_dir:
        policies.extend(rl_policies(checkpoint_dir))
    conditions = suite_conditions(suite)
    manifest = CompletionManifest(
        manifest_path,
        {
            "suite": suite,
            "conditions": conditions,
            "weather_blocks": weather_blocks,
            "task_seeds_per_block": task_seeds,
            "policies": [policy.name for policy, _, _ in policies],
            "matched_scenarios": True,
            "future_arrivals_hidden": True,
        },
    )
    attempted = completed = failed = 0
    for condition in conditions:
        cfg, condition_kwargs = condition_config(config, condition)
        factory = ScenarioFactory(cfg)
        for block_index, start in enumerate(starts):
            for task_seed in range(task_seeds):
                scenario_seed = 10_000 * block_index + task_seed
                scenario = factory.create(
                    scenario_seed,
                    start=start,
                    weather=repository,
                    scenario_id=f"{condition}-w{block_index:02d}-t{task_seed:02d}",
                    **condition_kwargs,
                )
                for policy, mode, training_seed in policies:
                    run_id = f"{condition}|{block_index}|{task_seed}|{policy.name}|{training_seed}"
                    if manifest.contains(run_id):
                        continue
                    attempted += 1
                    try:
                        result = run_episode(
                            EpisodeRun(policy, mode, scenario, condition, training_seed),
                            cfg,
                        )
                        result["weather_block"] = block_index
                        append_result(output_path, result)
                        manifest.complete(run_id)
                        completed += 1
                    except Exception as error:  # preserved for exact resume and audit
                        manifest.error(run_id, f"{type(error).__name__}: {error}")
                        failed += 1
    summary = {"attempted": attempted, "completed": completed, "failed": failed, "output": str(output_path)}
    Path(manifest_path).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

