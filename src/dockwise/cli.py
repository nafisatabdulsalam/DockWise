from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from dockwise.config import load_config


ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dockwise")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info")
    fetch = sub.add_parser("fetch-data")
    fetch.add_argument("--raw", default=str(ROOT / "data" / "cache" / "pvgis_ilorin_2019_2023_raw.json"))
    fetch.add_argument("--processed", default=str(ROOT / "data" / "cache" / "pvgis_ilorin_2019_2023.parquet"))
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--seed", type=int, default=7)
    bench = sub.add_parser("benchmark-training")
    bench.add_argument("--output", default=str(ROOT / "artifacts" / "checkpoints"))
    train = sub.add_parser("train")
    train.add_argument("--variant", choices=["fixed", "adaptive", "all"], default="all")
    train.add_argument("--steps", type=int)
    train.add_argument("--seed", type=int)
    train.add_argument("--output", default=str(ROOT / "artifacts" / "checkpoints"))
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--suite", choices=["pilot", "main", "stress", "sensitivity", "full"], default="pilot")
    evaluate.add_argument("--weather-blocks", type=int, default=3)
    evaluate.add_argument("--task-seeds", type=int, default=2)
    evaluate.add_argument("--include-rl", action="store_true")
    evaluate.add_argument("--output", default=str(ROOT / "artifacts" / "runs" / "results.parquet"))
    evaluate.add_argument("--manifest", default=str(ROOT / "artifacts" / "runs" / "manifest.json"))
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--results", default=str(ROOT / "artifacts" / "runs" / "results.parquet"))
    analysis.add_argument("--output", default=str(ROOT / "artifacts" / "analysis"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    weather = ROOT / "data" / "cache" / "pvgis_ilorin_2019_2023.parquet"
    if args.command == "info":
        print(json.dumps({"python": sys.version, "platform": platform.platform(), "config": config.to_dict()}, indent=2))
    elif args.command == "fetch-data":
        from dockwise.data import download_pvgis, inspect_cache

        print(json.dumps(download_pvgis(args.raw, args.processed), indent=2, default=str))
        print(json.dumps(inspect_cache(args.processed), indent=2))
    elif args.command == "smoke":
        from dockwise.policies.baselines import ModelBasedPolicy
        from dockwise.runner import EpisodeRun, run_episode
        from dockwise.scenario import ScenarioFactory

        scenario = ScenarioFactory(config).create(args.seed)
        result = run_episode(EpisodeRun(ModelBasedPolicy("joint_adaptive"), "joint", scenario), config)
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "benchmark-training":
        from dockwise.training import benchmark_training

        print(json.dumps(benchmark_training(config, args.output, weather if weather.exists() else None), indent=2))
    elif args.command == "train":
        from dockwise.training import train_final_models, train_one

        steps = args.steps or config.training.final_steps
        if args.variant == "all" and args.seed is None:
            results = train_final_models(config, args.output, weather if weather.exists() else None, steps=steps)
        else:
            variants = ("fixed", "adaptive") if args.variant == "all" else (args.variant,)
            seeds = (args.seed,) if args.seed is not None else config.training.seeds
            results = [
                train_one(
                    config,
                    variant=variant,
                    seed=seed,
                    total_steps=steps,
                    output_dir=args.output,
                    weather_path=weather if weather.exists() else None,
                )
                for variant in variants
                for seed in seeds
            ]
        print(json.dumps(results, indent=2))
    elif args.command == "evaluate":
        from dockwise.evaluation import evaluate_suite

        print(
            json.dumps(
                evaluate_suite(
                    config,
                    suite=args.suite,
                    output_path=args.output,
                    manifest_path=args.manifest,
                    weather_path=weather if weather.exists() else None,
                    checkpoint_dir=ROOT / "artifacts" / "checkpoints",
                    weather_blocks=args.weather_blocks,
                    task_seeds=args.task_seeds,
                    include_rl=args.include_rl,
                ),
                indent=2,
            )
        )
    elif args.command == "analyze":
        from dockwise.analysis import analyse

        print(json.dumps(analyse(args.results, args.output), indent=2))


if __name__ == "__main__":
    main()

