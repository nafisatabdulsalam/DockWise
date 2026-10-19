from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta


def _cluster_bootstrap_difference(
    frame: pd.DataFrame,
    first: str,
    second: str,
    metric: str,
    *,
    draws: int = 4000,
    seed: int = 20261019,
) -> dict[str, float]:
    selection = frame[frame["policy"].isin([first, second])].copy()
    # Average RL training seeds within scenario; seeds remain visible in seed-variability outputs.
    grouped = selection.groupby(["weather_block", "task_seed", "policy"], dropna=False)[metric].mean().unstack("policy")
    grouped = grouped.dropna(subset=[first, second])
    grouped["difference"] = grouped[first] - grouped[second]
    block_means = grouped.groupby(level="weather_block")["difference"].mean()
    if block_means.empty:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "blocks": 0}
    rng = np.random.default_rng(seed)
    values = block_means.to_numpy()
    samples = np.mean(rng.choice(values, size=(draws, len(values)), replace=True), axis=1)
    return {
        "estimate": float(values.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "blocks": int(len(values)),
    }


def _jeffreys_interval(events: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    return (
        float(beta.ppf(alpha / 2, events + 0.5, total - events + 0.5)),
        float(beta.ppf(1 - alpha / 2, events + 0.5, total - events + 0.5)),
    )


def analyse(results_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    results_path = Path(results_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(results_path)
    primary = frame[frame["condition"] == "primary"].copy()
    rows = []
    for policy, group in frame.groupby(["condition", "policy"], dropna=False):
        condition, policy_name = policy
        risk_count = int(group["episode_risk"].sum())
        risk_low, risk_high = _jeffreys_interval(risk_count, len(group))
        rows.append(
            {
                "condition": condition,
                "policy": policy_name,
                "episodes": len(group),
                "on_time_rate_mean": float(group["on_time_rate"].mean()),
                "on_time_rate_sd": float(group["on_time_rate"].std()),
                "episode_risk": risk_count / len(group),
                "risk_ci_low": risk_low,
                "risk_ci_high": risk_high,
                "failed_return_rate": float(group["failed_return"].mean()),
                "floor_violation_rate": float(group["robot_floor_violation"].mean()),
                "mean_energy_per_completion_wh": float(group["robot_energy_per_completed_wh"].mean()),
                "mean_filter_exclusion_rate": float(group["filter_exclusion_rate"].mean()),
                "mean_inference_us": float(group["mean_inference_us"].mean()),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["condition", "policy"])
    summary.to_csv(output / "policy_summary.csv", index=False)

    contrasts = {
        "H1_robot_adaptive_minus_model_fixed": _cluster_bootstrap_difference(
            primary, "robot_adaptive", "model_fixed", "on_time_rate"
        ),
        "H2_joint_minus_robot_adaptive": _cluster_bootstrap_difference(
            primary, "joint_adaptive", "robot_adaptive", "on_time_rate"
        ),
        "H4_ppo_adaptive_minus_joint_model": _cluster_bootstrap_difference(
            primary, "ppo_adaptive", "joint_adaptive", "on_time_rate"
        ),
        "H4_ppo_fixed_minus_fixed_model": _cluster_bootstrap_difference(
            primary, "ppo_fixed", "model_fixed", "on_time_rate"
        ),
    }
    stable = frame[frame["condition"] == "stable_abundant"]
    contrasts["H3_joint_minus_fixed_equivalence"] = _cluster_bootstrap_difference(
        stable, "joint_adaptive", "model_fixed", "on_time_rate"
    )
    (output / "paired_contrasts.json").write_text(json.dumps(contrasts, indent=2), encoding="utf-8")

    plot = summary[summary["condition"] == "primary"].sort_values("episode_risk")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(plot["episode_risk"], plot["on_time_rate_mean"])
    for _, row in plot.iterrows():
        ax.annotate(row["policy"], (row["episode_risk"], row["on_time_rate_mean"]), fontsize=8)
    ax.set_xlabel("Episode risk: floor violation or failed return")
    ax.set_ylabel("On-time completions / all arrivals")
    ax.set_title("Completion-risk trade-off (primary held-out scenarios)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "completion_risk.png", dpi=180)
    fig.savefig(output / "completion_risk.pdf")
    plt.close(fig)

    if not primary.empty:
        pivot = primary.groupby(["weather_block", "policy"])["on_time_rate"].mean().unstack()
        reference = "model_fixed" if "model_fixed" in pivot else pivot.columns[0]
        difference = pivot.subtract(pivot[reference], axis=0)
        fig, ax = plt.subplots(figsize=(9, 5))
        difference.boxplot(ax=ax, rot=45)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_ylabel(f"Paired on-time-rate difference vs {reference}")
        fig.tight_layout()
        fig.savefig(output / "paired_differences.png", dpi=180)
        fig.savefig(output / "paired_differences.pdf")
        plt.close(fig)

    rl = frame[frame["policy"].str.startswith("ppo", na=False) & frame["training_seed"].notna()]
    if not rl.empty:
        seed_summary = rl.groupby(["policy", "training_seed"])[["on_time_rate", "episode_risk"]].mean().reset_index()
        seed_summary.to_csv(output / "training_seed_variability.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        for policy, group in seed_summary.groupby("policy"):
            ax.plot(group["training_seed"].astype(str), group["on_time_rate"], marker="o", label=policy)
        ax.set_ylabel("Mean held-out on-time rate")
        ax.set_xlabel("Independent training seed")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "training_seed_variability.png", dpi=180)
        fig.savefig(output / "training_seed_variability.pdf")
        plt.close(fig)

    return {"rows": len(frame), "policies": sorted(frame["policy"].unique()), "contrasts": contrasts}

