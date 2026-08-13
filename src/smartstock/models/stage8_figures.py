"""Create focused Ridge-versus-baseline figures for SmartStock Stage 8."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RIDGE_COLOR = "#2563EB"
MEAN_COLOR = "#16A34A"
ZERO_COLOR = "#94A3B8"


def create_stage8_figures(comparison: pd.DataFrame, figure_dir: Path) -> list[dict[str, Any]]:
    """Create eight validation-only comparison figures."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    figures: list[dict[str, Any]] = []

    def save(fig: Any, filename: str, title: str) -> None:
        fig.tight_layout()
        path = figure_dir / filename
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures.append({"filename": filename, "title": title})

    primary = comparison[
        comparison["fold"].eq("combined")
        & comparison["horizon"].eq("days_1_30_daily")
        & comparison["segment_type"].eq("overall")
    ].iloc[0]

    # 1. Four principal metrics use separate panels because their scales differ.
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for ax, metric, title in zip(
        axes.flat,
        ["mae", "rmse", "wape", "rmsse"],
        ["MAE", "RMSE", "WAPE", "RMSSE"],
        strict=True,
    ):
        ax.bar(
            ["Ridge", "28-day mean"],
            [primary[metric], primary[f"mean_28_{metric}"]],
            color=[RIDGE_COLOR, MEAN_COLOR],
        )
        ax.set_title(title)
        ax.set_ylabel("Error (lower is better)")
    fig.suptitle("Ridge versus the strongest Stage 7 baseline\nDays 1-30 daily, three validation folds")
    save(fig, "01_ridge_vs_mean28_overall.png", "Ridge versus 28-day mean overall metrics")

    # 2. Fold stability.
    folds = comparison[
        comparison["fold"].ne("combined")
        & comparison["horizon"].eq("days_1_30_daily")
        & comparison["segment_type"].eq("overall")
    ].sort_values("fold")
    x = np.arange(len(folds))
    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.bar(x - 0.18, folds["mae"], 0.36, label="Ridge", color=RIDGE_COLOR)
    ax.bar(x + 0.18, folds["mean_28_mae"], 0.36, label="28-day mean", color=MEAN_COLOR)
    ax.set_xticks(x, ["Fold 1", "Fold 2", "Fold 3"])
    ax.set_ylabel("Daily MAE (units)")
    ax.set_title("30-day daily MAE by validation fold")
    ax.legend()
    save(fig, "02_mae_by_fold.png", "MAE comparison by fold")

    horizons = [
        "day_1_daily",
        "days_1_7_daily",
        "days_1_7_aggregate",
        "days_1_30_daily",
        "days_1_30_aggregate",
    ]
    horizon_labels = ["Day +1", "Days 1-7\ndaily", "7-day\naggregate", "Days 1-30\ndaily", "30-day\naggregate"]
    horizon = comparison[
        comparison["fold"].eq("combined")
        & comparison["segment_type"].eq("overall")
        & comparison["horizon"].isin(horizons)
    ].set_index("horizon").reindex(horizons)

    # 3-4. Horizon comparisons.
    for number, metric, title in [
        (3, "rmsse", "RMSSE across forecast horizons"),
        (4, "wape", "WAPE across forecast horizons"),
    ]:
        fig, ax = plt.subplots(figsize=(10.3, 5.2))
        x = np.arange(len(horizon))
        ax.bar(x - 0.18, horizon[metric], 0.36, label="Ridge", color=RIDGE_COLOR)
        ax.bar(x + 0.18, horizon[f"mean_28_{metric}"], 0.36, label="28-day mean", color=MEAN_COLOR)
        ax.set_xticks(x, horizon_labels)
        ax.set_ylabel(metric.upper())
        ax.set_title(title + "\nFinal test excluded")
        ax.legend()
        save(fig, f"{number:02d}_{metric}_by_horizon.png", title)

    # 5. Signed bias.
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    axes[0].bar(
        ["Ridge", "28-day mean"],
        [primary["bias"], primary["mean_28_bias"]],
        color=[RIDGE_COLOR, MEAN_COLOR],
    )
    axes[0].axhline(0, color="#334155", linewidth=1)
    axes[0].set_ylabel("Mean forecast - actual")
    axes[0].set_title("Primary comparison close-up")
    axes[1].bar(["Zero"], [primary["zero_bias"]], color=ZERO_COLOR)
    axes[1].axhline(0, color="#334155", linewidth=1)
    axes[1].set_title("Sanity-floor context")
    fig.suptitle("30-day daily forecast bias\nPositive is overforecast; negative is underforecast")
    save(fig, "05_forecast_bias.png", "Forecast bias comparison")

    # 6-7. Required product-segment sanity checks.
    for number, segment_type, order, labels, title in [
        (6, "demand_band", ["low", "medium", "high"], ["Low", "Medium", "High"], "MAE by demand band"),
        (
            7,
            "intermittency_class",
            ["regular", "intermittent", "highly_intermittent"],
            ["Regular", "Intermittent", "Highly intermittent"],
            "MAE by training-history intermittency",
        ),
    ]:
        segment = comparison[
            comparison["fold"].eq("combined")
            & comparison["horizon"].eq("days_1_30_daily")
            & comparison["segment_type"].eq(segment_type)
        ].set_index("segment_value").reindex(order)
        fig, ax = plt.subplots(figsize=(10, 5.1))
        x = np.arange(len(segment))
        width = 0.25
        ax.bar(x - width, segment["mae"], width, label="Ridge", color=RIDGE_COLOR)
        ax.bar(x, segment["mean_28_mae"], width, label="28-day mean", color=MEAN_COLOR)
        ax.bar(x + width, segment["zero_mae"], width, label="Zero", color=ZERO_COLOR)
        ax.set_xticks(x, labels)
        ax.set_ylabel("Daily MAE (units)")
        ax.set_title(title + "\nDays 1-30 daily; segments do not influence predictions")
        ax.legend()
        save(fig, f"{number:02d}_mae_by_{segment_type}.png", title)

    # 8. Segment improvement relative to the primary baseline.
    selected = comparison[
        comparison["fold"].eq("combined")
        & comparison["horizon"].eq("days_1_30_daily")
        & comparison["segment_type"].isin(["demand_band", "store_id", "dept_id", "intermittency_class"])
    ].copy()
    selected["label"] = selected["segment_type"] + ": " + selected["segment_value"]
    selected = selected.sort_values("mae_improvement_vs_mean_28_pct")
    colors = np.where(selected["mae_improvement_vs_mean_28_pct"].ge(0), RIDGE_COLOR, "#DC2626")
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.barh(selected["label"], selected["mae_improvement_vs_mean_28_pct"], color=colors)
    ax.axvline(0, color="#334155", linewidth=1)
    ax.set_xlabel("Ridge MAE improvement versus 28-day mean (%)")
    ax.set_title("Ridge improvement varies by evaluation segment\nPositive values favor Ridge")
    save(fig, "08_mae_improvement_by_segment.png", "Ridge MAE improvement by segment")
    return figures
