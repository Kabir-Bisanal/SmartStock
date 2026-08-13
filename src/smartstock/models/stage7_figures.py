"""Create focused Stage 7 baseline-evaluation figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASELINE_ORDER = ["zero", "last_value", "seasonal_naive_7", "mean_28", "croston_sba"]
BASELINE_LABELS = {
    "zero": "Zero",
    "last_value": "Last value",
    "seasonal_naive_7": "Seasonal naive 7",
    "mean_28": "28-day mean",
    "croston_sba": "Croston-SBA",
}
COLORS = {
    "zero": "#94A3B8",
    "last_value": "#EA580C",
    "seasonal_naive_7": "#2563EB",
    "mean_28": "#16A34A",
    "croston_sba": "#7C3AED",
}


def create_stage7_figures(metrics: pd.DataFrame, figure_dir: Path) -> list[dict[str, Any]]:
    """Create eight non-decorative figures from validation results only."""

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

    primary = metrics[
        metrics["fold"].eq("combined")
        & metrics["horizon"].eq("days_1_30_daily")
        & metrics["segment_type"].eq("overall")
    ].set_index("baseline_name").reindex(BASELINE_ORDER)
    labels = [BASELINE_LABELS[name] for name in BASELINE_ORDER]
    colors = [COLORS[name] for name in BASELINE_ORDER]

    # 1-4: approved validation metrics on the common 30-day daily comparison.
    for number, metric, ylabel, title in [
        (1, "mae", "Units", "30-day daily MAE by baseline"),
        (2, "rmsse", "Mean series RMSSE", "30-day daily RMSSE by baseline"),
        (3, "wape", "WAPE", "30-day daily WAPE by baseline"),
        (4, "bias", "Mean forecast − actual", "30-day daily forecast bias"),
    ]:
        fig, ax = plt.subplots(figsize=(9.4, 4.8))
        values = primary[metric].to_numpy()
        ax.bar(labels, values, color=colors)
        ax.set_title(title + "\nCombined across three validation folds; final test excluded")
        ax.set_ylabel(ylabel)
        if metric == "bias":
            ax.axhline(0, color="#334155", linewidth=1)
        ax.tick_params(axis="x", rotation=18)
        save(fig, f"{number:02d}_{metric}_by_baseline.png", title)

    # 5. Daily MAE by horizon.
    horizon_names = ["day_1_daily", "days_1_7_daily", "days_1_30_daily"]
    horizon_labels = ["Day +1", "Days 1–7", "Days 1–30"]
    horizon = metrics[
        metrics["fold"].eq("combined")
        & metrics["segment_type"].eq("overall")
        & metrics["horizon"].isin(horizon_names)
    ].pivot(index="horizon", columns="baseline_name", values="mae").reindex(horizon_names)
    fig, ax = plt.subplots(figsize=(10.2, 5.1))
    x = np.arange(len(horizon_names))
    width = 0.15
    for index, baseline in enumerate(BASELINE_ORDER):
        ax.bar(x + (index - 2) * width, horizon[baseline], width, label=BASELINE_LABELS[baseline], color=COLORS[baseline])
    ax.set_xticks(x, horizon_labels)
    ax.set_ylabel("Daily MAE (units)")
    ax.set_title("Daily forecast accuracy across horizons\nAll forecasts originate before each 30-day validation window")
    ax.legend(ncols=3, fontsize=8)
    save(fig, "05_daily_mae_by_horizon.png", "Daily MAE by horizon")

    # 6. Demand-band WAPE.
    band_order = ["low", "medium", "high"]
    bands = metrics[
        metrics["fold"].eq("combined")
        & metrics["horizon"].eq("days_1_30_daily")
        & metrics["segment_type"].eq("demand_band")
    ].pivot(index="segment_value", columns="baseline_name", values="wape").reindex(band_order)
    fig, ax = plt.subplots(figsize=(10.2, 5.1))
    x = np.arange(len(band_order))
    for index, baseline in enumerate(BASELINE_ORDER):
        ax.bar(x + (index - 2) * width, bands[baseline], width, label=BASELINE_LABELS[baseline], color=COLORS[baseline])
    ax.set_xticks(x, [name.title() for name in band_order])
    ax.set_ylabel("WAPE")
    ax.set_title("30-day daily WAPE by Stage 4 demand band\nBands are evaluation-only, never predictors")
    ax.legend(ncols=3, fontsize=8)
    save(fig, "06_wape_by_demand_band.png", "WAPE by demand band")

    # 7. Training-history intermittency MAE.
    class_order = ["regular", "intermittent", "highly_intermittent"]
    classes = metrics[
        metrics["fold"].eq("combined")
        & metrics["horizon"].eq("days_1_30_daily")
        & metrics["segment_type"].eq("intermittency_class")
    ].pivot(index="segment_value", columns="baseline_name", values="mae").reindex(class_order)
    fig, ax = plt.subplots(figsize=(10.2, 5.1))
    x = np.arange(len(class_order))
    for index, baseline in enumerate(BASELINE_ORDER):
        ax.bar(x + (index - 2) * width, classes[baseline], width, label=BASELINE_LABELS[baseline], color=COLORS[baseline])
    ax.set_xticks(x, ["Regular", "Intermittent", "Highly intermittent"])
    ax.set_ylabel("Daily MAE (units)")
    ax.set_title("30-day daily MAE by fold-specific training intermittency\nClassifications use no validation targets")
    ax.legend(ncols=3, fontsize=8)
    save(fig, "07_mae_by_intermittency.png", "MAE by training-history intermittency")

    # 8. Fold stability.
    folds = metrics[
        metrics["fold"].isin(["validation_fold_1", "validation_fold_2", "validation_fold_3"])
        & metrics["horizon"].eq("days_1_30_daily")
        & metrics["segment_type"].eq("overall")
    ].pivot(index="fold", columns="baseline_name", values="mae")
    fig, ax = plt.subplots(figsize=(9.8, 5))
    for baseline in BASELINE_ORDER:
        ax.plot(folds.index, folds[baseline], marker="o", linewidth=2, label=BASELINE_LABELS[baseline], color=COLORS[baseline])
    ax.set_xticks(range(len(folds.index)), ["Fold 1", "Fold 2", "Fold 3"])
    ax.set_ylabel("Daily MAE (units)")
    ax.set_title("Baseline stability across rolling validation folds\nDays 1–30 daily evaluation")
    ax.legend(ncols=3, fontsize=8)
    save(fig, "08_fold_stability_mae.png", "Fold stability comparison")
    return figures
