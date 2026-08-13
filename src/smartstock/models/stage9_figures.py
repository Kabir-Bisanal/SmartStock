"""Focused Stage 9 model-selection and final-test figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "mean_28": "28-Day Mean",
    "ridge_global_v1": "Global Ridge",
    "hist_gradient_boosting_base": "HistGradientBoosting",
    "xgb_tune_03": "XGBoost (selected config)",
}
MODEL_COLORS = {
    "mean_28": "#4c78a8",
    "ridge_global_v1": "#f58518",
    "hist_gradient_boosting_base": "#54a24b",
    "xgb_tune_03": "#e45756",
}


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path.as_posix()


def _overall_validation(metrics: pd.DataFrame, horizon: str) -> pd.DataFrame:
    return metrics[
        metrics["evaluation_split"].eq("validation")
        & metrics["fold"].eq("combined")
        & metrics["horizon"].eq(horizon)
        & metrics["segment_type"].eq("overall")
    ].copy()


def _bar_metric(
    metrics: pd.DataFrame, metric: str, title: str, ylabel: str, path: Path
) -> str:
    frame = _overall_validation(metrics, "days_1_30_daily").sort_values(metric)
    labels = [MODEL_LABELS.get(name, name) for name in frame["model_name"]]
    colors = [MODEL_COLORS.get(name, "#777777") for name in frame["model_name"]]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(labels, frame[metric], color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, frame[metric], strict=True):
        display = f"{100 * value:.1f}%" if metric == "wape" else f"{value:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), display, ha="center", va="bottom", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def create_stage9_figures(
    metrics: pd.DataFrame,
    final_predictions: pd.DataFrame,
    figure_dir: Path,
    selected_model: str,
) -> list[str]:
    """Create ten compact figures covering selection, segments, and the locked test."""

    paths = [
        _bar_metric(metrics, "mae", "Validation: 30-Day Daily MAE", "MAE (units)", figure_dir / "01_validation_mae.png"),
        _bar_metric(metrics, "wape", "Validation: 30-Day Daily WAPE", "WAPE", figure_dir / "02_validation_wape.png"),
        _bar_metric(metrics, "rmsse", "Validation: 30-Day Daily RMSSE", "RMSSE", figure_dir / "03_validation_rmsse.png"),
    ]

    horizon_order = ["day_1_daily", "days_1_7_daily", "days_1_30_daily"]
    horizon_labels = ["Day +1", "Days 1-7", "Days 1-30"]
    frame = metrics[
        metrics["evaluation_split"].eq("validation")
        & metrics["fold"].eq("combined")
        & metrics["segment_type"].eq("overall")
        & metrics["horizon"].isin(horizon_order)
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for model, group in frame.groupby("model_name", observed=True, sort=False):
        values = group.set_index("horizon").reindex(horizon_order)["mae"]
        ax.plot(horizon_labels, values, marker="o", linewidth=2, label=MODEL_LABELS.get(model, model), color=MODEL_COLORS.get(model))
    ax.set_title("Validation MAE by Recursive Forecast Horizon")
    ax.set_ylabel("MAE (units)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    paths.append(_save(fig, figure_dir / "04_horizon_mae.png"))

    fold_order = ["validation_fold_1", "validation_fold_2", "validation_fold_3"]
    fold_labels = ["Fold 1", "Fold 2", "Fold 3"]
    frame = metrics[
        metrics["evaluation_split"].eq("validation")
        & metrics["fold"].isin(fold_order)
        & metrics["horizon"].eq("days_1_30_daily")
        & metrics["segment_type"].eq("overall")
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for model, group in frame.groupby("model_name", observed=True, sort=False):
        values = group.set_index("fold").reindex(fold_order)["mae"]
        ax.plot(fold_labels, values, marker="o", linewidth=2, label=MODEL_LABELS.get(model, model), color=MODEL_COLORS.get(model))
    ax.set_title("30-Day Daily MAE Stability Across Validation Folds")
    ax.set_ylabel("MAE (units)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    paths.append(_save(fig, figure_dir / "05_fold_stability.png"))

    aggregate = _overall_validation(metrics, "days_1_30_aggregate").sort_values("mae")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(
        [MODEL_LABELS.get(name, name) for name in aggregate["model_name"]],
        aggregate["mae"],
        color=[MODEL_COLORS.get(name) for name in aggregate["model_name"]],
    )
    ax.set_title("Validation: 30-Day Aggregate Demand MAE")
    ax.set_ylabel("MAE per item-store total (units)")
    ax.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, aggregate["mae"], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    paths.append(_save(fig, figure_dir / "06_aggregate_30day_mae.png"))

    for index, (segment_type, title) in enumerate(
        (("demand_band", "Demand Band"), ("intermittency_class", "Intermittency Class")),
        start=7,
    ):
        segment = metrics[
            metrics["evaluation_split"].eq("validation")
            & metrics["fold"].eq("combined")
            & metrics["horizon"].eq("days_1_30_daily")
            & metrics["segment_type"].eq(segment_type)
        ].copy()
        pivot = segment.pivot(index="segment_value", columns="model_name", values="mae")
        ordered_models = [name for name in MODEL_LABELS if name in pivot.columns]
        x = np.arange(len(pivot.index))
        width = 0.8 / len(ordered_models)
        fig, ax = plt.subplots(figsize=(9, 5.0))
        for offset, model in enumerate(ordered_models):
            ax.bar(
                x + (offset - (len(ordered_models) - 1) / 2) * width,
                pivot[model],
                width,
                label=MODEL_LABELS[model],
                color=MODEL_COLORS[model],
            )
        ax.set_xticks(x, pivot.index)
        ax.set_title(f"Validation 30-Day Daily MAE by {title}")
        ax.set_ylabel("MAE (units)")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        paths.append(_save(fig, figure_dir / f"{index:02d}_{segment_type}_mae.png"))

    validation = _overall_validation(metrics, "days_1_30_daily")
    validation = validation[validation["model_name"].eq(selected_model)].iloc[0]
    test = metrics[
        metrics["evaluation_split"].eq("locked_final_test")
        & metrics["horizon"].eq("days_1_30_daily")
        & metrics["segment_type"].eq("overall")
    ].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, metric, label in zip(axes, ("mae", "wape"), ("MAE", "WAPE"), strict=True):
        values = [validation[metric], test[metric]]
        bars = ax.bar(["Validation", "Locked test"], values, color=["#4c78a8", "#72b7b2"])
        ax.set_title(f"Selected Model: {label}")
        for bar, value in zip(bars, values, strict=True):
            display = f"{100 * value:.1f}%" if metric == "wape" else f"{value:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), display, ha="center", va="bottom")
        ax.spines[["top", "right"]].set_visible(False)
    paths.append(_save(fig, figure_dir / "09_validation_vs_test.png"))

    series_errors = (
        final_predictions.assign(abs_error=lambda x: (x["actual"] - x["forecast"]).abs())
        .groupby(["store_id", "item_id"], observed=True)["abs_error"]
        .mean()
        .sort_values()
    )
    representative_key = series_errors.index[len(series_errors) // 2]
    example = final_predictions[
        final_predictions["store_id"].eq(representative_key[0])
        & final_predictions["item_id"].eq(representative_key[1])
    ].sort_values("target_date")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(example["target_date"], example["actual"], label="Actual", color="#222222", linewidth=1.8)
    ax.plot(example["target_date"], example["forecast"], label="Forecast", color="#4c78a8", linewidth=2)
    ax.set_title(f"Representative Locked-Test Series: {representative_key[0]} / {representative_key[1]}")
    ax.set_ylabel("Daily units")
    ax.legend(frameon=False)
    ax.tick_params(axis="x", rotation=30)
    ax.spines[["top", "right"]].set_visible(False)
    paths.append(_save(fig, figure_dir / "10_representative_test_series.png"))
    return paths
