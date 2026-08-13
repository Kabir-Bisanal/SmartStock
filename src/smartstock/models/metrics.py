"""Approved SmartStock forecast metrics with safe zero/scale handling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def basic_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, WAPE, and consistently signed bias."""

    actual_values = np.asarray(actual, dtype="float64")
    forecast_values = np.asarray(forecast, dtype="float64")
    if actual_values.shape != forecast_values.shape or actual_values.size == 0:
        raise ValueError("Actual and forecast arrays must be non-empty and have identical shapes.")
    error = forecast_values - actual_values
    denominator = actual_values.sum()
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "wape": float(np.abs(error).sum() / denominator) if denominator != 0 else float("nan"),
        "bias": float(np.mean(error)),
        "aggregate_bias": float(error.sum()),
        "actual_sum": float(denominator),
        "forecast_sum": float(forecast_values.sum()),
        "observations": int(actual_values.size),
    }


def rmsse_scale(training_actual: np.ndarray | list[float]) -> float:
    """Return the M5-style mean squared one-step naive training error."""

    values = np.asarray(training_actual, dtype="float64")
    if values.ndim != 1 or values.size < 2:
        return float("nan")
    scale = float(np.mean(np.square(np.diff(values))))
    return scale if scale > 0 else float("nan")


def mean_series_rmsse(
    frame: pd.DataFrame,
    *,
    aggregate_horizon_days: int | None = None,
) -> tuple[float, int, int]:
    """Average per-series RMSSE, equally weighting defined series.

    For daily evaluation, each series' horizon RMSE is divided by its training-only
    naive scale. For aggregate horizons, each total-demand error is divided by
    ``sqrt(horizon_days * scale)``. Undefined zero-scale series are excluded and
    counted rather than assigned an arbitrary denominator.
    """

    values: list[float] = []
    undefined = 0
    group_keys = ["fold", "store_id", "item_id"] if "fold" in frame.columns else ["store_id", "item_id"]
    for _, group in frame.groupby(group_keys, observed=True, sort=False):
        scale = float(group["rmsse_scale"].iloc[0])
        if not np.isfinite(scale) or scale <= 0:
            undefined += 1
            continue
        if aggregate_horizon_days is None:
            mse = float(np.mean(np.square(group["forecast"].to_numpy() - group["actual"].to_numpy())))
            values.append(float(np.sqrt(mse / scale)))
        else:
            total_error = float(group["forecast"].sum() - group["actual"].sum())
            values.append(float(abs(total_error) / np.sqrt(aggregate_horizon_days * scale)))
    return (float(np.mean(values)) if values else float("nan"), len(values), undefined)


def evaluate_prediction_group(
    frame: pd.DataFrame,
    *,
    aggregate_horizon_days: int | None = None,
) -> dict[str, Any]:
    """Evaluate daily rows or per-series horizon totals using approved metrics."""

    if aggregate_horizon_days is None:
        base = basic_metrics(frame["actual"].to_numpy(), frame["forecast"].to_numpy())
    else:
        totals = (
            frame.groupby(["fold", "store_id", "item_id"], observed=True, sort=False)
            .agg(actual=("actual", "sum"), forecast=("forecast", "sum"))
            .reset_index()
        )
        base = basic_metrics(totals["actual"].to_numpy(), totals["forecast"].to_numpy())
    rmsse, defined, undefined = mean_series_rmsse(frame, aggregate_horizon_days=aggregate_horizon_days)
    return {
        **base,
        "rmsse": rmsse,
        "rmsse_defined_series": defined,
        "rmsse_undefined_series": undefined,
    }
