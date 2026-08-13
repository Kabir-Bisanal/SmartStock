"""Validation-only residual calibration for SmartStock inventory uncertainty."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SERIES_KEYS = ["store_id", "item_id"]


def _residual_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Calculate actual-minus-forecast mean, sample deviation, RMSE, and count."""

    grouped = frame.groupby(group_columns, observed=True, sort=True)["residual"]
    summary = grouped.agg(
        residual_mean="mean",
        residual_std="std",
        calibration_count="size",
    ).reset_index()
    rmse = grouped.apply(lambda values: float(np.sqrt(np.mean(np.square(values))))).rename(
        "residual_rmse"
    ).reset_index()
    return summary.merge(rmse, on=group_columns, how="left", validate="one_to_one")


def calibrate_validation_errors(
    validation_predictions_path: Path,
    *,
    minimum_series_observations: int = 60,
    locked_test_start: str = "2016-04-23",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build item-store uncertainty with department-store and global fallbacks."""

    predictions = pd.read_csv(
        validation_predictions_path,
        usecols=[
            "fold",
            "baseline_name",
            "store_id",
            "item_id",
            "dept_id",
            "demand_band",
            "intermittency_class",
            "target_date",
            "forecast",
            "actual",
        ],
        parse_dates=["target_date"],
        dtype={
            "baseline_name": "string",
            "store_id": "string",
            "item_id": "string",
            "dept_id": "string",
            "demand_band": "string",
            "intermittency_class": "string",
            "forecast": "float64",
            "actual": "float64",
        },
    )
    selected = predictions[predictions["baseline_name"].eq("mean_28")].copy()
    if selected.empty:
        raise ValueError("No mean_28 validation predictions were found for calibration.")
    if selected["target_date"].ge(pd.Timestamp(locked_test_start)).any():
        raise ValueError("Locked final-test observations entered uncertainty calibration.")
    selected["residual"] = selected["actual"] - selected["forecast"]

    series = _residual_summary(selected, SERIES_KEYS)
    static_metadata = selected[[*SERIES_KEYS, "dept_id", "demand_band"]].drop_duplicates()
    if static_metadata.duplicated(SERIES_KEYS).any():
        raise ValueError("Department or demand-band metadata changed by item-store series.")
    latest_intermittency = (
        selected.sort_values("target_date", kind="stable")
        .drop_duplicates(SERIES_KEYS, keep="last")[[*SERIES_KEYS, "intermittency_class"]]
    )
    metadata = static_metadata.merge(
        latest_intermittency, on=SERIES_KEYS, how="left", validate="one_to_one"
    )
    series = series.merge(metadata, on=SERIES_KEYS, how="left", validate="one_to_one")
    department_store = _residual_summary(selected, ["store_id", "dept_id"]).rename(
        columns={
            "residual_mean": "fallback_residual_mean",
            "residual_std": "fallback_residual_std",
            "residual_rmse": "fallback_residual_rmse",
            "calibration_count": "fallback_count",
        }
    )
    series = series.merge(
        department_store, on=["store_id", "dept_id"], how="left", validate="many_to_one"
    )
    global_values = selected["residual"].to_numpy(dtype="float64")
    global_summary = {
        "residual_mean": float(global_values.mean()),
        "residual_std": float(global_values.std(ddof=1)),
        "residual_rmse": float(np.sqrt(np.mean(np.square(global_values)))),
        "calibration_count": int(len(global_values)),
    }

    enough = series["calibration_count"].ge(int(minimum_series_observations)) & series[
        "residual_std"
    ].gt(0)
    fallback_valid = series["fallback_count"].gt(1) & series["fallback_residual_std"].gt(0)
    series["calibration_level"] = np.where(
        enough, "item_store", np.where(fallback_valid, "department_store", "global")
    )
    for metric in ("residual_mean", "residual_std", "residual_rmse", "calibration_count"):
        fallback_column = f"fallback_{metric.replace('calibration_', '')}" if metric != "calibration_count" else "fallback_count"
        values = series[metric].where(enough, series[fallback_column].where(fallback_valid, global_summary[metric]))
        series[f"selected_{metric}"] = values
    series = series.rename(
        columns={
            "selected_residual_mean": "residual_mean_used",
            "selected_residual_std": "sigma_daily",
            "selected_residual_rmse": "residual_rmse_used",
            "selected_calibration_count": "calibration_count_used",
        }
    )
    series["residual_definition"] = "actual_minus_forecast"
    series["calibration_source"] = "validation_only_mean_28"
    output_columns = [
        "item_id",
        "store_id",
        "dept_id",
        "demand_band",
        "intermittency_class",
        "calibration_level",
        "calibration_count_used",
        "residual_mean_used",
        "sigma_daily",
        "residual_rmse_used",
        "residual_definition",
        "calibration_source",
    ]
    result = series[output_columns].sort_values(SERIES_KEYS, kind="stable").reset_index(drop=True)
    if result.duplicated(SERIES_KEYS).any() or result["sigma_daily"].isna().any():
        raise RuntimeError("Uncertainty calibration did not produce one usable row per series.")
    audit = {
        "baseline": "mean_28",
        "residual_definition": "actual - forecast",
        "validation_rows": len(selected),
        "date_min": selected["target_date"].min().date().isoformat(),
        "date_max": selected["target_date"].max().date().isoformat(),
        "final_test_rows_used": 0,
        "series": len(result),
        "minimum_series_observations": int(minimum_series_observations),
        "calibration_level_counts": result["calibration_level"].value_counts().to_dict(),
        "global_fallback": global_summary,
    }
    return result, audit
