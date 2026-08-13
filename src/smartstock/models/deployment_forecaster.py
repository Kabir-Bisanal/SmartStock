"""Post-evaluation deployment refit for the frozen SmartStock mean-28 forecaster."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from smartstock.models.baseline_evaluation import file_sha256
except ModuleNotFoundError:
    from baseline_evaluation import file_sha256


SERIES_KEYS = ["store_id", "item_id"]
DEPLOYMENT_ARTIFACT_VERSION = 1


def load_observed_demand(
    feature_data_path: Path,
    through_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Load only identity and observed demand through the deployment cutoff."""

    cutoff = pd.Timestamp(through_date)
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        feature_data_path,
        usecols=["date", "store_id", "item_id", "dept_id", "demand_band", "sales"],
        parse_dates=["date"],
        dtype={
            "store_id": "string",
            "item_id": "string",
            "dept_id": "string",
            "demand_band": "string",
            "sales": "float32",
        },
        chunksize=100_000,
    ):
        selected = chunk[chunk["date"].le(cutoff)]
        if not selected.empty:
            frames.append(selected)
    if not frames:
        raise ValueError("No observed demand was loaded for the deployment refit.")
    data = pd.concat(frames, ignore_index=True).sort_values(
        [*SERIES_KEYS, "date"], kind="stable"
    )
    if data["date"].max() != cutoff:
        raise ValueError(
            f"Deployment history ends {data['date'].max().date()}, expected {cutoff.date()}."
        )
    return data.reset_index(drop=True)


def fit_deployment_mean28(
    observed_demand: pd.DataFrame,
    frozen_config: dict[str, Any],
    trained_through: str | pd.Timestamp,
) -> dict[str, Any]:
    """Capture the latest 28 actual observations for each frozen Version 1 series."""

    if (
        frozen_config.get("selected_model_family") != "mean_28"
        or int(frozen_config.get("model_parameters", {}).get("window_days", -1)) != 28
    ):
        raise ValueError("Stage 10 deployment refit requires the frozen mean_28 model.")
    cutoff = pd.Timestamp(trained_through)
    if observed_demand["date"].gt(cutoff).any():
        raise ValueError("Observed demand contains rows after the deployment cutoff.")
    history: dict[tuple[str, str], np.ndarray] = {}
    metadata: dict[tuple[str, str], dict[str, str]] = {}
    for (store_id, item_id), group in observed_demand.groupby(
        SERIES_KEYS, observed=True, sort=True
    ):
        if group["date"].max() != cutoff or len(group) < 28:
            raise ValueError(f"Insufficient cutoff history for {store_id}/{item_id}.")
        key = (str(store_id), str(item_id))
        history[key] = group["sales"].tail(28).to_numpy(dtype="float64")
        metadata[key] = {
            "dept_id": str(group["dept_id"].iloc[-1]),
            "demand_band": str(group["demand_band"].iloc[-1]),
        }
    expected = int(frozen_config.get("expected_series", 300))
    if len(history) != expected:
        raise ValueError(f"Expected {expected} deployment histories; found {len(history)}.")
    return {
        "artifact_version": DEPLOYMENT_ARTIFACT_VERSION,
        "artifact_role": "post_evaluation_deployment_refit",
        "model_name": "mean_28",
        "model_family": "mean_28",
        "window_days": 28,
        "recursive_policy": "append each nonnegative prediction before forecasting the next day",
        "trained_through": cutoff.date().isoformat(),
        "forecast_start": (cutoff + pd.DateOffset(days=1)).date().isoformat(),
        "series_count": len(history),
        "history": history,
        "metadata": metadata,
        "frozen_config_sha256": None,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }


def save_deployment_forecaster(bundle: dict[str, Any], path: Path) -> Path:
    """Serialize the separate deployment artifact without touching evaluation files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_deployment_forecaster(path: Path | str) -> dict[str, Any]:
    """Load and validate the lightweight deployment history bundle."""

    bundle = joblib.load(Path(path))
    required = {
        "artifact_role",
        "model_family",
        "window_days",
        "trained_through",
        "history",
        "metadata",
    }
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("Invalid SmartStock deployment forecaster artifact.")
    if bundle["artifact_role"] != "post_evaluation_deployment_refit":
        raise ValueError("Artifact is not the separate deployment refit.")
    return bundle


def forecast_deployment_demand(
    forecaster: dict[str, Any] | Path | str,
    horizon_days: int = 30,
) -> pd.DataFrame:
    """Generate algorithmic future dates for the calendar-agnostic frozen mean model."""

    if not 1 <= int(horizon_days) <= 30:
        raise ValueError("Deployment horizon_days must be between 1 and 30.")
    bundle = (
        load_deployment_forecaster(forecaster)
        if isinstance(forecaster, (str, Path))
        else forecaster
    )
    origin = pd.Timestamp(bundle["trained_through"])
    rows: list[dict[str, Any]] = []
    for key in sorted(bundle["history"]):
        values = list(np.asarray(bundle["history"][key], dtype="float64"))
        if len(values) != int(bundle["window_days"]):
            raise ValueError(f"Deployment history window is invalid for {key}.")
        for horizon_day in range(1, int(horizon_days) + 1):
            prediction = max(float(np.mean(values[-28:])), 0.0)
            target_date = origin + pd.DateOffset(days=horizon_day)
            rows.append(
                {
                    "item_id": key[1],
                    "store_id": key[0],
                    "dept_id": bundle["metadata"][key]["dept_id"],
                    "demand_band": bundle["metadata"][key]["demand_band"],
                    "forecast_origin": origin,
                    "target_date": target_date,
                    "horizon_day": horizon_day,
                    "forecast": prediction,
                }
            )
            values.append(prediction)
    predictions = pd.DataFrame(rows).sort_values(
        ["target_date", "store_id", "item_id"], kind="stable"
    ).reset_index(drop=True)
    expected = int(bundle["series_count"]) * int(horizon_days)
    if len(predictions) != expected or predictions["forecast"].lt(0).any():
        raise RuntimeError("Deployment forecast dimensions or nonnegative policy failed.")
    if predictions.duplicated(["target_date", "store_id", "item_id"]).any():
        raise RuntimeError("Deployment forecast keys are not unique.")
    return predictions


def create_deployment_artifact(
    feature_data_path: Path,
    frozen_config_path: Path,
    output_path: Path,
    *,
    trained_through: str = "2016-05-22",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refit the exact frozen mean method on all observed demand and serialize it."""

    frozen_config = json.loads(frozen_config_path.read_text(encoding="utf-8"))
    observed = load_observed_demand(feature_data_path, trained_through)
    bundle = fit_deployment_mean28(observed, frozen_config, trained_through)
    bundle["frozen_config_sha256"] = file_sha256(frozen_config_path)
    save_deployment_forecaster(bundle, output_path)
    audit = {
        "evaluation_artifact_overwritten": False,
        "model_family_matches_frozen_config": True,
        "window_matches_frozen_config": True,
        "trained_through": trained_through,
        "series_count": bundle["series_count"],
        "artifact_path": output_path.as_posix(),
        "artifact_size_bytes": output_path.stat().st_size,
        "artifact_sha256": file_sha256(output_path),
    }
    return bundle, audit
