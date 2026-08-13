"""Serializable production forecaster and leakage-safe daily forecast interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from smartstock.models.recursive_forecasting import (
        SERIES_KEYS,
        _build_histories,
        build_restricted_future_context,
        clip_nonnegative,
        recursive_demand_features,
    )
    from smartstock.models.ridge_model import prepare_feature_frame
except ModuleNotFoundError:  # Supports direct execution from the models directory.
    from recursive_forecasting import (
        SERIES_KEYS,
        _build_histories,
        build_restricted_future_context,
        clip_nonnegative,
        recursive_demand_features,
    )
    from ridge_model import prepare_feature_frame


SUPPORTED_HORIZONS = {1, 7, 30}
FORECAST_COLUMNS = [
    "item_id",
    "store_id",
    "forecast_origin",
    "target_date",
    "horizon_day",
    "forecast",
]


def save_forecaster(bundle: dict[str, Any], path: Path) -> Path:
    """Serialize the fitted pipeline together with its frozen forecast contract."""

    required = {"pipeline", "config", "model_family", "trained_through"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Forecaster bundle is missing: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_forecaster(path: Path | str) -> dict[str, Any]:
    """Load and minimally validate a serialized SmartStock forecaster bundle."""

    bundle = joblib.load(Path(path))
    required = {"pipeline", "config", "model_family", "trained_through"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("The model artifact is not a valid SmartStock forecaster bundle.")
    return bundle


def _resolve_bundle(forecaster: dict[str, Any] | Path | str) -> dict[str, Any]:
    return load_forecaster(forecaster) if isinstance(forecaster, (str, Path)) else forecaster


def forecast_demand(
    forecaster: dict[str, Any] | Path | str,
    feature_data: pd.DataFrame,
    forecast_origin: str | pd.Timestamp,
    horizon_days: int = 30,
    *,
    return_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Forecast 1, 7, or 30 daily steps using only origin-known demand history.

    Future rows supply calendar, event, SNAP, identity, and product-age context.
    Their sales and precomputed demand-history columns are never selected. Each
    official prediction is clipped at zero, appended to synthetic history, and
    used to recompute the next day's lags and rolling statistics.
    """

    if horizon_days not in SUPPORTED_HORIZONS:
        raise ValueError(f"horizon_days must be one of {sorted(SUPPORTED_HORIZONS)}")
    bundle = _resolve_bundle(forecaster)
    pipeline = bundle["pipeline"]
    config = bundle["config"]
    origin = pd.Timestamp(forecast_origin)
    start = origin + pd.DateOffset(days=1)
    end = origin + pd.DateOffset(days=horizon_days)
    expected_series = int(config.get("expected_series", 300))

    context = build_restricted_future_context(feature_data, start, end, config)
    counts = context.groupby("date", observed=True).size()
    if (
        len(counts) != horizon_days
        or counts.nunique() != 1
        or int(counts.iloc[0]) != expected_series
    ):
        raise ValueError(
            f"Every target day must contain exactly {expected_series} item-store contexts."
        )
    histories, _ = _build_histories(feature_data, origin)
    if len(histories) != expected_series:
        raise ValueError(f"Expected {expected_series} origin histories, found {len(histories)}.")

    categorical = list(config["categorical_features"])
    numeric = list(config["numeric_features"])
    outputs: list[pd.DataFrame] = []
    previous_forecast: dict[tuple[str, str], float] = {}
    first_forecast: dict[tuple[str, str], float] = {}
    lag1_check = True
    lag7_check = True
    rolling_check = True

    for horizon_day, target_date in enumerate(pd.date_range(start, end), start=1):
        day_context = context[context["date"].eq(target_date)].copy()
        keys: list[tuple[str, str]] = []
        feature_rows: list[dict[str, float]] = []
        for row in day_context.itertuples(index=False):
            key = (str(row.store_id), str(row.item_id))
            keys.append(key)
            feature_rows.append(recursive_demand_features(histories[key]))
        demand_frame = pd.DataFrame(feature_rows, index=day_context.index)
        model_frame = pd.concat([day_context, demand_frame], axis=1)

        if horizon_day == 2:
            lag1_check &= all(
                np.isclose(model_frame.iloc[index]["sales_lag_1"], previous_forecast[key])
                for index, key in enumerate(keys)
            )
            rolling_check &= all(
                np.isclose(
                    model_frame.iloc[index]["sales_roll_mean_7"],
                    np.asarray(histories[key][-7:], dtype="float64").mean(),
                )
                for index, key in enumerate(keys)
            )
        if horizon_day == 8:
            lag7_check &= all(
                np.isclose(model_frame.iloc[index]["sales_lag_7"], first_forecast[key])
                for index, key in enumerate(keys)
            )

        model_features = prepare_feature_frame(model_frame, categorical, numeric)
        raw = np.asarray(pipeline.predict(model_features), dtype="float64")
        official = clip_nonnegative(raw)
        day_output = day_context[["item_id", "store_id", "date"]].rename(
            columns={"date": "target_date"}
        )
        day_output["forecast_origin"] = origin
        day_output["horizon_day"] = horizon_day
        day_output["raw_forecast"] = raw
        day_output["forecast"] = official
        for index, key in enumerate(keys):
            value = float(official[index])
            histories[key].append(value)
            previous_forecast[key] = value
            if horizon_day == 1:
                first_forecast[key] = value
        outputs.append(day_output)

    predictions = pd.concat(outputs, ignore_index=True)
    predictions = predictions[
        [*FORECAST_COLUMNS[:-1], "raw_forecast", FORECAST_COLUMNS[-1]]
    ].sort_values(["target_date", "store_id", "item_id"], kind="stable").reset_index(drop=True)
    audit = {
        "forecast_origin": origin.date().isoformat(),
        "forecast_start": start.date().isoformat(),
        "forecast_end": end.date().isoformat(),
        "horizon_days": horizon_days,
        "series": expected_series,
        "prediction_rows": len(predictions),
        "future_sales_columns_selected": False,
        "future_precomputed_demand_features_selected": False,
        "recursive_lag_1_uses_prediction": bool(lag1_check),
        "recursive_lag_7_uses_prediction": bool(lag7_check),
        "recursive_rolling_uses_synthetic_history": bool(rolling_check),
        "official_forecasts_nonnegative": bool(predictions["forecast"].ge(0).all()),
        "forecasts_rounded": False,
    }
    if not all(
        audit[name]
        for name in (
            "recursive_lag_1_uses_prediction",
            "recursive_lag_7_uses_prediction",
            "recursive_rolling_uses_synthetic_history",
            "official_forecasts_nonnegative",
        )
    ):
        raise RuntimeError("Production recursive forecast audit failed.")
    return (predictions, audit) if return_audit else predictions
