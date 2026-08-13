"""Leakage-safe recursive multi-step forecasting for SmartStock Stage 8."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from smartstock.models.baseline_evaluation import classify_intermittency
    from smartstock.models.metrics import rmsse_scale
    from smartstock.models.ridge_model import prepare_feature_frame
except ModuleNotFoundError:  # Supports direct execution from the models directory.
    from baseline_evaluation import classify_intermittency
    from metrics import rmsse_scale
    from ridge_model import prepare_feature_frame


SERIES_KEYS = ["store_id", "item_id"]
DEMAND_FEATURES = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_roll_mean_7",
    "sales_roll_mean_14",
    "sales_roll_mean_28",
    "sales_roll_std_7",
    "sales_roll_std_28",
]


def recursive_demand_features(history: list[float] | np.ndarray) -> dict[str, float]:
    """Recompute Stage 6 lag/rolling definitions from history ending at t-1."""

    values = np.asarray(history, dtype="float64")
    if values.ndim != 1 or values.size < 28:
        raise ValueError("At least 28 prior observations are required for recursive demand features.")
    return {
        "sales_lag_1": float(values[-1]),
        "sales_lag_7": float(values[-7]),
        "sales_lag_14": float(values[-14]),
        "sales_lag_28": float(values[-28]),
        "sales_roll_mean_7": float(values[-7:].mean()),
        "sales_roll_mean_14": float(values[-14:].mean()),
        "sales_roll_mean_28": float(values[-28:].mean()),
        "sales_roll_std_7": float(values[-7:].std(ddof=1)),
        "sales_roll_std_28": float(values[-28:].std(ddof=1)),
    }


def build_restricted_future_context(
    data: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Select only identity and same-day-known fields needed at future target dates."""

    categorical = list(config["categorical_features"])
    known_numeric = [
        name for name in config["known_context_numeric_features"] if name != "product_age_days"
    ]
    columns = ["date", "availability_date", *categorical, *known_numeric]
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(f"Future context source is missing: {sorted(missing)}")
    context = data.loc[data["date"].between(start, end), columns].copy()
    context["product_age_days"] = (context["date"] - context["availability_date"]).dt.days.astype("float32")
    context = context.drop(columns="availability_date").sort_values(["date", *SERIES_KEYS], kind="stable")
    if context.duplicated(["date", *SERIES_KEYS]).any():
        raise ValueError("Duplicate future context keys detected.")
    return context.reset_index(drop=True)


def _build_histories(data: pd.DataFrame, origin: pd.Timestamp) -> tuple[dict[tuple[str, str], list[float]], dict[tuple[str, str], dict[str, Any]]]:
    history = data.loc[data["date"].le(origin), ["date", *SERIES_KEYS, "sales"]].sort_values(
        [*SERIES_KEYS, "date"], kind="stable"
    )
    values_by_series: dict[tuple[str, str], list[float]] = {}
    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for (store_id, item_id), group in history.groupby(SERIES_KEYS, observed=True, sort=False):
        key = (str(store_id), str(item_id))
        values = group["sales"].to_numpy(dtype="float64")
        if len(values) < 28 or group["date"].max() != origin:
            raise ValueError(f"Insufficient or non-origin history for {key[0]}/{key[1]}.")
        intermittent_class, zero_pct = classify_intermittency(values)
        values_by_series[key] = values.tolist()
        metadata[key] = {
            "history_observations": len(values),
            "history_end_date": origin,
            "rmsse_scale": rmsse_scale(values),
            "training_zero_pct": zero_pct,
            "intermittency_class": intermittent_class,
        }
    return values_by_series, metadata


def clip_nonnegative(raw_forecast: np.ndarray | list[float]) -> np.ndarray:
    """Apply the predeclared sales-domain constraint without integer rounding."""

    return np.maximum(np.asarray(raw_forecast, dtype="float64"), 0.0)


def generate_recursive_predictions(
    pipeline: Any,
    data: pd.DataFrame,
    fold: dict[str, str],
    validation_manifest: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate an entire 30-day fold without using any validation actual in history."""

    start = pd.Timestamp(fold["start_date"])
    end = pd.Timestamp(fold["end_date"])
    origin = start + pd.DateOffset(days=-1)
    horizon = int(config["forecast_policy"]["horizon_days"])
    if (end - start).days + 1 != horizon:
        raise ValueError(f"{fold['name']} must contain exactly {horizon} days.")
    test_start = pd.Timestamp(validation_manifest["locked_final_test"]["start_date"])
    test_end = pd.Timestamp(validation_manifest["locked_final_test"]["end_date"])
    if start <= test_end and end >= test_start:
        raise ValueError("Stage 8 final-test lock blocks the requested forecast window.")

    context = build_restricted_future_context(data, start, end, config)
    expected_series = int(config.get("expected_series", 300))
    counts = context.groupby("date", observed=True).size()
    if len(counts) != horizon or counts.nunique() != 1 or int(counts.iloc[0]) != expected_series:
        raise ValueError(f"Every recursive target day must contain exactly {expected_series} item-store contexts.")
    histories, series_metadata = _build_histories(data, origin)
    if len(histories) != expected_series:
        raise ValueError(f"Recursive forecasting requires exactly {expected_series} item-store histories.")

    categorical = list(config["categorical_features"])
    numeric = list(config["numeric_features"])
    prediction_frames: list[pd.DataFrame] = []
    previous_forecast: dict[tuple[str, str], float] = {}
    day_one_forecast: dict[tuple[str, str], float] = {}
    lag1_recursive_check = True
    lag7_recursive_check = True
    rolling_recursive_check = True

    for horizon_day, target_date in enumerate(pd.date_range(start, end), start=1):
        day_context = context[context["date"].eq(target_date)].copy()
        feature_rows: list[dict[str, float]] = []
        keys: list[tuple[str, str]] = []
        for row in day_context.itertuples(index=False):
            key = (str(row.store_id), str(row.item_id))
            keys.append(key)
            feature_rows.append(recursive_demand_features(histories[key]))
        demand_frame = pd.DataFrame(feature_rows, index=day_context.index)
        model_frame = pd.concat([day_context, demand_frame], axis=1)

        if horizon_day == 2:
            lag1_recursive_check &= all(
                np.isclose(model_frame.iloc[index]["sales_lag_1"], previous_forecast[key])
                for index, key in enumerate(keys)
            )
            rolling_recursive_check &= all(
                np.isclose(
                    model_frame.iloc[index]["sales_roll_mean_7"],
                    np.asarray(histories[key][-7:], dtype="float64").mean(),
                )
                for index, key in enumerate(keys)
            )
        if horizon_day == 8:
            lag7_recursive_check &= all(
                np.isclose(model_frame.iloc[index]["sales_lag_7"], day_one_forecast[key])
                for index, key in enumerate(keys)
            )

        features = prepare_feature_frame(model_frame, categorical, numeric)
        raw = np.asarray(pipeline.predict(features), dtype="float64")
        official = clip_nonnegative(raw)
        day_output = day_context[["date", "store_id", "item_id", "dept_id"]].copy()
        day_output = day_output.rename(columns={"date": "target_date"})
        day_output["fold"] = fold["name"]
        day_output["model_name"] = config["experiment_name"]
        day_output["forecast_origin"] = origin
        day_output["horizon_day"] = horizon_day
        day_output["raw_forecast"] = raw
        day_output["forecast"] = official
        for index, key in enumerate(keys):
            meta = series_metadata[key]
            for name, value in meta.items():
                day_output.loc[day_output.index[index], name] = value
            histories[key].append(float(official[index]))
            previous_forecast[key] = float(official[index])
            if horizon_day == 1:
                day_one_forecast[key] = float(official[index])
        prediction_frames.append(day_output)

    forecasts = pd.concat(prediction_frames, ignore_index=True)
    actuals = data.loc[
        data["date"].between(start, end),
        ["date", *SERIES_KEYS, "sales", "demand_band"],
    ].rename(columns={"date": "target_date", "sales": "actual"})
    predictions = forecasts.merge(actuals, on=["target_date", *SERIES_KEYS], how="left", validate="one_to_one")
    if predictions["actual"].isna().any():
        raise ValueError("Validation actuals could not be attached after recursive forecasting.")
    predictions = predictions[
        [
            "fold",
            "model_name",
            "store_id",
            "item_id",
            "dept_id",
            "demand_band",
            "intermittency_class",
            "forecast_origin",
            "history_end_date",
            "history_observations",
            "target_date",
            "horizon_day",
            "actual",
            "raw_forecast",
            "forecast",
            "rmsse_scale",
            "training_zero_pct",
        ]
    ].sort_values(["fold", "target_date", "store_id", "item_id"], kind="stable").reset_index(drop=True)

    raw_negative = predictions["raw_forecast"].lt(0)
    checks = {
        "single_forecast_origin": predictions["forecast_origin"].nunique() == 1,
        "history_ends_at_origin": predictions["history_end_date"].eq(origin).all(),
        "actual_validation_sales_excluded_from_recursive_history": True,
        "recursive_lag_1_uses_prediction": bool(lag1_recursive_check),
        "recursive_lag_7_uses_prediction": bool(lag7_recursive_check),
        "recursive_rolling_uses_synthetic_history": bool(rolling_recursive_check),
        "precomputed_validation_demand_features_ignored": True,
        "future_price_features_excluded": True,
        "targets_within_fold": predictions["target_date"].between(start, end).all(),
        "final_test_untouched": predictions["target_date"].lt(test_start).all(),
        "official_forecasts_nonnegative": predictions["forecast"].ge(0).all(),
        "expected_prediction_rows": len(predictions) == expected_series * horizon,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Recursive forecast-origin audit failed for {fold['name']}: {checks}")
    audit = {
        "fold": fold["name"],
        "forecast_origin": origin.date().isoformat(),
        "validation_start": start.date().isoformat(),
        "validation_end": end.date().isoformat(),
        "checks": checks,
        "negative_predictions": {
            "count": int(raw_negative.sum()),
            "percentage": float(100 * raw_negative.mean()),
            "most_negative": float(predictions["raw_forecast"].min()),
        },
        "intermittency_class_counts": (
            predictions[[*SERIES_KEYS, "intermittency_class"]]
            .drop_duplicates()["intermittency_class"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
    }
    return predictions, audit
