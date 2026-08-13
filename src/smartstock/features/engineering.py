"""Core leakage-safe transformations for the SmartStock Stage 6 dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from smartstock.features.feature_definitions import REQUIRED_HISTORY_FEATURES
except ModuleNotFoundError:  # Supports direct execution of the Stage 6 CLI.
    from feature_definitions import REQUIRED_HISTORY_FEATURES


SERIES_KEYS = ["store_id", "item_id"]
EXPECTED_STORES = ("CA_1", "TX_2", "WI_3")
SNAP_COLUMNS = {"CA_1": "snap_CA", "TX_2": "snap_TX", "WI_3": "snap_WI"}


def load_v1_dataset(path: Path) -> pd.DataFrame:
    """Load the frozen V1 CSV with compact, calculation-safe dtypes."""

    categorical = [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "d",
        "weekday",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "demand_band",
    ]
    # Pandas 3 may infer different category scalar dtypes in separate CSV parser
    # chunks. Read stable strings first, then compact them after concatenation.
    dtypes: dict[str, str] = {column: "string" for column in categorical}
    dtypes.update(
        {
            "sales": "uint16",
            "wm_yr_wk": "int32",
            "wday": "int8",
            "month": "int8",
            "year": "int16",
            "snap_CA": "int8",
            "snap_TX": "int8",
            "snap_WI": "int8",
            "sell_price": "float32",
        }
    )
    data = pd.read_csv(path, dtype=dtypes, parse_dates=["date"])
    for column in categorical:
        data[column] = data[column].astype("category")
    return data


def validate_input(data: pd.DataFrame, manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate the frozen input before deriving any features."""

    required = {
        "item_id", "dept_id", "cat_id", "store_id", "state_id", "d", "sales", "date",
        "wm_yr_wk", "weekday", "wday", "month", "year", "event_name_1", "event_type_1",
        "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI", "sell_price", "demand_band",
    }
    missing = sorted(required.difference(data.columns))
    manifest_items = {item["item_id"] for item in manifest["items"]}
    pair_sizes = data.groupby(SERIES_KEYS, observed=True).size()
    checks = {
        "required_columns_present": not missing,
        "expected_rows": len(data) == 582_300,
        "expected_columns": len(data.columns) == 23,
        "expected_items": data["item_id"].nunique() == 100,
        "expected_stores": set(data["store_id"].astype("string").unique()) == set(EXPECTED_STORES),
        "expected_pairs": len(pair_sizes) == 300,
        "complete_daily_history": pair_sizes.min() == 1_941 and pair_sizes.max() == 1_941,
        "expected_date_range": data["date"].min() == pd.Timestamp("2011-01-29") and data["date"].max() == pd.Timestamp("2016-05-22"),
        "foods_only": set(data["cat_id"].astype("string").unique()) == {"FOODS"},
        "manifest_items_match": set(data["item_id"].astype("string").unique()) == manifest_items,
        "no_negative_sales": bool(data["sales"].ge(0).all()),
        "unique_daily_key": not data.duplicated(["date", *SERIES_KEYS]).any(),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Frozen V1 input validation failed: {', '.join(failed)}; missing columns: {missing}")
    return {
        "rows": len(data),
        "columns": len(data.columns),
        "items": int(data["item_id"].nunique()),
        "stores": int(data["store_id"].nunique()),
        "item_store_pairs": int(len(pair_sizes)),
        "date_min": data["date"].min().date().isoformat(),
        "date_max": data["date"].max().date().isoformat(),
        "memory_mib": round(data.memory_usage(index=True, deep=True).sum() / 2**20, 2),
        "checks": checks,
    }


def filter_active_period(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Exclude pre-launch targets using first known price with a positive-sale fallback."""

    price_dates = data["date"].where(data["sell_price"].notna())
    first_price = price_dates.groupby([data[key] for key in SERIES_KEYS], observed=True).transform("min")
    positive_dates = data["date"].where(data["sales"].gt(0))
    first_positive = positive_dates.groupby([data[key] for key in SERIES_KEYS], observed=True).transform("min")
    availability = first_price.fillna(first_positive)
    if availability.isna().any():
        raise ValueError("At least one item-store pair has neither a known price nor a positive sale.")

    is_active = data["date"].ge(availability)
    prelaunch = data.loc[~is_active]
    active = data.loc[is_active].copy()
    active["availability_date"] = availability.loc[is_active].to_numpy()
    active["is_active"] = True
    active["product_age_days"] = (active["date"] - active["availability_date"]).dt.days.astype("int16")
    active["product_age_weeks"] = (active["product_age_days"] // 7).astype("int16")

    source_by_pair = pd.DataFrame(
        {
            "store_id": data["store_id"],
            "item_id": data["item_id"],
            "first_price": first_price,
            "first_positive": first_positive,
        }
    ).drop_duplicates(SERIES_KEYS)
    fallback_pairs = int(source_by_pair["first_price"].isna().sum())
    summary = {
        "availability_definition": "first known selling-price date per item-store; first positive sale only as fallback",
        "active_rows": len(active),
        "prelaunch_rows_excluded": len(prelaunch),
        "prelaunch_pct": 100 * len(prelaunch) / len(data),
        "active_zero_rows_retained": int(active["sales"].eq(0).sum()),
        "active_zero_pct": 100 * active["sales"].eq(0).mean(),
        "prelaunch_positive_sales_rows": int(prelaunch["sales"].gt(0).sum()),
        "prelaunch_nonmissing_price_rows": int(prelaunch["sell_price"].notna().sum()),
        "active_missing_price_rows": int(active["sell_price"].isna().sum()),
        "price_proxy_pairs": int(len(source_by_pair) - fallback_pairs),
        "positive_sale_fallback_pairs": fallback_pairs,
        "availability_date_min": active["availability_date"].min().date().isoformat(),
        "availability_date_max": active["availability_date"].max().date().isoformat(),
        "product_age_days_max": int(active["product_age_days"].max()),
    }
    if summary["prelaunch_positive_sales_rows"] or summary["prelaunch_nonmissing_price_rows"]:
        raise ValueError("Availability-boundary integrity checks failed.")
    return active, summary


def add_calendar_features(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, bool]]:
    """Add same-day calendar features that are known in advance."""

    result = data.copy()
    parsed_day = result["date"].dt.dayofweek
    checks = {
        "source_weekday_matches_parsed_date": bool(result["weekday"].astype("string").eq(result["date"].dt.day_name()).all()),
        "source_month_matches_parsed_date": bool(result["month"].eq(result["date"].dt.month).all()),
        "source_year_matches_parsed_date": bool(result["year"].eq(result["date"].dt.year).all()),
    }
    if not all(checks.values()):
        raise ValueError(f"Calendar source fields conflict with parsed dates: {checks}")
    result["day_of_week"] = parsed_day.astype("int8")
    result["day_of_month"] = result["date"].dt.day.astype("int8")
    result["quarter"] = result["date"].dt.quarter.astype("int8")
    result["week_of_year"] = result["date"].dt.isocalendar().week.astype("int8")
    result["is_weekend"] = result["day_of_week"].ge(5)
    result["day_of_week_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7).astype("float32")
    result["day_of_week_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7).astype("float32")
    result["month_sin"] = np.sin(2 * np.pi * (result["month"] - 1) / 12).astype("float32")
    result["month_cos"] = np.cos(2 * np.pi * (result["month"] - 1) / 12).astype("float32")
    result["is_event"] = result[["event_name_1", "event_name_2"]].notna().any(axis=1)
    return result, checks


def store_relevant_snap(data: pd.DataFrame) -> pd.Series:
    """Map each store to only its corresponding state SNAP flag."""

    values = np.select(
        [data["store_id"].astype("string").eq(store_id) for store_id in EXPECTED_STORES],
        [data[SNAP_COLUMNS[store_id]] for store_id in EXPECTED_STORES],
        default=-1,
    )
    result = pd.Series(values, index=data.index, dtype="int8", name="snap_active")
    if result.eq(-1).any():
        raise ValueError("SNAP mapping encountered an unexpected store_id.")
    return result


def add_demand_history_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add grouped past-demand features without inspecting the target day's sales."""

    result = data.sort_values([*SERIES_KEYS, "date"], kind="stable").copy()
    grouped_sales = result.groupby(SERIES_KEYS, observed=True, sort=False)["sales"]
    for lag in (1, 7, 14, 28):
        result[f"sales_lag_{lag}"] = grouped_sales.shift(lag).astype("float32")

    shifted = grouped_sales.shift(1).astype("float32")
    shifted_group = shifted.groupby([result[key] for key in SERIES_KEYS], observed=True, sort=False)
    for window in (7, 14, 28):
        values = shifted_group.transform(lambda series: series.rolling(window, min_periods=window).mean())
        result[f"sales_roll_mean_{window}"] = values.astype("float32")
    for window in (7, 28):
        values = shifted_group.transform(lambda series: series.rolling(window, min_periods=window).std())
        result[f"sales_roll_std_{window}"] = values.astype("float32")

    historical_zero = shifted.eq(0).where(shifted.notna()).astype("float32")
    zero_group = historical_zero.groupby([result[key] for key in SERIES_KEYS], observed=True, sort=False)
    for window in (7, 28):
        values = zero_group.transform(lambda series: series.rolling(window, min_periods=window).mean())
        result[f"zero_rate_{window}"] = values.astype("float32")

    positive_date = result["date"].where(result["sales"].gt(0))
    prior_candidate = positive_date.groupby([result[key] for key in SERIES_KEYS], observed=True, sort=False).shift(1)
    prior_positive = prior_candidate.groupby([result[key] for key in SERIES_KEYS], observed=True, sort=False).ffill()
    result["days_since_last_positive_sale"] = (result["date"] - prior_positive).dt.days.astype("float32")
    return result


def add_weekly_price_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Add completed-prior-week price features and return extreme-change evidence."""

    weekly = (
        data.groupby([*SERIES_KEYS, "wm_yr_wk"], observed=True, sort=False)
        .agg(
            week_start=("date", "min"),
            weekly_price=("sell_price", "first"),
            distinct_daily_prices=("sell_price", "nunique"),
        )
        .reset_index()
        .sort_values([*SERIES_KEYS, "week_start"], kind="stable")
    )
    if weekly["distinct_daily_prices"].gt(1).any():
        raise ValueError("A single item-store-week contains multiple selling prices.")
    price_group = weekly.groupby(SERIES_KEYS, observed=True, sort=False)["weekly_price"]
    weekly["previous_week_price"] = price_group.shift(1)
    price_two_weeks_ago = price_group.shift(2)
    weekly["price_change_previous_week"] = weekly["previous_week_price"] - price_two_weeks_ago
    weekly["price_pct_change_previous_week"] = 100 * (weekly["previous_week_price"] / price_two_weeks_ago - 1)
    four_week_median = price_group.transform(lambda series: series.shift(1).rolling(4, min_periods=4).median())
    weekly["price_vs_4week_median_lagged"] = weekly["previous_week_price"] / four_week_median - 1

    weekly["current_change_absolute"] = weekly["weekly_price"] - weekly["previous_week_price"]
    weekly["current_change_pct"] = 100 * (weekly["weekly_price"] / weekly["previous_week_price"] - 1)
    changed = weekly.loc[weekly["current_change_absolute"].notna() & weekly["current_change_absolute"].ne(0)].copy()
    changed["absolute_pct_change"] = changed["current_change_pct"].abs()
    extremes = changed.nlargest(10, "absolute_pct_change")[
        ["store_id", "item_id", "wm_yr_wk", "week_start", "previous_week_price", "weekly_price", "current_change_absolute", "current_change_pct", "absolute_pct_change"]
    ]

    merge_columns = [*SERIES_KEYS, "wm_yr_wk", *[name for name in REQUIRED_HISTORY_FEATURES if name.startswith("price_") or name == "previous_week_price"]]
    before = len(data)
    result = data.merge(weekly[merge_columns], on=[*SERIES_KEYS, "wm_yr_wk"], how="left", validate="many_to_one")
    if len(result) != before:
        raise ValueError("Weekly price-feature join multiplied rows.")
    result = result.rename(columns={"sell_price": "known_future_sell_price"})
    for column in ["known_future_sell_price", "previous_week_price", "price_change_previous_week", "price_pct_change_previous_week", "price_vs_4week_median_lagged"]:
        result[column] = result[column].astype("float32")

    largest = extremes.iloc[0] if not extremes.empty else None
    review = {
        "weekly_records": len(weekly),
        "price_change_events": len(changed),
        "maximum_absolute_pct_change": float(changed["absolute_pct_change"].max()),
        "largest_change_previous_price": float(largest["previous_week_price"]) if largest is not None else None,
        "largest_change_new_price": float(largest["weekly_price"]) if largest is not None else None,
        "largest_change_low_denominator": bool(largest is not None and largest["previous_week_price"] < 0.5),
        "interpretation": "The largest percentage is denominator-sensitive. Prices are retained without clipping; any later transform or cap must be selected within training/validation data only.",
    }
    return result, extremes, review


def finalize_feature_dataset(data: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the explicit readiness rule, safe dtypes, column order, and final checks."""

    result = data.copy()
    result["is_feature_ready"] = result[REQUIRED_HISTORY_FEATURES].notna().all(axis=1)
    result["sales"] = result["sales"].astype("uint16")
    for column in ("item_id", "dept_id", "cat_id", "store_id", "state_id", "d", "event_name_1", "event_type_1", "event_name_2", "event_type_2", "demand_band"):
        result[column] = result[column].astype("category")
    result = result[feature_names].sort_values(["date", "store_id", "item_id"], kind="stable").reset_index(drop=True)

    pair_sizes = result.groupby(SERIES_KEYS, observed=True).size()
    ready = result[result["is_feature_ready"]]
    first_ready_age = result.loc[result["is_feature_ready"]].groupby(SERIES_KEYS, observed=True)["product_age_days"].min()
    missingness = []
    for name in REQUIRED_HISTORY_FEATURES:
        missing = int(result[name].isna().sum())
        missingness.append({"feature": name, "missing_rows": missing, "missing_pct": 100 * missing / len(result)})

    expected_lag1 = result.sort_values([*SERIES_KEYS, "date"], kind="stable").groupby(SERIES_KEYS, observed=True)["sales"].shift(1)
    sorted_result = result.sort_values([*SERIES_KEYS, "date"], kind="stable")
    lag1_safe = bool(np.allclose(sorted_result["sales_lag_1"], expected_lag1, equal_nan=True))
    first_rows = sorted_result.groupby(SERIES_KEYS, observed=True, sort=False).head(1)
    checks = {
        "active_rows_only": bool(result["is_active"].all()),
        "active_price_present": not result["known_future_sell_price"].isna().any(),
        "active_zeros_retained": bool(result["sales"].eq(0).any()),
        "expected_items": result["item_id"].nunique() == 100,
        "expected_stores": set(result["store_id"].astype("string").unique()) == set(EXPECTED_STORES),
        "expected_pairs": len(pair_sizes) == 300,
        "unique_daily_key": not result.duplicated(["date", *SERIES_KEYS]).any(),
        "nonnegative_product_age": bool(result["product_age_days"].ge(0).all()),
        "lag_1_matches_prior_series_value": lag1_safe,
        "first_series_rows_have_no_lag": bool(first_rows["sales_lag_1"].isna().all()),
        "ready_rows_have_complete_required_history": not ready[REQUIRED_HISTORY_FEATURES].isna().any().any(),
        "warmup_rows_persisted": bool((~result["is_feature_ready"]).any()),
        "manifest_columns_match_dataset": list(result.columns) == feature_names,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Final feature validation failed: {', '.join(failed)}")
    summary = {
        "rows": len(result),
        "columns": len(result.columns),
        "feature_ready_rows": len(ready),
        "warmup_rows": int((~result["is_feature_ready"]).sum()),
        "feature_ready_pct": 100 * result["is_feature_ready"].mean(),
        "items": int(result["item_id"].nunique()),
        "stores": int(result["store_id"].nunique()),
        "item_store_pairs": int(len(pair_sizes)),
        "date_min": result["date"].min().date().isoformat(),
        "date_max": result["date"].max().date().isoformat(),
        "feature_ready_date_min": ready["date"].min().date().isoformat(),
        "feature_ready_date_max": ready["date"].max().date().isoformat(),
        "first_ready_product_age_days_min": int(first_ready_age.min()),
        "first_ready_product_age_days_max": int(first_ready_age.max()),
        "memory_mib": round(result.memory_usage(index=True, deep=True).sum() / 2**20, 2),
        "missingness": missingness,
        "checks": checks,
    }
    return result, summary
