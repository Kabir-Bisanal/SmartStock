"""Run Stage 5 exploratory data analysis on the frozen Version 1 dataset.

All derived columns and aggregates in this module are analytical only. The
Version 1 CSV and subset manifest are read but never modified or overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    from smartstock.analysis.stage5_figures import create_figures
    from smartstock.analysis.stage5_report import render_stage5_report
except ModuleNotFoundError:  # Supports direct execution from the repository root.
    from stage5_figures import create_figures
    from stage5_report import render_stage5_report


DEFAULT_DATA_PATH = Path("data") / "interim" / "smartstock_v1_long.csv"
DEFAULT_MANIFEST_PATH = Path("config") / "v1_subset.json"
DEFAULT_REPORT_DIR = Path("reports")
DEFAULT_FIGURE_DIR = DEFAULT_REPORT_DIR / "figures" / "stage5"

EXPECTED_ROWS = 582_300
EXPECTED_COLUMNS = 23
EXPECTED_ITEMS = 100
EXPECTED_STORES = ("CA_1", "TX_2", "WI_3")
EXPECTED_PAIRS = 300
EXPECTED_DAYS_PER_PAIR = 1_941

STORE_SNAP_COLUMNS = {"CA_1": "snap_CA", "TX_2": "snap_TX", "WI_3": "snap_WI"}
WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
MONTH_ORDER = list(range(1, 13))
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def file_sha256(path: Path) -> str:
    """Calculate a streaming SHA-256 hash for read-only integrity checks."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Recursively convert pandas/NumPy values into JSON-safe Python values."""

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return [json_safe(record) for record in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else round(float(value), 10)
    if pd.isna(value):
        return None
    return value


def load_v1_data(data_path: Path) -> pd.DataFrame:
    """Load the frozen Version 1 CSV once with compact analytical dtypes."""

    categorical_columns = [
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
    dtype_map: dict[str, Any] = {column: "category" for column in categorical_columns}
    dtype_map.update(
        {
            "sales": np.int16,
            "wm_yr_wk": np.int32,
            "wday": np.int8,
            "month": np.int8,
            "year": np.int16,
            "snap_CA": np.int8,
            "snap_TX": np.int8,
            "snap_WI": np.int8,
            "sell_price": np.float32,
        }
    )
    return pd.read_csv(data_path, parse_dates=["date"], dtype=dtype_map)


def validate_v1_data(data: pd.DataFrame, manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate Stage 4 dimensions, manifest alignment, and key cardinality."""

    pair_day_counts = data.groupby(["store_id", "item_id"], observed=True)["date"].nunique()
    manifest_items = {item["item_id"] for item in manifest["items"]}
    observed_items = {str(value) for value in data["item_id"].unique()}
    checks = {
        "expected_rows": len(data) == EXPECTED_ROWS,
        "expected_columns": data.shape[1] == EXPECTED_COLUMNS,
        "expected_items": data["item_id"].nunique() == EXPECTED_ITEMS,
        "expected_stores": sorted(str(value) for value in data["store_id"].unique()) == sorted(EXPECTED_STORES),
        "expected_item_store_pairs": pair_day_counts.size == EXPECTED_PAIRS,
        "complete_days_per_pair": bool(pair_day_counts.eq(EXPECTED_DAYS_PER_PAIR).all()),
        "unique_date_store_item_key": not data.duplicated(["date", "store_id", "item_id"]).any(),
        "foods_only": set(str(value) for value in data["cat_id"].unique()) == {"FOODS"},
        "manifest_items_match": observed_items == manifest_items,
        "no_negative_sales": not data["sales"].lt(0).any(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Version 1 validation failed: " + ", ".join(failed))

    return {
        "rows": int(len(data)),
        "columns": int(data.shape[1]),
        "memory_mib": round(data.memory_usage(deep=True).sum() / 1024**2, 2),
        "date_min": data["date"].min().date().isoformat(),
        "date_max": data["date"].max().date().isoformat(),
        "items": int(data["item_id"].nunique()),
        "stores": int(data["store_id"].nunique()),
        "item_store_pairs": int(pair_day_counts.size),
        "days_per_pair_min": int(pair_day_counts.min()),
        "days_per_pair_max": int(pair_day_counts.max()),
        "checks": checks,
    }


def identify_active_periods(data: pd.DataFrame) -> pd.DataFrame:
    """Add temporary launch, pre-launch, and active-period analytical columns.

    Availability begins at the first known selling price. If a series has no
    known price, the first positive sale is used as a fallback. This distinguishes
    unavailable pre-launch zeros from genuine active-period zero demand.
    """

    keys = ["store_id", "item_id"]
    first_known_price = data["date"].where(data["sell_price"].notna()).groupby(
        [data["store_id"], data["item_id"]], observed=True
    ).transform("min")
    first_positive_sale = data["date"].where(data["sales"].gt(0)).groupby(
        [data["store_id"], data["item_id"]], observed=True
    ).transform("min")
    launch_date = first_known_price.fillna(first_positive_sale)

    enriched = data.copy()
    enriched["first_known_price_date"] = first_known_price
    enriched["first_positive_sale_date"] = first_positive_sale
    enriched["launch_date"] = launch_date
    enriched["prelaunch_flag"] = enriched["date"].lt(enriched["launch_date"])
    enriched["active_flag"] = enriched["date"].ge(enriched["launch_date"])

    if enriched.groupby(keys, observed=True)["launch_date"].nunique().gt(1).any():
        raise ValueError("Launch dates are inconsistent within an item-store series.")
    return enriched


def store_specific_snap_flag(data: pd.DataFrame) -> pd.Series:
    """Return each row's state-appropriate SNAP flag based on its frozen store."""

    result = np.select(
        [data["store_id"].eq(store) for store in EXPECTED_STORES],
        [data[column].to_numpy() for store, column in STORE_SNAP_COLUMNS.items()],
        default=-1,
    )
    if np.any(result == -1):
        raise ValueError("Unknown store encountered while choosing the SNAP flag.")
    return pd.Series(result.astype(np.int8), index=data.index, name="snap_active")


def summarize_prelaunch(data: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Summarize availability boundaries and pre-launch durations."""

    keys = ["store_id", "item_id"]
    series_launches = data.groupby(keys, observed=True).agg(
        department=("dept_id", "first"),
        demand_band=("demand_band", "first"),
        history_start=("date", "min"),
        first_known_price_date=("first_known_price_date", "first"),
        first_positive_sale_date=("first_positive_sale_date", "first"),
        launch_date=("launch_date", "first"),
        prelaunch_rows=("prelaunch_flag", "sum"),
        total_rows=("date", "size"),
    ).reset_index()
    series_launches["prelaunch_days"] = (
        series_launches["launch_date"] - series_launches["history_start"]
    ).dt.days
    series_launches["price_to_first_sale_lag_days"] = (
        series_launches["first_positive_sale_date"]
        - series_launches["first_known_price_date"]
    ).dt.days

    active = data[data["active_flag"]]
    prelaunch_rows = int(data["prelaunch_flag"].sum())
    active_rows = int(data["active_flag"].sum())
    duration = series_launches["prelaunch_days"]
    lag = series_launches["price_to_first_sale_lag_days"]
    summary = {
        "boundary_definition": "first known price date; first positive sale is the fallback when no price exists",
        "prelaunch_rows": prelaunch_rows,
        "prelaunch_pct": 100 * prelaunch_rows / len(data),
        "active_rows": active_rows,
        "active_pct": 100 * active_rows / len(data),
        "prelaunch_zero_sales_rows": int(
            (data["prelaunch_flag"] & data["sales"].eq(0)).sum()
        ),
        "prelaunch_positive_sales_rows": int(
            (data["prelaunch_flag"] & data["sales"].gt(0)).sum()
        ),
        "prelaunch_missing_price_rows": int(
            (data["prelaunch_flag"] & data["sell_price"].isna()).sum()
        ),
        "active_missing_price_rows": int(
            (data["active_flag"] & data["sell_price"].isna()).sum()
        ),
        "active_zero_sales_rows": int(active["sales"].eq(0).sum()),
        "active_zero_sales_pct": 100 * active["sales"].eq(0).mean(),
        "prelaunch_days_distribution": {
            "minimum": int(duration.min()),
            "median": float(duration.median()),
            "mean": float(duration.mean()),
            "p75": float(duration.quantile(0.75)),
            "p90": float(duration.quantile(0.90)),
            "maximum": int(duration.max()),
        },
        "price_to_first_sale_lag_days": {
            "minimum": int(lag.min()),
            "median": float(lag.median()),
            "mean": float(lag.mean()),
            "p90": float(lag.quantile(0.90)),
            "maximum": int(lag.max()),
        },
        "latest_10_item_store_launches": series_launches.nlargest(10, "launch_date")[
            ["store_id", "item_id", "department", "launch_date", "first_positive_sale_date", "prelaunch_days"]
        ],
    }
    return json_safe(summary), series_launches


def describe_sales(series: pd.Series) -> dict[str, Any]:
    """Return a stable descriptive demand profile for a sales series."""

    percentiles = series.quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 0.9999])
    p999 = float(percentiles.loc[0.999])
    return json_safe(
        {
            "observations": len(series),
            "mean": series.mean(),
            "median": series.median(),
            "standard_deviation": series.std(),
            "minimum": series.min(),
            "maximum": series.max(),
            "skewness": series.skew(),
            "zero_rows": series.eq(0).sum(),
            "zero_pct": 100 * series.eq(0).mean(),
            "percentiles": {str(index): value for index, value in percentiles.items()},
            "rows_above_p99_9": series.gt(p999).sum(),
            "rows_above_p99_9_pct": 100 * series.gt(p999).mean(),
        }
    )


def compute_series_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate active-period intermittency and variability by item-store series."""

    active = data[data["active_flag"]]
    metrics = active.groupby(["store_id", "item_id"], observed=True).agg(
        department=("dept_id", "first"),
        demand_band=("demand_band", "first"),
        active_days=("sales", "size"),
        total_sales=("sales", "sum"),
        mean_sales=("sales", "mean"),
        median_sales=("sales", "median"),
        std_sales=("sales", "std"),
        zero_days=("sales", lambda values: int(values.eq(0).sum())),
        positive_days=("sales", lambda values: int(values.gt(0).sum())),
        first_active_date=("date", "min"),
        last_active_date=("date", "max"),
    ).reset_index()
    metrics["active_zero_pct"] = 100 * metrics["zero_days"] / metrics["active_days"]
    metrics["coefficient_of_variation"] = metrics["std_sales"] / metrics["mean_sales"].replace(0, np.nan)
    metrics["average_demand_interval"] = metrics["active_days"] / metrics["positive_days"].replace(0, np.nan)

    positive = active.loc[active["sales"].gt(0), ["store_id", "item_id", "date"]].sort_values(
        ["store_id", "item_id", "date"]
    )
    positive["gap_days"] = positive.groupby(["store_id", "item_id"], observed=True)["date"].diff().dt.days
    gap_mean = positive.groupby(["store_id", "item_id"], observed=True)["gap_days"].mean()
    metrics = metrics.join(gap_mean.rename("mean_positive_gap_days"), on=["store_id", "item_id"])
    metrics["intermittency_class"] = pd.cut(
        metrics["active_zero_pct"],
        bins=[-np.inf, 50, 80, np.inf],
        labels=["regular (<50% zero)", "intermittent (50-80% zero)", "highly intermittent (>80% zero)"],
        right=True,
    ).astype("string")
    return metrics


def summarize_group_demand(
    data: pd.DataFrame,
    group_column: str,
    series_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize all-history and active-period demand for a grouping dimension."""

    active = data[data["active_flag"]]
    overall = data.groupby(group_column, observed=True).agg(
        rows=("sales", "size"),
        total_sales=("sales", "sum"),
        mean_sales=("sales", "mean"),
        median_sales=("sales", "median"),
        standard_deviation=("sales", "std"),
        zero_rows=("sales", lambda values: int(values.eq(0).sum())),
    )
    active_summary = active.groupby(group_column, observed=True).agg(
        active_rows=("sales", "size"),
        active_total_sales=("sales", "sum"),
        active_mean_sales=("sales", "mean"),
        active_median_sales=("sales", "median"),
        active_standard_deviation=("sales", "std"),
        active_zero_rows=("sales", lambda values: int(values.eq(0).sum())),
        selected_products=("item_id", "nunique"),
    )
    summary = overall.join(active_summary)
    summary["zero_pct"] = 100 * summary["zero_rows"] / summary["rows"]
    summary["active_zero_pct"] = 100 * summary["active_zero_rows"] / summary["active_rows"]
    summary["sales_share_pct"] = 100 * summary["total_sales"] / summary["total_sales"].sum()

    series_group_column = {
        "demand_band": "demand_band",
        "dept_id": "department",
        "store_id": "store_id",
    }[group_column]
    series_summary = series_metrics.groupby(series_group_column, observed=True).agg(
        median_series_active_zero_pct=("active_zero_pct", "median"),
        median_series_adi=("average_demand_interval", "median"),
        median_series_cv=("coefficient_of_variation", "median"),
    )
    return summary.join(series_summary).reset_index()


def compute_time_tables(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build daily, weekly, monthly, yearly, weekday, and seasonal aggregate tables."""

    daily_store = data.groupby(["date", "store_id"], observed=True)["sales"].sum().reset_index()
    daily_total = daily_store.groupby("date", observed=True)["sales"].sum().reset_index()
    daily_total["year"] = daily_total["date"].dt.year
    daily_total["month"] = daily_total["date"].dt.month

    weekly_store = (
        daily_store.set_index("date")
        .groupby("store_id", observed=True)["sales"]
        .resample("W-SUN")
        .sum()
        .rename("weekly_sales")
        .reset_index()
    )
    store_weekly_median = weekly_store.groupby("store_id", observed=True)["weekly_sales"].transform("median")
    weekly_store["normalized_weekly_sales"] = weekly_store["weekly_sales"] / store_weekly_median
    weekly_store["normalized_13_week_mean"] = weekly_store.groupby("store_id", observed=True)[
        "normalized_weekly_sales"
    ].transform(lambda values: values.rolling(13, min_periods=1, center=True).mean())

    monthly_store = (
        daily_store.set_index("date")
        .groupby("store_id", observed=True)["sales"]
        .resample("MS")
        .sum()
        .rename("monthly_sales")
        .reset_index()
    )
    monthly_total = monthly_store.groupby("date", observed=True)["monthly_sales"].sum().reset_index()
    yearly = data.groupby("year", observed=True)["sales"].agg(total_sales="sum", mean_daily_observation="mean").reset_index()
    yearly_days = data[["year", "date"]].drop_duplicates().groupby("year", observed=True).size()
    yearly["observed_days"] = yearly["year"].map(yearly_days)
    yearly["average_daily_total"] = yearly["total_sales"] / yearly["observed_days"]

    active = data[data["active_flag"]]
    weekday = active.groupby("weekday", observed=True).agg(
        active_observations=("sales", "size"),
        total_sales=("sales", "sum"),
        mean_sales=("sales", "mean"),
        median_sales=("sales", "median"),
    ).reindex(WEEKDAY_ORDER).reset_index()
    weekday_store = active.groupby(["store_id", "weekday"], observed=True)["sales"].mean().reset_index()
    weekday_store["weekday"] = pd.Categorical(weekday_store["weekday"], WEEKDAY_ORDER, ordered=True)
    weekday_store = weekday_store.sort_values(["store_id", "weekday"])

    daily_month_store = daily_store.copy()
    daily_month_store["month"] = daily_month_store["date"].dt.month
    monthly_seasonality = daily_month_store.groupby(["store_id", "month"], observed=True)["sales"].mean().reset_index(name="average_daily_sales")
    monthly_overall = daily_total.groupby("month", observed=True)["sales"].mean().reindex(MONTH_ORDER).reset_index(name="average_daily_sales")

    monthly_trend = monthly_total.copy()
    sequence = np.arange(len(monthly_trend), dtype=float)
    slope, intercept = np.polyfit(sequence, monthly_trend["monthly_sales"], 1)
    monthly_trend["linear_trend"] = intercept + slope * sequence
    monthly_trend["12_month_mean"] = monthly_trend["monthly_sales"].rolling(12, min_periods=6, center=True).mean()

    return {
        "daily_store": daily_store,
        "daily_total": daily_total,
        "weekly_store": weekly_store,
        "monthly_store": monthly_store,
        "monthly_total": monthly_total,
        "monthly_trend": monthly_trend,
        "yearly": yearly,
        "weekday": weekday,
        "weekday_store": weekday_store,
        "monthly_seasonality": monthly_seasonality,
        "monthly_overall": monthly_overall,
        "monthly_trend_slope_units_per_month": pd.DataFrame({"slope": [slope]}),
    }


def analyze_events(data: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Compare observed active demand on event and non-event dates."""

    date_events = data[
        ["date", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    ].drop_duplicates("date")
    date_events["event_day"] = date_events[["event_name_1", "event_name_2"]].notna().any(axis=1)
    event_map = date_events.set_index("date")["event_day"]

    active = data[data["active_flag"]].copy()
    active["event_day"] = active["date"].map(event_map)
    comparison = active.groupby("event_day", observed=True).agg(
        observations=("sales", "size"),
        total_sales=("sales", "sum"),
        mean_sales=("sales", "mean"),
        median_sales=("sales", "median"),
        zero_pct=("sales", lambda values: 100 * values.eq(0).mean()),
    ).reset_index()

    daily_active = active.groupby("date", observed=True)["sales"].sum().rename("daily_sales").reset_index()
    daily_active = daily_active.merge(date_events, on="date", how="left", validate="one_to_one")
    event_type_frames: list[pd.DataFrame] = []
    for column in ("event_type_1", "event_type_2"):
        part = daily_active.loc[daily_active[column].notna(), ["date", "daily_sales", column]].rename(columns={column: "event_type"})
        event_type_frames.append(part)
    event_types = pd.concat(event_type_frames, ignore_index=True)
    event_type_summary = event_types.groupby("event_type", observed=True).agg(
        event_occurrences=("date", "nunique"),
        mean_daily_sales=("daily_sales", "mean"),
        median_daily_sales=("daily_sales", "median"),
    ).reset_index()

    event_mean = float(comparison.loc[comparison["event_day"], "mean_sales"].iloc[0])
    non_event_mean = float(comparison.loc[~comparison["event_day"], "mean_sales"].iloc[0])
    summary = {
        "distinct_calendar_days": int(len(date_events)),
        "event_days": int(date_events["event_day"].sum()),
        "non_event_days": int((~date_events["event_day"]).sum()),
        "active_observation_comparison": comparison,
        "event_vs_non_event_mean_difference_pct": 100 * (event_mean / non_event_mean - 1),
        "event_type_summary": event_type_summary,
        "interpretation_limit": "observational comparison only; event timing, seasonality, stores, products, and prices are confounded",
    }
    return json_safe(summary), comparison, event_type_summary


def analyze_snap(data: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compare active-period demand using each store's state-specific SNAP flag."""

    active = data[data["active_flag"]].copy()
    active["snap_active"] = store_specific_snap_flag(active)
    comparison = active.groupby(["store_id", "snap_active"], observed=True).agg(
        observations=("sales", "size"),
        total_sales=("sales", "sum"),
        mean_sales=("sales", "mean"),
        median_sales=("sales", "median"),
        zero_pct=("sales", lambda values: 100 * values.eq(0).mean()),
    ).reset_index()
    overall = active.groupby("snap_active", observed=True).agg(
        observations=("sales", "size"),
        mean_sales=("sales", "mean"),
        median_sales=("sales", "median"),
        total_sales=("sales", "sum"),
    ).reset_index()
    department = active.groupby(["dept_id", "snap_active"], observed=True)["sales"].mean().reset_index(name="mean_sales")

    pivot = comparison.pivot(index="store_id", columns="snap_active", values="mean_sales")
    differences = 100 * (pivot[1] / pivot[0] - 1)
    summary = {
        "overall": overall,
        "by_store": comparison,
        "by_department": department,
        "store_mean_difference_pct": differences,
        "interpretation_limit": "observational association; SNAP timing may overlap with payday, seasonality, and other factors",
    }
    return json_safe(summary), comparison


def detect_weekly_price_changes(data: pd.DataFrame) -> pd.DataFrame:
    """Create one weekly price row per item-store and detect sequential changes."""

    weekly_prices = data.loc[
        data["active_flag"] & data["sell_price"].notna(),
        ["store_id", "item_id", "dept_id", "wm_yr_wk", "sell_price"],
    ].drop_duplicates(["store_id", "item_id", "wm_yr_wk"])
    weekly_prices = weekly_prices.sort_values(["store_id", "item_id", "wm_yr_wk"]).reset_index(drop=True)
    grouped = weekly_prices.groupby(["store_id", "item_id"], observed=True)["sell_price"]
    weekly_prices["previous_price"] = grouped.shift(1)
    weekly_prices["price_change"] = weekly_prices["sell_price"] - weekly_prices["previous_price"]
    weekly_prices["price_change_pct"] = 100 * weekly_prices["price_change"] / weekly_prices["previous_price"]
    tolerance = 1e-7
    weekly_prices["change_direction"] = np.select(
        [
            weekly_prices["previous_price"].isna(),
            weekly_prices["price_change"].lt(-tolerance),
            weekly_prices["price_change"].gt(tolerance),
        ],
        ["first observation", "decrease", "increase"],
        default="unchanged",
    )
    return weekly_prices


def analyze_prices(
    data: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Analyze weekly prices, price changes, and price-demand associations."""

    weekly_prices = detect_weekly_price_changes(data)
    price_series = weekly_prices["sell_price"]
    percentiles = price_series.quantile([0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])

    series_price = weekly_prices.groupby(["store_id", "item_id"], observed=True).agg(
        department=("dept_id", "first"),
        priced_weeks=("wm_yr_wk", "size"),
        distinct_prices=("sell_price", "nunique"),
        price_changes=("change_direction", lambda values: int(values.isin(["increase", "decrease"]).sum())),
        mean_absolute_change_pct=("price_change_pct", lambda values: values.abs().mean()),
    ).reset_index()
    changed_series = series_price["price_changes"].gt(0)
    change_rows = weekly_prices[weekly_prices["change_direction"].isin(["increase", "decrease"])]

    weekly_sales = data.loc[data["active_flag"]].groupby(
        ["store_id", "item_id", "wm_yr_wk"], observed=True
    )["sales"].sum().rename("weekly_sales").reset_index()
    price_demand = weekly_prices.merge(
        weekly_sales,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="one_to_one",
    )
    direction_summary = price_demand[
        price_demand["change_direction"].ne("first observation")
    ].groupby("change_direction", observed=True).agg(
        weeks=("weekly_sales", "size"),
        mean_weekly_sales=("weekly_sales", "mean"),
        median_weekly_sales=("weekly_sales", "median"),
        mean_price_change_pct=("price_change_pct", "mean"),
    ).reset_index()

    correlation_records: list[dict[str, Any]] = []
    for (store_id, item_id), group in price_demand.groupby(["store_id", "item_id"], observed=True):
        if len(group) >= 20 and group["sell_price"].nunique() >= 2:
            correlation = group["sell_price"].corr(group["weekly_sales"])
            if pd.notna(correlation):
                correlation_records.append(
                    {"store_id": str(store_id), "item_id": str(item_id), "correlation": float(correlation)}
                )
    correlations = pd.DataFrame.from_records(correlation_records)

    price_by_department = weekly_prices.groupby("dept_id", observed=True)["sell_price"].agg(
        observations="size", mean="mean", median="median", minimum="min", maximum="max"
    ).reset_index()
    price_by_store = weekly_prices.groupby("store_id", observed=True)["sell_price"].agg(
        observations="size", mean="mean", median="median", minimum="min", maximum="max"
    ).reset_index()

    summary = {
        "weekly_price_records": int(len(weekly_prices)),
        "mean": price_series.mean(),
        "median": price_series.median(),
        "standard_deviation": price_series.std(),
        "minimum": price_series.min(),
        "maximum": price_series.max(),
        "percentiles": {str(index): value for index, value in percentiles.items()},
        "by_department": price_by_department,
        "by_store": price_by_store,
        "item_store_series": int(len(series_price)),
        "series_with_price_change": int(changed_series.sum()),
        "series_with_price_change_pct": 100 * changed_series.mean(),
        "distinct_prices_per_series": {
            "minimum": int(series_price["distinct_prices"].min()),
            "median": float(series_price["distinct_prices"].median()),
            "mean": float(series_price["distinct_prices"].mean()),
            "maximum": int(series_price["distinct_prices"].max()),
        },
        "total_price_change_events": int(len(change_rows)),
        "increase_events": int(change_rows["change_direction"].eq("increase").sum()),
        "decrease_events": int(change_rows["change_direction"].eq("decrease").sum()),
        "absolute_change_pct": {
            "median": change_rows["price_change_pct"].abs().median(),
            "mean": change_rows["price_change_pct"].abs().mean(),
            "p90": change_rows["price_change_pct"].abs().quantile(0.90),
            "maximum": change_rows["price_change_pct"].abs().max(),
        },
        "most_frequent_10_price_changes": series_price.nlargest(10, "price_changes")[
            ["store_id", "item_id", "department", "priced_weeks", "distinct_prices", "price_changes", "mean_absolute_change_pct"]
        ],
        "price_change_direction_demand": direction_summary,
        "within_series_price_demand_correlations": {
            "series_analyzed": int(len(correlations)),
            "median": correlations["correlation"].median(),
            "mean": correlations["correlation"].mean(),
            "p25": correlations["correlation"].quantile(0.25),
            "p75": correlations["correlation"].quantile(0.75),
            "negative_pct": 100 * correlations["correlation"].lt(0).mean(),
            "positive_pct": 100 * correlations["correlation"].gt(0).mean(),
        },
        "interpretation_limit": "price-demand associations are observational and confounded by product, promotion, seasonality, events, and life cycle",
    }
    return json_safe(summary), weekly_prices, direction_summary, correlations


def analyze_intermittency(series_metrics: pd.DataFrame) -> dict[str, Any]:
    """Summarize active-period zero demand and average demand intervals."""

    class_counts = series_metrics["intermittency_class"].value_counts().sort_index()
    zero_pct = series_metrics["active_zero_pct"]
    adi = series_metrics["average_demand_interval"]
    summary = {
        "series": int(len(series_metrics)),
        "active_zero_pct_distribution": {
            "minimum": zero_pct.min(),
            "median": zero_pct.median(),
            "mean": zero_pct.mean(),
            "p75": zero_pct.quantile(0.75),
            "p90": zero_pct.quantile(0.90),
            "maximum": zero_pct.max(),
        },
        "average_demand_interval_distribution": {
            "minimum": adi.min(),
            "median": adi.median(),
            "mean": adi.mean(),
            "p90": adi.quantile(0.90),
            "maximum": adi.max(),
        },
        "intermittency_class_counts": class_counts,
        "most_intermittent_10_series": series_metrics.nlargest(10, "active_zero_pct")[
            ["store_id", "item_id", "department", "demand_band", "active_days", "active_zero_pct", "average_demand_interval", "mean_positive_gap_days"]
        ],
        "croston_consideration": "specialized intermittent-demand baselines may be worth testing later for the most zero-heavy series; no such model was implemented",
    }
    return json_safe(summary)


def analyze_outliers(data: pd.DataFrame) -> dict[str, Any]:
    """Describe extreme daily observations without removing or recoding them."""

    active = data[data["active_flag"]]
    thresholds = active["sales"].quantile([0.99, 0.999, 0.9999])
    p999 = float(thresholds.loc[0.999])
    extremes = active[active["sales"].gt(p999)].copy()
    extremes["event_day"] = extremes[["event_name_1", "event_name_2"]].notna().any(axis=1)
    top = active.nlargest(15, "sales")[
        ["date", "store_id", "item_id", "dept_id", "sales", "sell_price", "event_name_1", "event_type_1", "demand_band"]
    ]
    return json_safe(
        {
            "population": "active-period rows only",
            "p99": thresholds.loc[0.99],
            "p99_9": thresholds.loc[0.999],
            "p99_99": thresholds.loc[0.9999],
            "maximum": active["sales"].max(),
            "rows_above_p99_9": len(extremes),
            "rows_above_p99_9_pct": 100 * len(extremes) / len(active),
            "extreme_rows_on_event_days": int(extremes["event_day"].sum()),
            "extreme_event_day_pct": 100 * extremes["event_day"].mean(),
            "top_15_observations": top,
            "classification": "requires investigation; values are retained and no evidence establishes a data error",
        }
    )


def analyze_product_launches(series_launches: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Summarize first known price and first sale timing at product level."""

    product_launches = series_launches.groupby("item_id", observed=True).agg(
        department=("department", "first"),
        demand_band=("demand_band", "first"),
        earliest_known_price_date=("first_known_price_date", "min"),
        latest_known_price_date=("first_known_price_date", "max"),
        earliest_positive_sale_date=("first_positive_sale_date", "min"),
        latest_positive_sale_date=("first_positive_sale_date", "max"),
        maximum_store_prelaunch_days=("prelaunch_days", "max"),
    ).reset_index()
    product_launches["launch_year"] = product_launches["earliest_known_price_date"].dt.year
    year_counts = product_launches["launch_year"].value_counts().sort_index()
    summary = {
        "products": int(len(product_launches)),
        "earliest_known_price_date": product_launches["earliest_known_price_date"].min(),
        "latest_earliest_known_price_date": product_launches["earliest_known_price_date"].max(),
        "earliest_positive_sale_date": product_launches["earliest_positive_sale_date"].min(),
        "latest_earliest_positive_sale_date": product_launches["earliest_positive_sale_date"].max(),
        "launch_year_distribution": year_counts,
        "latest_10_products": product_launches.nlargest(10, "earliest_known_price_date")[
            ["item_id", "department", "demand_band", "earliest_known_price_date", "earliest_positive_sale_date", "maximum_store_prelaunch_days"]
        ],
        "recommended_training_boundary": "train each series from apparent availability rather than treating pre-launch zeros as demand",
    }
    return json_safe(summary), product_launches


def build_modeling_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    """Translate measured findings into bounded future-feature and risk recommendations."""

    return {
        "assessment": "ready for a controlled feature-engineering and chronological-split design after launch and missing-price policies are agreed",
        "supported_future_features": [
            {"feature": "store_id", "evidence": "store active-demand scales and SNAP associations differ"},
            {"feature": "item_id and department", "evidence": "demand bands, departments, and intermittency differ materially"},
            {"feature": "weekday", "evidence": "weekday mean demand varies in the active-period analysis"},
            {"feature": "month and chronological trend", "evidence": "monthly seasonality and multi-year structural movement are visible"},
            {"feature": "event indicators", "evidence": "event and non-event observations have different average demand, subject to confounding"},
            {"feature": "state-specific SNAP flag", "evidence": "SNAP/non-SNAP associations differ by store"},
            {"feature": "sell_price and price-change proxies", "evidence": "most item-store series change price and demand differs across price-change directions"},
            {"feature": "product age or active flag", "evidence": "pre-launch rows are structurally different from active zero-demand rows"},
            {"feature": "lagged sales and rolling statistics", "evidence": "daily demand has temporal structure and intermittency; must be calculated using past values only"},
        ],
        "leakage_risks": [
            "Random train/test splitting would mix future observations into training; use chronological splits.",
            "Rolling and lag features must be shifted so the target day never contributes to its own predictors.",
            "Demand bands and product statistics calculated over all 1,941 days contain future-period target information and must not be model inputs as currently defined.",
            "Imputation, scaling, encoders, and aggregate statistics must be fitted on training periods only.",
            "Launch boundaries based on first future positive sale can leak future knowledge; availability should use contemporaneously known price/assortment information in production.",
            "Future event or promotion fields are valid only when genuinely known at forecast creation time.",
        ],
        "forecasting_challenges": [
            "zero-heavy and intermittent active demand",
            "different product launch dates and active-history lengths",
            "store-level demand-scale differences",
            "department and product heterogeneity",
            "weekday, seasonal, event, and SNAP-associated patterns",
            "weekly price changes and confounded price-demand relationships",
            "extreme but potentially plausible retail spikes",
            "one-day, seven-day, and thirty-day forecast horizons",
        ],
        "decisions_before_next_stage": [
            "Choose whether model-ready rows begin at first known price for each item-store series.",
            "Choose a non-causal missing-price policy for any future active-period gaps; current V1 has none after launch.",
            "Define chronological validation windows for all three forecast horizons.",
            "Decide whether to use global models across series, per-series baselines, or both.",
            "Decide how intermittent-demand baselines will be compared with general forecasting models.",
        ],
    }


def compute_stage5_analysis(
    data: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Compute all Stage 5 summaries and reusable aggregate tables from one load."""

    validation = validate_v1_data(data, manifest)
    enriched = identify_active_periods(data)
    prelaunch, series_launches = summarize_prelaunch(enriched)
    series_metrics = compute_series_metrics(enriched)
    demand_band = summarize_group_demand(enriched, "demand_band", series_metrics)
    department = summarize_group_demand(enriched, "dept_id", series_metrics)
    store = summarize_group_demand(enriched, "store_id", series_metrics)

    store_department = enriched.groupby(["store_id", "dept_id"], observed=True)["sales"].sum().reset_index(name="total_sales")
    store_department["store_sales_share_pct"] = 100 * store_department["total_sales"] / store_department.groupby("store_id", observed=True)["total_sales"].transform("sum")

    time_tables = compute_time_tables(enriched)
    event_summary, event_comparison, event_types = analyze_events(enriched)
    snap_summary, snap_comparison = analyze_snap(enriched)
    price_summary, weekly_prices, price_direction, correlations = analyze_prices(enriched)
    intermittency = analyze_intermittency(series_metrics)
    outliers = analyze_outliers(enriched)
    product_launch, product_launches = analyze_product_launches(series_launches)

    slope = float(time_tables["monthly_trend_slope_units_per_month"].iloc[0, 0])
    monthly_total_mean = float(time_tables["monthly_total"]["monthly_sales"].mean())
    time_summary = {
        "monthly_linear_slope_units_per_month": slope,
        "monthly_slope_as_pct_of_mean": 100 * slope / monthly_total_mean,
        "yearly": time_tables["yearly"],
        "weekday": time_tables["weekday"],
        "monthly_overall": time_tables["monthly_overall"],
        "highest_daily_total_10": time_tables["daily_total"].nlargest(10, "sales"),
        "trend_caution": "aggregate growth mixes demand changes with product launches and active assortment; recurring within-year patterns are seasonality, while multi-year movement is trend",
    }

    summary: dict[str, Any] = {
        "stage": "Stage 5 - Exploratory Data Analysis & Modeling Readiness",
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "dataset_validation": validation,
        "prelaunch_vs_active": prelaunch,
        "overall_demand": {
            "all_rows": describe_sales(enriched["sales"]),
            "active_rows": describe_sales(enriched.loc[enriched["active_flag"], "sales"]),
        },
        "demand_band_analysis": demand_band,
        "department_analysis": department,
        "store_analysis": store,
        "store_department_mix": store_department,
        "time_analysis": time_summary,
        "event_analysis": event_summary,
        "snap_analysis": snap_summary,
        "price_analysis": price_summary,
        "intermittent_demand": intermittency,
        "outlier_analysis": outliers,
        "product_launch_analysis": product_launch,
    }
    summary["modeling_readiness"] = build_modeling_readiness(summary)

    tables = {
        "enriched": enriched,
        "series_launches": series_launches,
        "series_metrics": series_metrics,
        "demand_band": demand_band,
        "department": department,
        "store": store,
        "store_department": store_department,
        "event_comparison": event_comparison,
        "event_types": event_types,
        "snap_comparison": snap_comparison,
        "weekly_prices": weekly_prices,
        "price_direction": price_direction,
        "correlations": correlations,
        "product_launches": product_launches,
        **time_tables,
    }
    return json_safe(summary), tables


def run_stage5(
    data_path: Path,
    manifest_path: Path,
    report_dir: Path,
    figure_dir: Path,
) -> dict[str, Any]:
    """Run the complete Stage 5 EDA and persist only report artifacts."""

    data_hash_before = file_sha256(data_path)
    manifest_hash_before = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    print("Loading the frozen Version 1 dataset once ...", flush=True)
    data = load_v1_data(data_path)
    print("Identifying launch boundaries and calculating EDA aggregates ...", flush=True)
    summary, tables = compute_stage5_analysis(data, manifest)
    print("Creating the focused Stage 5 figure set ...", flush=True)
    figures = create_figures(tables, figure_dir)
    summary["figures"] = figures
    summary["performance"] = {
        "csv_load_count": 1,
        "vectorized_operations": True,
        "full_row_python_loops": False,
        "persistent_feature_dataset_created": False,
        "notes": [
            "categorical dtypes reduce repeated identifier memory",
            "daily/weekly/monthly aggregates are reused across report sections",
            "price analysis uses one row per item-store-week instead of repeated daily prices",
            "300 series are summarized through aggregates rather than plotted individually",
        ],
    }

    data_hash_after = file_sha256(data_path)
    manifest_hash_after = file_sha256(manifest_path)
    summary["source_integrity"] = {
        "interim_csv_sha256_before": data_hash_before,
        "interim_csv_sha256_after": data_hash_after,
        "manifest_sha256_before": manifest_hash_before,
        "manifest_sha256_after": manifest_hash_after,
        "unchanged": data_hash_before == data_hash_after and manifest_hash_before == manifest_hash_after,
    }
    if not summary["source_integrity"]["unchanged"]:
        raise RuntimeError("The frozen Stage 4 dataset or manifest changed during Stage 5.")

    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "stage5_summary.json"
    report_path = report_dir / "stage5_eda_report.md"
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path.write_text(render_stage5_report(summary), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage 5 command-line interface."""

    parser = argparse.ArgumentParser(
        description="Run SmartStock Stage 5 EDA and modeling-readiness analysis.",
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run Stage 5 and print a compact outcome summary."""

    args = build_parser().parse_args(argv)
    summary = run_stage5(args.data_path, args.manifest_path, args.report_dir, args.figure_dir)
    validation = summary["dataset_validation"]
    prelaunch = summary["prelaunch_vs_active"]
    print()
    print("Stage 5 EDA complete")
    print(f"Dataset: {validation['rows']:,} rows x {validation['columns']} columns")
    print(f"Pre-launch: {prelaunch['prelaunch_rows']:,} rows ({prelaunch['prelaunch_pct']:.2f}%)")
    print(f"Active zero demand: {prelaunch['active_zero_sales_pct']:.2f}%")
    print(f"Figures: {len(summary['figures'])}")
    print(f"Report: {(args.report_dir / 'stage5_eda_report.md').resolve()}")
    print(f"Summary: {(args.report_dir / 'stage5_summary.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
