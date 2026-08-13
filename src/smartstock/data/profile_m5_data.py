"""Profile the raw Walmart M5 files without cleaning or transforming them.

Stage 3 is intentionally read-only with respect to ``data/raw``. The sales file
remains wide: daily values are scanned in row chunks and summarized, never
permanently melted into a much larger long-format table.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from smartstock.data.stage3_report import render_stage3_report
except ModuleNotFoundError:  # Supports running this file directly from the repo root.
    from stage3_report import render_stage3_report


DEFAULT_DATA_DIR = Path("data") / "raw"
DEFAULT_REPORT_DIR = Path("reports")
SALES_FILENAME = "sales_train_evaluation.csv"
PRICES_FILENAME = "sell_prices.csv"
CALENDAR_FILENAME = "calendar.csv"
IDENTIFIER_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
FOODS_CATEGORY = "FOODS"


def find_daily_sales_columns(columns: Iterable[str]) -> list[str]:
    """Return columns named ``d_<integer>`` in numeric day order."""

    daily_columns = [
        column
        for column in columns
        if column.startswith("d_") and column.removeprefix("d_").isdigit()
    ]
    return sorted(daily_columns, key=lambda column: int(column.removeprefix("d_")))


def bytes_to_mib(size_bytes: int | float) -> float:
    """Convert bytes to mebibytes."""

    return round(float(size_bytes) / (1024**2), 2)


def value_counts_dict(series: pd.Series) -> dict[str, int]:
    """Return deterministic value counts with JSON-safe keys and values."""

    counts = series.value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def records_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a small aggregate DataFrame to JSON-safe records."""

    records: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        clean_record: dict[str, Any] = {}
        for key, value in record.items():
            if pd.isna(value):
                clean_record[str(key)] = None
            elif isinstance(value, (np.integer,)):
                clean_record[str(key)] = int(value)
            elif isinstance(value, (np.floating,)):
                clean_record[str(key)] = round(float(value), 6)
            elif isinstance(value, pd.Timestamp):
                clean_record[str(key)] = value.date().isoformat()
            else:
                clean_record[str(key)] = value
        records.append(clean_record)
    return records


def scan_daily_sales_values(
    sales_path: Path,
    daily_columns: Sequence[str],
    *,
    chunk_size: int = 1_000,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Scan daily sales once and retain only one compact metric row per CSV row."""

    total_values = 0
    zero_values = 0
    negative_values = 0
    missing_values = 0
    minimum: int | None = None
    maximum: int | None = None
    observed_dtypes: set[str] = set()

    total_units_parts: list[np.ndarray] = []
    zero_day_parts: list[np.ndarray] = []
    positive_day_parts: list[np.ndarray] = []
    first_positive_parts: list[np.ndarray] = []
    last_positive_parts: list[np.ndarray] = []

    reader = pd.read_csv(
        sales_path,
        usecols=list(daily_columns),
        dtype={column: np.int32 for column in daily_columns},
        chunksize=chunk_size,
    )

    for chunk in reader:
        observed_dtypes.update(str(dtype) for dtype in chunk.dtypes.unique())
        missing_values += int(chunk.isna().to_numpy().sum())
        values = chunk.to_numpy(dtype=np.int32, copy=False)

        total_values += int(values.size)
        zero_mask = values == 0
        positive_mask = values > 0
        negative_values += int(np.count_nonzero(values < 0))
        zero_values += int(np.count_nonzero(zero_mask))

        chunk_minimum = int(values.min())
        chunk_maximum = int(values.max())
        minimum = chunk_minimum if minimum is None else min(minimum, chunk_minimum)
        maximum = chunk_maximum if maximum is None else max(maximum, chunk_maximum)

        has_positive = positive_mask.any(axis=1)
        first_positive = np.where(has_positive, positive_mask.argmax(axis=1) + 1, -1)
        last_positive = np.where(
            has_positive,
            values.shape[1] - positive_mask[:, ::-1].argmax(axis=1),
            -1,
        )

        total_units_parts.append(values.sum(axis=1, dtype=np.int64))
        zero_day_parts.append(zero_mask.sum(axis=1, dtype=np.int32))
        positive_day_parts.append(positive_mask.sum(axis=1, dtype=np.int32))
        first_positive_parts.append(first_positive.astype(np.int32))
        last_positive_parts.append(last_positive.astype(np.int32))

    row_metrics = pd.DataFrame(
        {
            "total_units": np.concatenate(total_units_parts),
            "zero_days": np.concatenate(zero_day_parts),
            "positive_days": np.concatenate(positive_day_parts),
            "first_positive_day": np.concatenate(first_positive_parts),
            "last_positive_day": np.concatenate(last_positive_parts),
        }
    )

    value_profile = {
        "total_values": total_values,
        "missing_values": missing_values,
        "negative_values": negative_values,
        "zero_values": zero_values,
        "zero_prevalence_pct": round(100 * zero_values / total_values, 4),
        "minimum": minimum,
        "maximum": maximum,
        "profiling_dtypes": sorted(observed_dtypes),
    }
    return value_profile, row_metrics


def profile_calendar(calendar: pd.DataFrame) -> dict[str, Any]:
    """Calculate calendar structure, event, missingness, and SNAP statistics."""

    event_name_columns = ["event_name_1", "event_name_2"]
    event_type_columns = ["event_type_1", "event_type_2"]
    event_type_values = pd.concat(
        [calendar[column].dropna() for column in event_type_columns],
        ignore_index=True,
    )

    snap_coverage: dict[str, dict[str, float | int]] = {}
    for state, column in (("California", "snap_CA"), ("Texas", "snap_TX"), ("Wisconsin", "snap_WI")):
        enabled_days = int(calendar[column].eq(1).sum())
        snap_coverage[state] = {
            "enabled_days": enabled_days,
            "coverage_pct": round(100 * enabled_days / len(calendar), 4),
        }

    return {
        "rows": int(calendar.shape[0]),
        "columns": int(calendar.shape[1]),
        "memory_mib": bytes_to_mib(calendar.memory_usage(deep=True).sum()),
        "column_dtypes": {column: str(dtype) for column, dtype in calendar.dtypes.items()},
        "date_min": calendar["date"].min().date().isoformat(),
        "date_max": calendar["date"].max().date().isoformat(),
        "unique_d": int(calendar["d"].nunique()),
        "d_is_unique": bool(calendar["d"].is_unique),
        "unique_weeks": int(calendar["wm_yr_wk"].nunique()),
        "week_min": int(calendar["wm_yr_wk"].min()),
        "week_max": int(calendar["wm_yr_wk"].max()),
        "years": sorted(int(value) for value in calendar["year"].unique()),
        "months": sorted(int(value) for value in calendar["month"].unique()),
        "weekdays": [str(value) for value in calendar["weekday"].drop_duplicates()],
        "missing_values": {column: int(value) for column, value in calendar.isna().sum().items()},
        "event_value_counts": {
            column: value_counts_dict(calendar[column].dropna())
            for column in event_name_columns + event_type_columns
        },
        "event_type_distribution": value_counts_dict(event_type_values),
        "event_days": int(calendar[event_name_columns].notna().any(axis=1).sum()),
        "two_event_days": int(calendar[event_name_columns].notna().all(axis=1).sum()),
        "snap_coverage": snap_coverage,
    }


def load_calendar(calendar_path: Path) -> pd.DataFrame:
    """Load the small calendar table with explicit, compact dtypes."""

    return pd.read_csv(
        calendar_path,
        parse_dates=["date"],
        dtype={
            "wm_yr_wk": np.int32,
            "weekday": "category",
            "wday": np.int8,
            "month": np.int8,
            "year": np.int16,
            "d": "string",
            "event_name_1": "category",
            "event_type_1": "category",
            "event_name_2": "category",
            "event_type_2": "category",
            "snap_CA": np.int8,
            "snap_TX": np.int8,
            "snap_WI": np.int8,
        },
    )


def load_sales_identifiers(sales_path: Path) -> pd.DataFrame:
    """Load only the six hierarchy/key columns from the wide sales table."""

    return pd.read_csv(
        sales_path,
        usecols=IDENTIFIER_COLUMNS,
        dtype={column: "category" for column in IDENTIFIER_COLUMNS},
    )


def profile_sales(
    sales_path: Path,
    sales_ids: pd.DataFrame,
    daily_columns: Sequence[str],
    row_metrics: pd.DataFrame,
    value_profile: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Combine identifier and chunked value findings into the sales profile."""

    if len(sales_ids) != len(row_metrics):
        raise ValueError("Identifier rows and sales metric rows do not align.")

    metrics = pd.concat([sales_ids.reset_index(drop=True), row_metrics], axis=1)
    identifier_memory = int(sales_ids.memory_usage(deep=True).sum())
    rows = int(len(sales_ids))
    day_count = int(len(daily_columns))
    default_daily_memory = rows * day_count * np.dtype(np.int64).itemsize
    optimized_daily_memory = rows * day_count * np.dtype(np.int32).itemsize

    inferred_sample = pd.read_csv(
        sales_path,
        usecols=[daily_columns[0], daily_columns[-1]],
        nrows=200,
    )
    daily_numbers = [int(column.removeprefix("d_")) for column in daily_columns]
    expected_numbers = list(range(daily_numbers[0], daily_numbers[-1] + 1))

    category_rows = value_counts_dict(sales_ids["cat_id"])
    department_rows = value_counts_dict(sales_ids["dept_id"])
    store_rows = value_counts_dict(sales_ids["store_id"])
    state_rows = value_counts_dict(sales_ids["state_id"])

    category_items = (
        sales_ids.groupby("cat_id", observed=True)["item_id"]
        .nunique()
        .sort_index()
    )
    department_items = (
        sales_ids.groupby("dept_id", observed=True)["item_id"]
        .nunique()
        .sort_index()
    )

    item_summary = metrics.groupby("item_id", observed=True).agg(
        store_count=("store_id", "nunique"),
        total_units=("total_units", "sum"),
        zero_days=("zero_days", "sum"),
        first_positive_day=(
            "first_positive_day",
            lambda values: int(values[values >= 0].min()) if (values >= 0).any() else -1,
        ),
    )
    item_summary["zero_prevalence_pct"] = (
        100 * item_summary["zero_days"] / (item_summary["store_count"] * day_count)
    )

    food_item_ids = set(
        sales_ids.loc[sales_ids["cat_id"].astype("string").eq(FOODS_CATEGORY), "item_id"]
        .astype("string")
        .unique()
    )
    food_item_summary = item_summary.loc[
        item_summary.index.astype("string").isin(food_item_ids)
    ]

    store_summary = metrics.groupby("store_id", observed=True).agg(
        state_id=("state_id", "first"),
        item_rows=("item_id", "size"),
        unique_items=("item_id", "nunique"),
        total_units=("total_units", "sum"),
        zero_days=("zero_days", "sum"),
    )
    store_summary["zero_prevalence_pct"] = (
        100 * store_summary["zero_days"] / (store_summary["item_rows"] * day_count)
    )
    store_summary = store_summary.reset_index()
    store_summary["store_id"] = store_summary["store_id"].astype("string")
    store_summary["state_id"] = store_summary["state_id"].astype("string")

    profile = {
        "rows": rows,
        "columns": len(IDENTIFIER_COLUMNS) + day_count,
        "file_size_mib": bytes_to_mib(sales_path.stat().st_size),
        "identifier_columns": list(IDENTIFIER_COLUMNS),
        "identifier_memory_mib": bytes_to_mib(identifier_memory),
        "estimated_default_dataframe_memory_mib": bytes_to_mib(
            identifier_memory + default_daily_memory
        ),
        "estimated_int32_dataframe_memory_mib": bytes_to_mib(
            identifier_memory + optimized_daily_memory
        ),
        "daily_column_count": day_count,
        "first_daily_column": daily_columns[0],
        "last_daily_column": daily_columns[-1],
        "daily_columns_contiguous": daily_numbers == expected_numbers,
        "source_inferred_daily_dtypes": sorted(str(dtype) for dtype in inferred_sample.dtypes.unique()),
        "unique_items": int(sales_ids["item_id"].nunique()),
        "unique_stores": int(sales_ids["store_id"].nunique()),
        "unique_states": int(sales_ids["state_id"].nunique()),
        "unique_categories": int(sales_ids["cat_id"].nunique()),
        "unique_departments": int(sales_ids["dept_id"].nunique()),
        "category_row_counts": category_rows,
        "category_item_counts": {str(key): int(value) for key, value in category_items.items()},
        "department_row_counts": department_rows,
        "department_item_counts": {str(key): int(value) for key, value in department_items.items()},
        "store_row_counts": store_rows,
        "state_row_counts": state_rows,
        "unique_item_store_combinations": int(
            sales_ids[["item_id", "store_id"]].drop_duplicates().shape[0]
        ),
        "id_is_unique": bool(sales_ids["id"].is_unique),
        "item_store_is_unique": bool(
            ~sales_ids.duplicated(subset=["item_id", "store_id"]).any()
        ),
        "identifier_missing_values": {
            column: int(value)
            for column, value in sales_ids[IDENTIFIER_COLUMNS].isna().sum().items()
        },
        "sales_values": value_profile,
        "items_at_least_95_pct_zero": int(item_summary["zero_prevalence_pct"].ge(95).sum()),
        "foods_items_at_least_95_pct_zero": int(
            food_item_summary["zero_prevalence_pct"].ge(95).sum()
        ),
        "foods_items_first_positive_after_day_365": int(
            food_item_summary["first_positive_day"].gt(365).sum()
        ),
        "foods_items_never_positive": int(
            food_item_summary["first_positive_day"].eq(-1).sum()
        ),
        "highest_zero_food_items": records_from_frame(
            food_item_summary.reset_index()
            .assign(item_id=lambda frame: frame["item_id"].astype("string"))
            .sort_values(["zero_prevalence_pct", "item_id"], ascending=[False, True])
            .head(10)[["item_id", "zero_prevalence_pct", "first_positive_day", "total_units"]]
        ),
        "store_sales_summary": records_from_frame(
            store_summary.sort_values("store_id")
        ),
    }
    return profile, metrics


def profile_hierarchy(sales_ids: pd.DataFrame) -> dict[str, Any]:
    """Verify the geographic and product hierarchies in the sales identifiers."""

    food_rows = sales_ids[sales_ids["cat_id"].astype("string").eq(FOODS_CATEGORY)]
    food_department_items = (
        food_rows.groupby("dept_id", observed=True)["item_id"]
        .nunique()
        .sort_index()
    )
    food_store_items = (
        food_rows.groupby("store_id", observed=True)["item_id"]
        .nunique()
        .sort_index()
    )
    stores_per_food_item = food_rows.groupby("item_id", observed=True)["store_id"].nunique()

    stores_per_state = (
        sales_ids[["state_id", "store_id"]]
        .drop_duplicates()
        .groupby("state_id", observed=True)["store_id"]
        .nunique()
        .sort_index()
    )

    return {
        "stores_per_state": {str(key): int(value) for key, value in stores_per_state.items()},
        "store_maps_to_one_state": bool(
            sales_ids.groupby("store_id", observed=True)["state_id"].nunique().eq(1).all()
        ),
        "department_maps_to_one_category": bool(
            sales_ids.groupby("dept_id", observed=True)["cat_id"].nunique().eq(1).all()
        ),
        "item_maps_to_one_department": bool(
            sales_ids.groupby("item_id", observed=True)["dept_id"].nunique().eq(1).all()
        ),
        "item_maps_to_one_category": bool(
            sales_ids.groupby("item_id", observed=True)["cat_id"].nunique().eq(1).all()
        ),
        "foods_unique_items": int(food_rows["item_id"].nunique()),
        "foods_departments": [str(value) for value in food_rows["dept_id"].drop_duplicates().sort_values()],
        "foods_items_by_department": {
            str(key): int(value) for key, value in food_department_items.items()
        },
        "foods_items_by_store": {str(key): int(value) for key, value in food_store_items.items()},
        "foods_items_in_multiple_stores": int(stores_per_food_item.gt(1).sum()),
        "foods_items_in_every_store": int(
            stores_per_food_item.eq(sales_ids["store_id"].nunique()).sum()
        ),
        "minimum_stores_per_food_item": int(stores_per_food_item.min()),
        "maximum_stores_per_food_item": int(stores_per_food_item.max()),
    }


def load_prices(prices_path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load prices once using compact dtypes and return default inferred dtypes too."""

    inferred_sample = pd.read_csv(prices_path, nrows=500)
    inferred_dtypes = {column: str(dtype) for column, dtype in inferred_sample.dtypes.items()}
    prices = pd.read_csv(
        prices_path,
        dtype={
            "store_id": "category",
            "item_id": "category",
            "wm_yr_wk": np.int32,
            "sell_price": np.float32,
        },
    )
    return prices, inferred_dtypes


def profile_prices(
    prices_path: Path,
    prices: pd.DataFrame,
    inferred_dtypes: dict[str, str],
    sales_ids: pd.DataFrame,
    calendar_week_count: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calculate price statistics and compact store/category coverage tables."""

    optimized_raw_memory = int(prices.memory_usage(deep=True).sum())
    duplicate_rows = int(prices.duplicated().sum())
    duplicate_keys = int(
        prices.duplicated(subset=["store_id", "item_id", "wm_yr_wk"]).sum()
    )

    item_hierarchy = (
        sales_ids[["item_id", "cat_id", "dept_id"]]
        .drop_duplicates("item_id")
        .copy()
    )
    item_to_category = dict(
        zip(item_hierarchy["item_id"].astype("string"), item_hierarchy["cat_id"].astype("string"))
    )
    item_to_department = dict(
        zip(item_hierarchy["item_id"].astype("string"), item_hierarchy["dept_id"].astype("string"))
    )
    prices["cat_id"] = prices["item_id"].map(item_to_category).astype("category")
    prices["dept_id"] = prices["item_id"].map(item_to_department).astype("category")

    store_coverage = prices.groupby("store_id", observed=True).agg(
        price_records=("sell_price", "size"),
        unique_items=("item_id", "nunique"),
        unique_weeks=("wm_yr_wk", "nunique"),
        earliest_week=("wm_yr_wk", "min"),
        latest_week=("wm_yr_wk", "max"),
    )
    weeks_per_store_item = prices.groupby(
        ["store_id", "item_id"], observed=True
    )["wm_yr_wk"].nunique()
    store_coverage["median_weeks_per_item"] = weeks_per_store_item.groupby(level=0).median()
    store_coverage["raw_week_grid_coverage_pct"] = (
        100
        * store_coverage["price_records"]
        / (store_coverage["unique_items"] * calendar_week_count)
    )
    store_coverage = store_coverage.reset_index()
    store_coverage["store_id"] = store_coverage["store_id"].astype("string")

    category_coverage = prices.groupby("cat_id", observed=True).agg(
        price_records=("sell_price", "size"),
        unique_items=("item_id", "nunique"),
        unique_stores=("store_id", "nunique"),
        unique_weeks=("wm_yr_wk", "nunique"),
    )
    category_pairs = (
        prices[["cat_id", "store_id", "item_id"]]
        .drop_duplicates()
        .groupby("cat_id", observed=True)
        .size()
    )
    category_coverage["item_store_pairs"] = category_pairs
    category_coverage["raw_week_grid_coverage_pct"] = (
        100
        * category_coverage["price_records"]
        / (category_coverage["item_store_pairs"] * calendar_week_count)
    )
    category_coverage = category_coverage.reset_index()
    category_coverage["cat_id"] = category_coverage["cat_id"].astype("string")

    distinct_prices = prices.groupby("item_id", observed=True)["sell_price"].nunique()
    item_store_pairs = prices[["store_id", "item_id"]].drop_duplicates()
    unique_composite_records = int(len(prices) - duplicate_keys)
    possible_raw_week_grid = int(len(item_store_pairs) * calendar_week_count)
    missing_raw_week_positions = int(possible_raw_week_grid - unique_composite_records)

    profile = {
        "rows": int(prices.shape[0]),
        "columns": 4,
        "file_size_mib": bytes_to_mib(prices_path.stat().st_size),
        "optimized_memory_mib": bytes_to_mib(optimized_raw_memory),
        "source_inferred_dtypes": inferred_dtypes,
        "profiling_dtypes": {
            column: str(prices[column].dtype)
            for column in ["store_id", "item_id", "wm_yr_wk", "sell_price"]
        },
        "unique_stores": int(prices["store_id"].nunique()),
        "unique_items": int(prices["item_id"].nunique()),
        "unique_weeks": int(prices["wm_yr_wk"].nunique()),
        "earliest_week": int(prices["wm_yr_wk"].min()),
        "latest_week": int(prices["wm_yr_wk"].max()),
        "minimum_price": round(float(prices["sell_price"].min()), 4),
        "maximum_price": round(float(prices["sell_price"].max()), 4),
        "mean_price": round(float(prices["sell_price"].mean()), 4),
        "median_price": round(float(prices["sell_price"].median()), 4),
        "missing_values": {
            column: int(value)
            for column, value in prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]]
            .isna()
            .sum()
            .items()
        },
        "negative_prices": int(prices["sell_price"].lt(0).sum()),
        "zero_prices": int(prices["sell_price"].eq(0).sum()),
        "duplicate_rows": duplicate_rows,
        "duplicate_composite_keys": duplicate_keys,
        "unique_composite_price_records": unique_composite_records,
        "unique_item_store_pairs": int(len(item_store_pairs)),
        "possible_raw_week_grid_positions": possible_raw_week_grid,
        "missing_raw_week_grid_positions": missing_raw_week_positions,
        "overall_raw_week_grid_coverage_pct": round(
            100 * unique_composite_records / possible_raw_week_grid,
            4,
        ),
        "store_coverage": records_from_frame(store_coverage.sort_values("store_id")),
        "category_coverage": records_from_frame(category_coverage.sort_values("cat_id")),
        "distinct_prices_per_item": {
            "minimum": int(distinct_prices.min()),
            "median": round(float(distinct_prices.median()), 2),
            "mean": round(float(distinct_prices.mean()), 2),
            "maximum": int(distinct_prices.max()),
            "items_with_price_changes": int(distinct_prices.gt(1).sum()),
            "items_with_one_distinct_price": int(distinct_prices.eq(1).sum()),
        },
    }
    return profile, store_coverage


def profile_relationships(
    sales_ids: pd.DataFrame,
    daily_columns: Sequence[str],
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
) -> dict[str, Any]:
    """Verify key coverage and report join cardinality evidence."""

    sales_days = set(daily_columns)
    calendar_days = set(calendar["d"].astype("string"))
    price_weeks = set(int(value) for value in prices["wm_yr_wk"].unique())
    calendar_weeks = set(int(value) for value in calendar["wm_yr_wk"].unique())

    sales_pairs = pd.MultiIndex.from_frame(
        sales_ids[["store_id", "item_id"]]
        .astype("string")
        .drop_duplicates()
    )
    price_pairs = pd.MultiIndex.from_frame(
        prices[["store_id", "item_id"]]
        .drop_duplicates()
        .astype("string")
    )
    sales_items = set(sales_ids["item_id"].astype("string"))
    price_items = set(str(value) for value in prices["item_id"].cat.categories)
    sales_stores = set(sales_ids["store_id"].astype("string"))
    price_stores = set(str(value) for value in prices["store_id"].cat.categories)

    sales_calendar = calendar[calendar["d"].isin(daily_columns)].sort_values("date")
    days_per_week = calendar.groupby("wm_yr_wk")["d"].nunique()

    return {
        "sales_calendar": {
            "sales_day_columns": len(sales_days),
            "matched_sales_days": len(sales_days & calendar_days),
            "unmatched_sales_days": sorted(sales_days - calendar_days),
            "calendar_days_not_in_sales": len(calendar_days - sales_days),
            "sales_date_min": sales_calendar["date"].min().date().isoformat(),
            "sales_date_max": sales_calendar["date"].max().date().isoformat(),
            "calendar_d_is_unique": bool(calendar["d"].is_unique),
            "key_cardinality": "one-to-one for day labels; many sales observations to one calendar day after reshaping",
        },
        "prices_calendar": {
            "price_weeks": len(price_weeks),
            "matched_price_weeks": len(price_weeks & calendar_weeks),
            "unmatched_price_weeks": sorted(price_weeks - calendar_weeks),
            "calendar_weeks_without_prices": sorted(calendar_weeks - price_weeks),
            "calendar_days_per_week_min": int(days_per_week.min()),
            "calendar_days_per_week_max": int(days_per_week.max()),
            "key_cardinality": "many-to-many on wm_yr_wk alone; one weekly item-store price expands to multiple calendar days",
        },
        "sales_prices": {
            "matched_items": len(sales_items & price_items),
            "sales_items_without_prices": sorted(sales_items - price_items),
            "price_items_without_sales": sorted(price_items - sales_items),
            "matched_stores": len(sales_stores & price_stores),
            "sales_stores_without_prices": sorted(sales_stores - price_stores),
            "price_stores_without_sales": sorted(price_stores - sales_stores),
            "matched_item_store_pairs": len(sales_pairs.intersection(price_pairs)),
            "sales_item_store_pairs_without_prices": len(sales_pairs.difference(price_pairs)),
            "price_item_store_pairs_without_sales": len(price_pairs.difference(sales_pairs)),
            "price_composite_key_is_unique": bool(
                ~prices.duplicated(subset=["store_id", "item_id", "wm_yr_wk"]).any()
            ),
            "key_cardinality": "one sales item-store row to many weekly price rows; after adding week to daily sales, many sales rows to one composite price key",
        },
    }


def candidate_product_screen(
    metrics: pd.DataFrame,
    prices: pd.DataFrame,
    candidate_stores: Sequence[str],
    *,
    day_count: int,
    calendar_week_count: int,
) -> dict[str, Any]:
    """Apply transparent, non-final product coverage screens to a store strategy."""

    candidate_store_set = set(candidate_stores)
    food_sales = metrics[
        metrics["cat_id"].eq(FOODS_CATEGORY)
        & metrics["store_id"].isin(candidate_store_set)
    ].copy()
    food_sales["row_zero_prevalence_pct"] = 100 * food_sales["zero_days"] / day_count
    sales_screen = food_sales.groupby("item_id", observed=True).agg(
        stores_present=("store_id", "nunique"),
        stores_active_by_day_365=("first_positive_day", lambda values: int(values.between(1, 365).sum())),
        stores_below_95_pct_zero=("row_zero_prevalence_pct", lambda values: int(values.lt(95).sum())),
    )

    food_prices = prices[
        prices["cat_id"].eq(FOODS_CATEGORY)
        & prices["store_id"].isin(candidate_store_set)
    ]
    item_store_price_weeks = food_prices.groupby(
        ["item_id", "store_id"], observed=True
    )["wm_yr_wk"].nunique()
    price_screen = item_store_price_weeks.groupby(level=0).agg(
        stores_with_any_price="size",
        minimum_price_weeks="min",
    )
    price_screen["minimum_raw_week_coverage_pct"] = (
        100 * price_screen["minimum_price_weeks"] / calendar_week_count
    )

    combined = sales_screen.join(price_screen, how="left").fillna(0)
    store_count = len(candidate_stores)
    all_sales_stores = combined["stores_present"].eq(store_count)
    all_price_stores = combined["stores_with_any_price"].eq(store_count)
    active_early_all = combined["stores_active_by_day_365"].eq(store_count)
    below_zero_threshold_all = combined["stores_below_95_pct_zero"].eq(store_count)
    price_80_all = all_price_stores & combined["minimum_raw_week_coverage_pct"].ge(80)

    return {
        "stores": list(candidate_stores),
        "foods_items_in_all_sales_stores": int(all_sales_stores.sum()),
        "foods_items_with_any_price_in_all_stores": int(all_price_stores.sum()),
        "foods_items_with_at_least_80_pct_raw_week_coverage_in_all_stores": int(price_80_all.sum()),
        "foods_items_active_by_day_365_in_all_stores": int(active_early_all.sum()),
        "foods_items_below_95_pct_zero_in_all_stores": int(below_zero_threshold_all.sum()),
        "foods_items_meeting_all_indicative_screens": int(
            (all_sales_stores & price_80_all & active_early_all & below_zero_threshold_all).sum()
        ),
    }


def recommend_subset_strategies(
    sales_profile: dict[str, Any],
    hierarchy_profile: dict[str, Any],
    metrics: pd.DataFrame,
    price_store_coverage: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    day_count: int,
    calendar_week_count: int,
) -> list[dict[str, Any]]:
    """Build three evidence-backed store strategies without choosing a final subset."""

    sales_stores = pd.DataFrame(sales_profile["store_sales_summary"])
    store_summary = price_store_coverage.merge(sales_stores, on="store_id", how="left")
    store_summary["state_id"] = store_summary["state_id"].astype("string")

    cross_state_stores = (
        store_summary.sort_values(
            ["state_id", "raw_week_grid_coverage_pct", "zero_prevalence_pct", "store_id"],
            ascending=[True, False, True, True],
        )
        .groupby("state_id", sort=True)
        .head(1)["store_id"]
        .astype(str)
        .tolist()
    )

    eligible_states = [
        state
        for state, count in hierarchy_profile["stores_per_state"].items()
        if count >= 3
    ]
    state_coverage = (
        store_summary[store_summary["state_id"].isin(eligible_states)]
        .groupby("state_id")["raw_week_grid_coverage_pct"]
        .mean()
        .sort_values(ascending=False)
    )
    comparable_state = str(state_coverage.index[0])
    same_state_stores = (
        store_summary[store_summary["state_id"].eq(comparable_state)]
        .sort_values(
            ["raw_week_grid_coverage_pct", "zero_prevalence_pct", "store_id"],
            ascending=[False, True, True],
        )
        .head(3)["store_id"]
        .astype(str)
        .tolist()
    )

    coverage_stores = (
        store_summary.sort_values(
            ["raw_week_grid_coverage_pct", "zero_prevalence_pct", "store_id"],
            ascending=[False, True, True],
        )
        .head(3)["store_id"]
        .astype(str)
        .tolist()
    )

    strategy_specs = [
        (
            "A. Cross-state representative stores",
            cross_state_stores,
            "Uses the best raw price-coverage store from each state, adding geographic and SNAP-policy diversity.",
            "State effects become mixed with store effects, so direct store comparisons are less controlled.",
        ),
        (
            f"B. Same-state comparable stores ({comparable_state})",
            same_state_stores,
            "Keeps state-level conditions and the state-specific SNAP flag constant while comparing stores.",
            "It reduces geographic diversity and may not generalize as well to the other states.",
        ),
        (
            "C. Highest raw price-coverage stores",
            coverage_stores,
            "Prioritizes complete weekly price histories and therefore simpler later joins.",
            "Coverage-driven store selection can overrepresent one geography and is not a business-representative sample by itself.",
        ),
    ]

    recommendations: list[dict[str, Any]] = []
    for name, stores, rationale, tradeoff in strategy_specs:
        evidence = store_summary[store_summary["store_id"].isin(stores)][
            ["store_id", "state_id", "raw_week_grid_coverage_pct", "zero_prevalence_pct"]
        ].sort_values("store_id")
        recommendations.append(
            {
                "name": name,
                "stores": stores,
                "rationale": rationale,
                "tradeoff": tradeoff,
                "store_evidence": records_from_frame(evidence),
                "product_screen": candidate_product_screen(
                    metrics,
                    prices,
                    stores,
                    day_count=day_count,
                    calendar_week_count=calendar_week_count,
                ),
            }
        )
    return recommendations


def build_quality_observations(profile: dict[str, Any]) -> list[dict[str, str]]:
    """Classify observed conditions without changing any data."""

    sales = profile["sales"]
    prices = profile["prices"]
    calendar = profile["calendar"]
    relationships = profile["relationships"]
    return [
        {
            "classification": "Expected Dataset Behavior",
            "observation": f"Daily demand is zero-heavy: {sales['sales_values']['zero_prevalence_pct']:.2f}% of sales cells are zero.",
            "implication": "Zero usually means no recorded unit sales, not missing data; later modeling must preserve it.",
        },
        {
            "classification": "Expected Dataset Behavior",
            "observation": f"Event fields are sparse: event_name_1 is null on {calendar['missing_values']['event_name_1']:,} of {calendar['rows']:,} days and event_name_2 is null on {calendar['missing_values']['event_name_2']:,} days.",
            "implication": "Null event fields normally mean no named event and should not be filled blindly.",
        },
        {
            "classification": "Expected Dataset Behavior",
            "observation": f"The price table covers {prices['overall_raw_week_grid_coverage_pct']:.2f}% of the raw item-store-week grid, leaving {prices['missing_raw_week_grid_positions']:,} positions without a price row; store coverage varies because items enter or leave assortment over time.",
            "implication": "A missing weekly price may be pre-launch or out-of-assortment behavior, not a defective record.",
        },
        {
            "classification": "Requires Investigation",
            "observation": f"{sales['foods_items_first_positive_after_day_365']:,} FOODS items first record positive demand after day 365 across the full store set.",
            "implication": "Product introduction timing should influence the final subset and later train/test design.",
        },
        {
            "classification": "Requires Investigation",
            "observation": f"{prices['distinct_prices_per_item']['items_with_price_changes']:,} items have more than one distinct observed price.",
            "implication": "Price changes may later support promotion/price features, but no promotion effect is inferred in Stage 3.",
        },
        {
            "classification": "Requires Investigation",
            "observation": f"The calendar contains {relationships['sales_calendar']['calendar_days_not_in_sales']:,} day labels beyond the evaluation sales horizon.",
            "implication": "Later joins must distinguish the observed sales period from future calendar rows.",
        },
        {
            "classification": "Requires Investigation",
            "observation": f"The wide sales table has {sales['daily_column_count']:,} day columns and an estimated default pandas footprint of about {sales['estimated_default_dataframe_memory_mib']:.1f} MiB.",
            "implication": "Later reshaping can multiply row count dramatically and needs an explicit memory-aware design.",
        },
        {
            "classification": "Potential Data Quality Issue" if sales["sales_values"]["negative_values"] else "Expected Dataset Behavior",
            "observation": f"Negative sales values found: {sales['sales_values']['negative_values']:,}.",
            "implication": "No correction was made; any nonzero result would require source investigation.",
        },
        {
            "classification": "Potential Data Quality Issue" if prices["duplicate_composite_keys"] else "Expected Dataset Behavior",
            "observation": f"Duplicate (store_id, item_id, wm_yr_wk) price keys found: {prices['duplicate_composite_keys']:,}.",
            "implication": "The future daily sales-to-price join is safe only when this composite key is unique.",
        },
    ]


def create_stage3_profile(
    data_dir: Path,
    *,
    chunk_size: int = 1_000,
) -> dict[str, Any]:
    """Create the complete Stage 3 profile from the three official M5 CSVs."""

    sales_path = data_dir / SALES_FILENAME
    prices_path = data_dir / PRICES_FILENAME
    calendar_path = data_dir / CALENDAR_FILENAME
    missing_files = [
        str(path) for path in (sales_path, prices_path, calendar_path) if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError("Missing required raw files: " + ", ".join(missing_files))

    sales_header = pd.read_csv(sales_path, nrows=0).columns.tolist()
    daily_columns = find_daily_sales_columns(sales_header)
    if not daily_columns:
        raise ValueError("No d_<number> sales columns were found.")

    print("Loading and profiling calendar.csv ...", flush=True)
    calendar = load_calendar(calendar_path)
    calendar_profile = profile_calendar(calendar)

    print("Loading sales identifiers ...", flush=True)
    sales_ids = load_sales_identifiers(sales_path)
    print("Scanning wide daily sales values in chunks ...", flush=True)
    value_profile, row_metrics = scan_daily_sales_values(
        sales_path,
        daily_columns,
        chunk_size=chunk_size,
    )
    sales_profile, metrics = profile_sales(
        sales_path,
        sales_ids,
        daily_columns,
        row_metrics,
        value_profile,
    )
    hierarchy_profile = profile_hierarchy(sales_ids)

    print("Loading and profiling sell_prices.csv with compact dtypes ...", flush=True)
    prices, inferred_price_dtypes = load_prices(prices_path)
    prices_profile, price_store_coverage = profile_prices(
        prices_path,
        prices,
        inferred_price_dtypes,
        sales_ids,
        calendar_profile["unique_weeks"],
    )

    print("Verifying dataset keys and developing subset strategy evidence ...", flush=True)
    relationships = profile_relationships(sales_ids, daily_columns, prices, calendar)
    subset_strategies = recommend_subset_strategies(
        sales_profile,
        hierarchy_profile,
        metrics,
        price_store_coverage,
        prices,
        day_count=len(daily_columns),
        calendar_week_count=calendar_profile["unique_weeks"],
    )

    profile: dict[str, Any] = {
        "stage": "Stage 3 - Data Understanding & Profiling",
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "data_directory": str(data_dir.resolve()),
        "performance": {
            "sales_chunk_size_rows": chunk_size,
            "full_sales_melt_performed": False,
            "sales_daily_scan_dtype": "int32",
            "sales_loading_method": "identifier-only DataFrame plus one chunked scan of daily columns",
            "prices_loading_method": "one optimized DataFrame using categorical IDs, int32 weeks, and float32 prices",
        },
        "sales": sales_profile,
        "prices": prices_profile,
        "calendar": calendar_profile,
        "hierarchy": hierarchy_profile,
        "relationships": relationships,
        "subset_strategies": subset_strategies,
    }
    profile["quality_observations"] = build_quality_observations(profile)
    return profile


def save_profile_outputs(profile: dict[str, Any], report_dir: Path) -> tuple[Path, Path]:
    """Save the human-readable Markdown report and machine-readable JSON profile."""

    report_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = report_dir / "stage3_data_profile.md"
    json_path = report_dir / "stage3_profile_summary.json"
    markdown_path.write_text(render_stage3_report(profile), encoding="utf-8")
    json_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    return markdown_path, json_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(
        description="Profile the three raw M5 datasets and save the Stage 3 reports.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the raw M5 CSV files (default: data/raw).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Output directory for Stage 3 reports (default: reports).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000,
        help="Number of sales rows per daily-value scan chunk (default: 1000).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the profiler and save its reports."""

    args = build_parser().parse_args(argv)
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")
    profile = create_stage3_profile(args.data_dir, chunk_size=args.chunk_size)
    markdown_path, json_path = save_profile_outputs(profile, args.report_dir)

    sales = profile["sales"]
    prices = profile["prices"]
    calendar = profile["calendar"]
    print()
    print("Stage 3 profiling complete")
    print(f"Sales: {sales['rows']:,} rows x {sales['columns']:,} columns")
    print(f"Prices: {prices['rows']:,} rows across {prices['unique_weeks']:,} weeks")
    print(f"Calendar: {calendar['date_min']} to {calendar['date_max']}")
    print(f"Markdown report: {markdown_path.resolve()}")
    print(f"JSON profile: {json_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
