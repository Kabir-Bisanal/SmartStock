"""Build the controlled SmartStock Version 1 long-format working dataset."""

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
    from smartstock.data.profile_m5_data import find_daily_sales_columns
    from smartstock.data.select_v1_subset import (
        MINIMUM_ACTIVE_PERIOD_PRICE_COVERAGE,
        MINIMUM_REMAINING_HISTORY_DAYS,
        RANDOM_SEED,
        SELECTED_CATEGORY,
        SELECTED_STORES,
        SELECTION_QUOTAS,
        ZERO_PREVALENCE_THRESHOLD,
        aggregate_item_sales_metrics,
        assign_demand_bands,
        compute_active_period_price_coverage,
        evaluate_eligibility,
        independent_failure_counts,
        select_stratified_items,
        validate_quota_availability,
    )
    from smartstock.data.stage4_report import render_stage4_report
except ModuleNotFoundError:  # Supports direct execution from the repository root.
    from profile_m5_data import find_daily_sales_columns
    from select_v1_subset import (
        MINIMUM_ACTIVE_PERIOD_PRICE_COVERAGE,
        MINIMUM_REMAINING_HISTORY_DAYS,
        RANDOM_SEED,
        SELECTED_CATEGORY,
        SELECTED_STORES,
        SELECTION_QUOTAS,
        ZERO_PREVALENCE_THRESHOLD,
        aggregate_item_sales_metrics,
        assign_demand_bands,
        compute_active_period_price_coverage,
        evaluate_eligibility,
        independent_failure_counts,
        select_stratified_items,
        validate_quota_availability,
    )
    from stage4_report import render_stage4_report


DEFAULT_DATA_DIR = Path("data") / "raw"
DEFAULT_INTERIM_DIR = Path("data") / "interim"
DEFAULT_CONFIG_DIR = Path("config")
DEFAULT_REPORT_DIR = Path("reports")
OUTPUT_FILENAME = "smartstock_v1_long.csv"
MANIFEST_FILENAME = "v1_subset.json"
REPORT_FILENAME = "stage4_transformation_report.md"
SUMMARY_FILENAME = "stage4_summary.json"

SALES_FILENAME = "sales_train_evaluation.csv"
PRICES_FILENAME = "sell_prices.csv"
CALENDAR_FILENAME = "calendar.csv"
SALES_ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CALENDAR_COLUMNS = [
    "d",
    "date",
    "wm_yr_wk",
    "weekday",
    "wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
]


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 hash for source provenance and immutability checks."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a small DataFrame to JSON-serializable records."""

    records: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in record.items():
            if pd.isna(value):
                clean[str(key)] = None
            elif isinstance(value, (np.integer,)):
                clean[str(key)] = int(value)
            elif isinstance(value, (np.floating,)):
                clean[str(key)] = round(float(value), 10)
            elif isinstance(value, pd.Timestamp):
                clean[str(key)] = value.date().isoformat()
            else:
                clean[str(key)] = value
        records.append(clean)
    return records


def load_stage4_inputs(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load only FOODS rows, frozen stores, calendar, and matching price rows."""

    sales_path = data_dir / SALES_FILENAME
    prices_path = data_dir / PRICES_FILENAME
    calendar_path = data_dir / CALENDAR_FILENAME
    for path in (sales_path, prices_path, calendar_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required raw file is missing: {path}")

    header = pd.read_csv(sales_path, nrows=0).columns.tolist()
    daily_columns = find_daily_sales_columns(header)
    if not daily_columns:
        raise ValueError("No daily sales columns were found.")

    print("Loading FOODS sales for the three frozen stores ...", flush=True)
    sales_parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        sales_path,
        dtype={**{column: "string" for column in SALES_ID_COLUMNS}, **{column: np.int32 for column in daily_columns}},
        chunksize=2_000,
    ):
        selected = chunk[
            chunk["cat_id"].eq(SELECTED_CATEGORY)
            & chunk["store_id"].isin(SELECTED_STORES)
        ]
        if not selected.empty:
            sales_parts.append(selected)
    sales_wide = pd.concat(sales_parts, ignore_index=True)

    print("Loading calendar and relevant weekly prices ...", flush=True)
    calendar = pd.read_csv(
        calendar_path,
        usecols=CALENDAR_COLUMNS,
        parse_dates=["date"],
        dtype={
            "d": "string",
            "wm_yr_wk": np.int32,
            "weekday": "string",
            "wday": np.int8,
            "month": np.int8,
            "year": np.int16,
            "event_name_1": "string",
            "event_type_1": "string",
            "event_name_2": "string",
            "event_type_2": "string",
            "snap_CA": np.int8,
            "snap_TX": np.int8,
            "snap_WI": np.int8,
        },
    )
    prices = pd.read_csv(
        prices_path,
        dtype={
            "store_id": "string",
            "item_id": "string",
            "wm_yr_wk": np.int32,
            "sell_price": np.float32,
        },
    )
    foods_items = set(sales_wide["item_id"].unique())
    prices = prices[
        prices["store_id"].isin(SELECTED_STORES)
        & prices["item_id"].isin(foods_items)
    ].reset_index(drop=True)
    return sales_wide, prices, calendar, daily_columns


def wide_to_long(selected_wide: pd.DataFrame, daily_columns: Sequence[str]) -> pd.DataFrame:
    """Melt only the selected item-store rows to daily long format."""

    long_sales = selected_wide.melt(
        id_vars=SALES_ID_COLUMNS,
        value_vars=list(daily_columns),
        var_name="d",
        value_name="sales",
    )
    long_sales["d"] = long_sales["d"].astype("string")
    long_sales["sales"] = pd.to_numeric(long_sales["sales"], errors="raise", downcast="integer")
    return long_sales


def join_calendar(
    long_sales: pd.DataFrame,
    calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join calendar attributes with a strict many-to-one cardinality check."""

    before_rows = len(long_sales)
    if calendar["d"].duplicated().any():
        raise ValueError("calendar.d is not unique; many-to-one join would be unsafe.")
    joined = long_sales.merge(
        calendar[CALENDAR_COLUMNS],
        on="d",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    unmatched = int(joined["_merge"].ne("both").sum())
    joined = joined.drop(columns="_merge")
    if len(joined) != before_rows:
        raise ValueError("Calendar join unexpectedly changed the row count.")
    return joined, {
        "rows_before": before_rows,
        "rows_after": int(len(joined)),
        "unmatched_rows": unmatched,
        "row_multiplier": len(joined) / before_rows,
    }


def join_prices(
    daily_sales: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join prices using the required item-store-week composite key."""

    key = ["store_id", "item_id", "wm_yr_wk"]
    duplicate_price_keys = int(prices.duplicated(subset=key).sum())
    if duplicate_price_keys:
        raise ValueError(
            f"Price composite key is not unique: {duplicate_price_keys} duplicate rows."
        )
    before_rows = len(daily_sales)
    joined = daily_sales.merge(
        prices[key + ["sell_price"]],
        on=key,
        how="left",
        validate="many_to_one",
    )
    if len(joined) != before_rows:
        raise ValueError("Price join unexpectedly changed the row count.")
    return joined, {
        "join_key": key,
        "rows_before": before_rows,
        "rows_after": int(len(joined)),
        "row_multiplier": len(joined) / before_rows,
        "duplicate_price_keys": duplicate_price_keys,
        "matched_rows": int(joined["sell_price"].notna().sum()),
        "missing_price_rows": int(joined["sell_price"].isna().sum()),
    }


def summarize_missing_prices(final_data: pd.DataFrame) -> dict[str, Any]:
    """Describe missing prices without imputing or deleting them."""

    missing = final_data["sell_price"].isna()
    known_price_dates = (
        final_data.loc[~missing]
        .groupby(["store_id", "item_id"], observed=True)["date"]
        .agg(first_known_price_date="min", last_known_price_date="max")
    )
    analysis = final_data[["store_id", "item_id", "date", "sell_price", "sales"]].join(
        known_price_dates,
        on=["store_id", "item_id"],
    )
    missing_analysis = analysis[analysis["sell_price"].isna()]

    before_first_known = int(
        missing_analysis["date"].lt(missing_analysis["first_known_price_date"]).sum()
    )
    after_first_known = int(
        missing_analysis["date"].ge(missing_analysis["first_known_price_date"]).sum()
    )
    after_last_known = int(
        missing_analysis["date"].gt(missing_analysis["last_known_price_date"]).sum()
    )

    first_positive_dates = (
        final_data.loc[final_data["sales"].gt(0)]
        .groupby(["store_id", "item_id"], observed=True)["date"]
        .min()
        .rename("first_positive_sale_date")
    )
    missing_analysis = missing_analysis.join(
        first_positive_dates,
        on=["store_id", "item_id"],
    )
    before_first_positive = int(
        missing_analysis["date"].lt(missing_analysis["first_positive_sale_date"]).sum()
    )
    on_or_after_first_positive = int(
        missing_analysis["date"].ge(missing_analysis["first_positive_sale_date"]).sum()
    )
    on_or_after_first_positive_with_sales = int(
        (
            missing_analysis["date"].ge(missing_analysis["first_positive_sale_date"])
            & missing_analysis["sales"].gt(0)
        ).sum()
    )

    by_store = (
        final_data.assign(missing_price=missing)
        .groupby("store_id", observed=True)
        .agg(total_rows=("missing_price", "size"), missing_rows=("missing_price", "sum"))
        .reset_index()
    )
    by_store["missing_pct"] = 100 * by_store["missing_rows"] / by_store["total_rows"]

    by_product = (
        final_data.assign(missing_price=missing)
        .groupby("item_id", observed=True)
        .agg(total_rows=("missing_price", "size"), missing_rows=("missing_price", "sum"))
        .reset_index()
    )
    by_product["missing_pct"] = 100 * by_product["missing_rows"] / by_product["total_rows"]
    by_product = by_product.sort_values(["missing_rows", "item_id"], ascending=[False, True])

    total_missing = int(missing.sum())
    return {
        "total_rows": int(len(final_data)),
        "missing_rows": total_missing,
        "missing_pct": round(100 * total_missing / len(final_data), 6),
        "before_first_known_price_rows": before_first_known,
        "on_or_after_first_known_price_rows": after_first_known,
        "after_last_known_price_rows": after_last_known,
        "before_first_positive_sale_rows": before_first_positive,
        "on_or_after_first_positive_sale_rows": on_or_after_first_positive,
        "on_or_after_first_positive_with_positive_sales_rows": on_or_after_first_positive_with_sales,
        "by_store": json_safe_records(by_store),
        "products_with_missing_prices": int(by_product["missing_rows"].gt(0).sum()),
        "products_without_missing_prices": int(by_product["missing_rows"].eq(0).sum()),
        "by_product": json_safe_records(by_product),
        "top_15_products": json_safe_records(by_product.head(15)),
    }


def validate_final_dataset(
    final_data: pd.DataFrame,
    selected_items: pd.DataFrame,
    daily_columns: Sequence[str],
) -> dict[str, Any]:
    """Run all required Stage 4 integrity checks and fail loudly on violations."""

    expected_rows = len(selected_items) * len(SELECTED_STORES) * len(daily_columns)
    observed_stores = sorted(str(value) for value in final_data["store_id"].unique())
    departments = {
        str(key): int(value)
        for key, value in selected_items["department"].value_counts().sort_index().items()
    }
    demand_bands = {
        str(key): int(value)
        for key, value in selected_items["demand_band"].value_counts().sort_index().items()
    }
    pair_day_counts = final_data.groupby(["store_id", "item_id"], observed=True)["d"].nunique()
    duplicate_final_keys = int(
        final_data.duplicated(subset=["date", "store_id", "item_id"]).sum()
    )

    checks = {
        "exactly_100_items": int(final_data["item_id"].nunique()) == 100,
        "exact_frozen_stores": observed_stores == sorted(SELECTED_STORES),
        "foods_only": set(final_data["cat_id"].dropna().unique()) == {SELECTED_CATEGORY},
        "department_quotas": departments == {"FOODS_1": 15, "FOODS_2": 28, "FOODS_3": 57},
        "demand_band_quotas": demand_bands == {"high": 33, "low": 33, "medium": 34},
        "exactly_300_item_store_pairs": int(pair_day_counts.size) == 300,
        "expected_rows": int(len(final_data)) == expected_rows,
        "complete_day_history_per_pair": bool(pair_day_counts.eq(len(daily_columns)).all()),
        "unique_date_store_item_key": duplicate_final_keys == 0,
        "no_negative_sales": int(final_data["sales"].lt(0).sum()) == 0,
        "all_calendar_rows_matched": int(final_data["date"].isna().sum()) == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Final dataset integrity checks failed: " + ", ".join(failed))

    return {
        "checks": checks,
        "expected_rows": expected_rows,
        "observed_rows": int(len(final_data)),
        "duplicate_final_keys": duplicate_final_keys,
        "minimum_days_per_item_store": int(pair_day_counts.min()),
        "maximum_days_per_item_store": int(pair_day_counts.max()),
        "negative_sales_rows": int(final_data["sales"].lt(0).sum()),
        "calendar_unmatched_rows": int(final_data["date"].isna().sum()),
    }


def build_manifest(
    selected_items: pd.DataFrame,
    thresholds: dict[str, float],
    source_hashes: dict[str, str],
    *,
    created_at: str,
) -> dict[str, Any]:
    """Create the tracked, machine-independent Version 1 subset manifest."""

    items = selected_items[
        [
            "item_id",
            "department",
            "demand_band",
            "mean_daily_demand",
            "zero_prevalence",
            "first_positive_day",
            "remaining_history_days",
            "minimum_active_period_price_coverage",
        ]
    ].sort_values("item_id")
    return {
        "manifest_version": 1,
        "project_stage": "Stage 4 - Version 1 Subset Creation & Controlled Data Transformation",
        "created_at_utc": created_at,
        "source_dataset": {
            "name": "Walmart M5 Forecasting - Accuracy",
            "competition_identifier": "m5-forecasting-accuracy",
            "files_sha256": source_hashes,
        },
        "category": SELECTED_CATEGORY,
        "stores": list(SELECTED_STORES),
        "random_seed": RANDOM_SEED,
        "selection_quotas": SELECTION_QUOTAS,
        "demand_metric": "total units across CA_1, TX_2, and WI_3 divided by (3 stores x 1,941 days)",
        "demand_tertile_thresholds": thresholds,
        "demand_band_boundary_policy": "low <= lower threshold; lower < medium <= upper threshold; high > upper threshold",
        "eligibility": {
            "required_stores": list(SELECTED_STORES),
            "maximum_zero_prevalence_exclusive": ZERO_PREVALENCE_THRESHOLD,
            "minimum_remaining_history_days": MINIMUM_REMAINING_HISTORY_DAYS,
            "remaining_history_formula": "1941 - earliest positive-sale day across selected stores",
            "minimum_active_period_price_coverage_per_store": MINIMUM_ACTIVE_PERIOD_PRICE_COVERAGE,
            "active_period_price_coverage_definition": "distinct priced weeks divided by distinct calendar weeks from each item-store's first positive-sale day through d_1941; all three stores must pass",
        },
        "item_count": int(len(items)),
        "items": json_safe_records(items),
    }


def build_summary(
    eligibility: pd.DataFrame,
    banded_items: pd.DataFrame,
    quota_availability: pd.DataFrame,
    selected_items: pd.DataFrame,
    thresholds: dict[str, float],
    final_data: pd.DataFrame,
    calendar_join: dict[str, Any],
    price_join: dict[str, Any],
    missing_prices: dict[str, Any],
    integrity: dict[str, Any],
    output_path: Path,
    raw_hashes_before: dict[str, str],
    raw_hashes_after: dict[str, str],
    *,
    created_at: str,
) -> dict[str, Any]:
    """Build the machine-readable Stage 4 execution summary."""

    selected_department_counts = {
        str(key): int(value)
        for key, value in selected_items["department"].value_counts().sort_index().items()
    }
    selected_band_counts = {
        str(key): int(value)
        for key, value in selected_items["demand_band"].value_counts().sort_index().items()
    }
    eligible_band_counts = {
        str(key): int(value)
        for key, value in banded_items["demand_band"].value_counts().sort_index().items()
    }
    rule_columns = [
        "passes_required_stores",
        "passes_sparsity",
        "passes_history",
        "passes_price_coverage",
    ]
    failed_rule_counts_per_item = (~eligibility[rule_columns]).sum(axis=1)

    return {
        "stage": "Stage 4 - Version 1 Subset Creation & Controlled Data Transformation",
        "created_at_utc": created_at,
        "stores": list(SELECTED_STORES),
        "category": SELECTED_CATEGORY,
        "eligibility": {
            "starting_food_items": int(len(eligibility)),
            "eligible_items": int(eligibility["eligible"].sum()),
            "ineligible_items": int((~eligibility["eligible"]).sum()),
            "independent_failure_counts": independent_failure_counts(eligibility),
            "items_failing_multiple_rules": int(failed_rule_counts_per_item.gt(1).sum()),
            "overlap_warning": "Rule failure counts are independent and may overlap; they must not be summed as sequential removals.",
            "minimum_observed_active_period_price_coverage": round(
                float(eligibility["minimum_active_period_price_coverage"].min()), 6
            ),
        },
        "demand_stratification": {
            "metric": "mean daily units across all three selected stores and all 1,941 days",
            "thresholds": thresholds,
            "eligible_band_counts": eligible_band_counts,
            "quota_availability": json_safe_records(quota_availability),
        },
        "selection": {
            "random_seed": RANDOM_SEED,
            "selected_items": int(selected_items["item_id"].nunique()),
            "department_counts": selected_department_counts,
            "demand_band_counts": selected_band_counts,
            "item_store_pairs": int(final_data[["item_id", "store_id"]].drop_duplicates().shape[0]),
        },
        "transformation": {
            "source_selected_wide_rows": 300,
            "daily_columns": 1_941,
            "final_rows": int(len(final_data)),
            "final_columns": int(final_data.shape[1]),
            "date_min": final_data["date"].min().date().isoformat(),
            "date_max": final_data["date"].max().date().isoformat(),
            "dataframe_memory_mib": round(final_data.memory_usage(deep=True).sum() / 1024**2, 2),
            "csv_file_size_mib": round(output_path.stat().st_size / 1024**2, 2),
            "output_path": output_path.as_posix(),
        },
        "calendar_join": calendar_join,
        "price_join": price_join,
        "missing_prices": missing_prices,
        "integrity": integrity,
        "raw_files": {
            "sha256_before": raw_hashes_before,
            "sha256_after": raw_hashes_after,
            "unchanged": raw_hashes_before == raw_hashes_after,
        },
        "performance": {
            "filter_before_melt": True,
            "full_m5_melt_performed": False,
            "full_m5_rows_avoided": 30_490 * 1_941,
            "selected_long_rows": int(len(final_data)),
            "safe_repeated_text_downcast": "categorical dtype in memory; identifiers remain text values in CSV",
        },
    }


def run_stage4(
    data_dir: Path,
    interim_dir: Path,
    config_dir: Path,
    report_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute Stage 4 and persist its manifest, working dataset, and reports."""

    source_paths = {
        SALES_FILENAME: data_dir / SALES_FILENAME,
        PRICES_FILENAME: data_dir / PRICES_FILENAME,
        CALENDAR_FILENAME: data_dir / CALENDAR_FILENAME,
    }
    raw_hashes_before = {name: file_sha256(path) for name, path in source_paths.items()}
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat()

    sales_wide, prices, calendar, daily_columns = load_stage4_inputs(data_dir)
    print("Calculating eligibility metrics ...", flush=True)
    item_metrics, item_store_metrics = aggregate_item_sales_metrics(
        sales_wide,
        daily_columns,
    )
    price_coverage = compute_active_period_price_coverage(
        item_store_metrics,
        prices,
        calendar,
        last_sales_day=len(daily_columns),
    )
    eligibility = evaluate_eligibility(item_metrics, price_coverage)
    eligible = eligibility[eligibility["eligible"]].copy()
    banded_items, thresholds = assign_demand_bands(eligible)
    quota_availability = validate_quota_availability(banded_items)
    selected_items = select_stratified_items(banded_items)

    if selected_items["item_id"].nunique() != 100:
        raise ValueError("Selection did not produce exactly 100 unique items.")
    selected_item_ids = set(selected_items["item_id"])
    selected_wide = sales_wide[
        sales_wide["item_id"].isin(selected_item_ids)
        & sales_wide["store_id"].isin(SELECTED_STORES)
    ].copy()
    if len(selected_wide) != 300:
        raise ValueError(f"Expected 300 selected wide rows, found {len(selected_wide)}.")

    print("Melting only the 300 selected item-store rows ...", flush=True)
    long_sales = wide_to_long(selected_wide, daily_columns)
    print("Joining calendar and weekly prices with cardinality validation ...", flush=True)
    with_calendar, calendar_join = join_calendar(long_sales, calendar)
    final_data, price_join = join_prices(with_calendar, prices)

    selected_band_map = selected_items.set_index("item_id")["demand_band"]
    final_data["demand_band"] = final_data["item_id"].map(selected_band_map).astype("string")
    repeated_text_columns = [
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
    for column in repeated_text_columns:
        final_data[column] = final_data[column].astype("category")
    final_data = final_data.sort_values(["date", "store_id", "item_id"]).reset_index(drop=True)
    final_data["sales"] = pd.to_numeric(final_data["sales"], downcast="integer")
    final_data["sell_price"] = pd.to_numeric(final_data["sell_price"], downcast="float")

    integrity = validate_final_dataset(final_data, selected_items, daily_columns)
    missing_prices = summarize_missing_prices(final_data)

    interim_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = interim_dir / OUTPUT_FILENAME
    manifest_path = config_dir / MANIFEST_FILENAME
    report_path = report_dir / REPORT_FILENAME
    summary_path = report_dir / SUMMARY_FILENAME

    print(f"Writing {len(final_data):,} long-format rows ...", flush=True)
    final_data.to_csv(output_path, index=False, date_format="%Y-%m-%d")

    manifest = build_manifest(
        selected_items,
        thresholds,
        raw_hashes_before,
        created_at=created_at,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    raw_hashes_after = {name: file_sha256(path) for name, path in source_paths.items()}
    if raw_hashes_before != raw_hashes_after:
        raise RuntimeError("A raw source file changed during Stage 4 execution.")

    summary = build_summary(
        eligibility,
        banded_items,
        quota_availability,
        selected_items,
        thresholds,
        final_data,
        calendar_join,
        price_join,
        missing_prices,
        integrity,
        output_path,
        raw_hashes_before,
        raw_hashes_after,
        created_at=created_at,
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_stage4_report(summary), encoding="utf-8")
    return manifest, summary


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage 4 command-line interface."""

    parser = argparse.ArgumentParser(
        description="Create the controlled SmartStock Version 1 subset and long dataset.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--interim-dir", type=Path, default=DEFAULT_INTERIM_DIR)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run Stage 4 and print a compact completion summary."""

    args = build_parser().parse_args(argv)
    _, summary = run_stage4(
        args.data_dir,
        args.interim_dir,
        args.config_dir,
        args.report_dir,
    )
    transformation = summary["transformation"]
    missing = summary["missing_prices"]
    print()
    print("Stage 4 transformation complete")
    print(f"Items: {summary['selection']['selected_items']:,}")
    print(f"Item-store pairs: {summary['selection']['item_store_pairs']:,}")
    print(f"Rows: {transformation['final_rows']:,}")
    print(f"Columns: {transformation['final_columns']:,}")
    print(f"Missing prices: {missing['missing_rows']:,} ({missing['missing_pct']:.2f}%)")
    print(f"Dataset: {(args.interim_dir / OUTPUT_FILENAME).resolve()}")
    print(f"Manifest: {(args.config_dir / MANIFEST_FILENAME).resolve()}")
    print(f"Report: {(args.report_dir / REPORT_FILENAME).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
