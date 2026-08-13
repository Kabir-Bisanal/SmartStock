"""Select the reproducible SmartStock Version 1 FOODS product subset.

Selection is data-driven and deterministic. This module does not reshape sales
or write output files; transformation and persistence are handled separately.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SELECTED_STORES = ("CA_1", "TX_2", "WI_3")
SELECTED_CATEGORY = "FOODS"
RANDOM_SEED = 42
ZERO_PREVALENCE_THRESHOLD = 0.95
MINIMUM_REMAINING_HISTORY_DAYS = 730
MINIMUM_ACTIVE_PERIOD_PRICE_COVERAGE = 0.70

SELECTION_QUOTAS: dict[str, dict[str, int]] = {
    "FOODS_1": {"low": 5, "medium": 5, "high": 5},
    "FOODS_2": {"low": 9, "medium": 10, "high": 9},
    "FOODS_3": {"low": 19, "medium": 19, "high": 19},
}


def aggregate_item_sales_metrics(
    selected_sales_wide: pd.DataFrame,
    daily_columns: Sequence[str],
    *,
    selected_stores: Sequence[str] = SELECTED_STORES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate item and item-store demand metrics from the selected sales rows."""

    expected_stores = set(selected_stores)
    selected_sales_wide = selected_sales_wide.copy()
    selected_sales_wide["item_id"] = selected_sales_wide["item_id"].astype("string")
    selected_sales_wide["store_id"] = selected_sales_wide["store_id"].astype("string")
    values = selected_sales_wide[list(daily_columns)].to_numpy(dtype=np.int32, copy=False)
    positive = values > 0
    has_positive = positive.any(axis=1)
    first_positive = np.where(has_positive, positive.argmax(axis=1) + 1, -1)

    item_store = selected_sales_wide[
        ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    ].reset_index(drop=True)
    item_store["first_positive_day"] = first_positive.astype(np.int32)
    item_store["zero_observations"] = (values == 0).sum(axis=1, dtype=np.int32)
    item_store["total_units"] = values.sum(axis=1, dtype=np.int64)
    item_store["mean_daily_demand"] = values.mean(axis=1)

    item_metrics = item_store.groupby("item_id", observed=True).agg(
        department=("dept_id", "first"),
        category=("cat_id", "first"),
        store_count=("store_id", "nunique"),
        first_positive_day=(
            "first_positive_day",
            lambda series: int(series[series >= 0].min()) if (series >= 0).any() else -1,
        ),
        zero_observations=("zero_observations", "sum"),
        total_units=("total_units", "sum"),
    )
    item_metrics["structurally_in_required_stores"] = item_metrics["store_count"].eq(
        len(expected_stores)
    )
    item_metrics["zero_prevalence"] = item_metrics["zero_observations"] / (
        item_metrics["store_count"] * len(daily_columns)
    )
    item_metrics["mean_daily_demand"] = item_metrics["total_units"] / (
        item_metrics["store_count"] * len(daily_columns)
    )
    item_metrics["remaining_history_days"] = np.where(
        item_metrics["first_positive_day"].ge(1),
        len(daily_columns) - item_metrics["first_positive_day"],
        -1,
    )
    return item_metrics.reset_index(), item_store


def compute_active_period_price_coverage(
    item_store_metrics: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    last_sales_day: int,
) -> pd.DataFrame:
    """Measure weekly price coverage from first positive sale through the last day.

    Coverage is calculated independently for each item-store pair. The relevant
    denominator is the number of distinct calendar weeks from that pair's first
    positive sale day through ``d_<last_sales_day>``. Weeks before launch are not
    counted. A week is covered when at least one price row exists for the exact
    ``(store_id, item_id, wm_yr_wk)`` key.
    """

    calendar_weeks = calendar[["d", "wm_yr_wk"]].copy()
    calendar_weeks["day_number"] = calendar_weeks["d"].str.removeprefix("d_").astype(int)
    calendar_weeks = calendar_weeks[calendar_weeks["day_number"].le(last_sales_day)]

    week_sets_by_start_day: dict[int, set[int]] = {}
    for first_day in sorted(item_store_metrics["first_positive_day"].unique()):
        first_day = int(first_day)
        if first_day < 1:
            week_sets_by_start_day[first_day] = set()
            continue
        week_sets_by_start_day[first_day] = set(
            int(value)
            for value in calendar_weeks.loc[
                calendar_weeks["day_number"].ge(first_day), "wm_yr_wk"
            ].unique()
        )

    price_week_sets = (
        prices.groupby(["store_id", "item_id"], observed=True)["wm_yr_wk"]
        .agg(lambda values: set(int(value) for value in values))
        .to_dict()
    )

    coverage_records: list[dict[str, Any]] = []
    for row in item_store_metrics.itertuples(index=False):
        relevant_weeks = week_sets_by_start_day[int(row.first_positive_day)]
        known_price_weeks = price_week_sets.get((str(row.store_id), str(row.item_id)), set())
        covered_weeks = len(relevant_weeks & known_price_weeks)
        relevant_week_count = len(relevant_weeks)
        coverage = covered_weeks / relevant_week_count if relevant_week_count else 0.0
        coverage_records.append(
            {
                "item_id": str(row.item_id),
                "store_id": str(row.store_id),
                "first_positive_day": int(row.first_positive_day),
                "active_period_weeks": relevant_week_count,
                "priced_active_period_weeks": covered_weeks,
                "active_period_price_coverage": coverage,
            }
        )
    return pd.DataFrame.from_records(coverage_records)


def evaluate_eligibility(
    item_metrics: pd.DataFrame,
    active_price_coverage: pd.DataFrame,
    *,
    zero_prevalence_threshold: float = ZERO_PREVALENCE_THRESHOLD,
    minimum_remaining_history_days: int = MINIMUM_REMAINING_HISTORY_DAYS,
    minimum_price_coverage: float = MINIMUM_ACTIVE_PERIOD_PRICE_COVERAGE,
    required_store_count: int = len(SELECTED_STORES),
) -> pd.DataFrame:
    """Apply the four Version 1 product eligibility rules independently."""

    price_summary = active_price_coverage.groupby("item_id", observed=True).agg(
        stores_with_price_coverage=("store_id", "nunique"),
        minimum_active_period_price_coverage=("active_period_price_coverage", "min"),
    )
    eligibility = item_metrics.merge(price_summary, on="item_id", how="left")
    eligibility["stores_with_price_coverage"] = (
        eligibility["stores_with_price_coverage"].fillna(0).astype(int)
    )
    eligibility["minimum_active_period_price_coverage"] = eligibility[
        "minimum_active_period_price_coverage"
    ].fillna(0.0)

    eligibility["passes_required_stores"] = eligibility[
        "structurally_in_required_stores"
    ] & eligibility["store_count"].eq(required_store_count)
    eligibility["passes_sparsity"] = eligibility["zero_prevalence"].lt(
        zero_prevalence_threshold
    )
    eligibility["passes_history"] = eligibility["remaining_history_days"].ge(
        minimum_remaining_history_days
    )
    eligibility["passes_price_coverage"] = eligibility[
        "stores_with_price_coverage"
    ].eq(required_store_count) & eligibility["minimum_active_period_price_coverage"].ge(
        minimum_price_coverage
    )
    rule_columns = [
        "passes_required_stores",
        "passes_sparsity",
        "passes_history",
        "passes_price_coverage",
    ]
    eligibility["eligible"] = eligibility[rule_columns].all(axis=1)
    return eligibility


def assign_demand_bands(
    eligible_items: pd.DataFrame,
    *,
    metric_column: str = "mean_daily_demand",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Assign low/medium/high bands using reproducible eligible-population tertiles."""

    if eligible_items.empty:
        raise ValueError("Cannot assign demand bands to an empty eligible population.")

    lower = float(eligible_items[metric_column].quantile(1 / 3, interpolation="linear"))
    upper = float(eligible_items[metric_column].quantile(2 / 3, interpolation="linear"))
    if lower >= upper:
        raise ValueError("Demand tertile thresholds are not distinct.")

    assigned = eligible_items.copy()
    assigned["demand_band"] = pd.cut(
        assigned[metric_column],
        bins=[-np.inf, lower, upper, np.inf],
        labels=["low", "medium", "high"],
        include_lowest=True,
        right=True,
    ).astype("string")
    thresholds = {
        "low_upper_inclusive": lower,
        "medium_upper_inclusive": upper,
    }
    return assigned, thresholds


def validate_quota_availability(
    eligible_items: pd.DataFrame,
    quotas: Mapping[str, Mapping[str, int]] = SELECTION_QUOTAS,
) -> pd.DataFrame:
    """Return quota availability and raise before selection if a cell is too small."""

    availability_records: list[dict[str, Any]] = []
    shortages: list[str] = []
    for department, band_quotas in quotas.items():
        for demand_band, required in band_quotas.items():
            available = int(
                (
                    eligible_items["department"].eq(department)
                    & eligible_items["demand_band"].eq(demand_band)
                ).sum()
            )
            availability_records.append(
                {
                    "department": department,
                    "demand_band": demand_band,
                    "available": available,
                    "required": int(required),
                }
            )
            if available < required:
                shortages.append(
                    f"{department}/{demand_band}: available {available}, required {required}"
                )

    availability = pd.DataFrame.from_records(availability_records)
    if shortages:
        raise ValueError("Selection quotas cannot be satisfied: " + "; ".join(shortages))
    return availability


def select_stratified_items(
    eligible_items: pd.DataFrame,
    quotas: Mapping[str, Mapping[str, int]] = SELECTION_QUOTAS,
    *,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Select exact department × demand-band quotas using a fixed random seed."""

    validate_quota_availability(eligible_items, quotas)
    selected_parts: list[pd.DataFrame] = []
    rng = np.random.default_rng(random_seed)

    for department, band_quotas in quotas.items():
        for demand_band, required in band_quotas.items():
            pool = eligible_items[
                eligible_items["department"].eq(department)
                & eligible_items["demand_band"].eq(demand_band)
            ].sort_values("item_id")
            chosen_positions = rng.choice(len(pool), size=required, replace=False)
            selected_parts.append(pool.iloc[np.sort(chosen_positions)])

    selected = pd.concat(selected_parts, ignore_index=True)
    return selected.sort_values(["department", "demand_band", "item_id"]).reset_index(
        drop=True
    )


def independent_failure_counts(eligibility: pd.DataFrame) -> dict[str, int]:
    """Count independent rule failures; overlaps are deliberately retained."""

    return {
        "required_stores": int((~eligibility["passes_required_stores"]).sum()),
        "extreme_sparsity": int((~eligibility["passes_sparsity"]).sum()),
        "insufficient_remaining_history": int((~eligibility["passes_history"]).sum()),
        "insufficient_active_period_price_coverage": int(
            (~eligibility["passes_price_coverage"]).sum()
        ),
    }
