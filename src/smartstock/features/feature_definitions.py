"""Declarative Stage 6 feature and chronological-validation specifications."""

from __future__ import annotations

from typing import Any


DEMAND_HISTORY_FEATURES = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_roll_mean_7",
    "sales_roll_mean_14",
    "sales_roll_mean_28",
    "sales_roll_std_7",
    "sales_roll_std_28",
    "zero_rate_7",
    "zero_rate_28",
    "days_since_last_positive_sale",
]

PRICE_HISTORY_FEATURES = [
    "previous_week_price",
    "price_change_previous_week",
    "price_pct_change_previous_week",
    "price_vs_4week_median_lagged",
]

REQUIRED_HISTORY_FEATURES = DEMAND_HISTORY_FEATURES + PRICE_HISTORY_FEATURES


def feature_specifications() -> list[dict[str, Any]]:
    """Return one auditable record for every persisted Stage 6 column."""

    features: list[dict[str, Any]] = []

    def add(
        name: str,
        category: str,
        description: str,
        dtype: str,
        source_timing: str,
        *,
        uses_target_history: bool = False,
        known_at_forecast_time: bool = True,
        default_model_feature: bool = False,
        leakage_notes: str = "",
    ) -> None:
        features.append(
            {
                "name": name,
                "category": category,
                "description": description,
                "dtype": dtype,
                "source_timing": source_timing,
                "uses_target_history": uses_target_history,
                "known_at_forecast_time": known_at_forecast_time,
                "default_model_feature": default_model_feature,
                "leakage_notes": leakage_notes,
            }
        )

    add("date", "calendar", "Target calendar date.", "date", "same_day_known_calendar")
    add("d", "calendar", "Original M5 daily key.", "string", "same_day_known_calendar")
    add("wm_yr_wk", "calendar", "Original M5 week key used for weekly prices.", "int32", "same_day_known_calendar")
    add("item_id", "identity", "Product identifier shared across stores.", "category", "static_identity", default_model_feature=True)
    add("dept_id", "identity", "FOODS department identifier.", "category", "static_identity", default_model_feature=True)
    add("cat_id", "identity", "Category identifier; constant FOODS in V1.", "category", "static_identity", leakage_notes="Constant in Version 1, so excluded from default features.")
    add("store_id", "identity", "Store identifier.", "category", "static_identity", default_model_feature=True)
    add("state_id", "identity", "State identifier.", "category", "static_identity", default_model_feature=True)
    add("availability_date", "product_age", "First known selling-price date for the item-store pair.", "date", "availability_proxy", leakage_notes="M5 price availability proxy, not explicit inventory availability.")
    add("is_active", "product_age", "True for every retained row on or after availability.", "bool", "availability_proxy", leakage_notes="Constant true after pre-launch filtering; retained for auditability.")
    add("product_age_days", "product_age", "Days since first known price, starting at zero.", "int16", "availability_proxy", default_model_feature=True)
    add("product_age_weeks", "product_age", "Completed weeks since first known price.", "int16", "availability_proxy", default_model_feature=True)

    calendar_defaults = {
        "day_of_week": ("Monday=0 through Sunday=6.", "int8"),
        "day_of_month": ("Calendar day within month.", "int8"),
        "month": ("Calendar month 1-12.", "int8"),
        "quarter": ("Calendar quarter 1-4.", "int8"),
        "year": ("Calendar year.", "int16"),
        "week_of_year": ("ISO calendar week 1-53.", "int8"),
        "is_weekend": ("True on Saturday or Sunday.", "bool"),
        "day_of_week_sin": ("Sine encoding of the seven-day cycle.", "float32"),
        "day_of_week_cos": ("Cosine encoding of the seven-day cycle.", "float32"),
        "month_sin": ("Sine encoding of the 12-month cycle.", "float32"),
        "month_cos": ("Cosine encoding of the 12-month cycle.", "float32"),
    }
    for name, (description, dtype) in calendar_defaults.items():
        add(name, "calendar", description, dtype, "same_day_known_calendar", default_model_feature=True)

    add("is_event", "event", "True when either M5 event slot is populated.", "bool", "same_day_known_calendar", default_model_feature=True)
    for name in ("event_name_1", "event_type_1", "event_name_2", "event_type_2"):
        add(name, "event", f"Raw M5 calendar field {name}.", "category", "same_day_known_calendar", default_model_feature=True, leakage_notes="Missing means no event in that slot; no target encoding is applied.")
    add("snap_active", "SNAP", "Store-relevant state SNAP calendar flag.", "int8", "same_day_known_calendar", default_model_feature=True)

    lag_windows = {"sales_lag_1": 1, "sales_lag_7": 7, "sales_lag_14": 14, "sales_lag_28": 28}
    for name, lag in lag_windows.items():
        add(name, "sales_history", f"Sales for the same item-store pair {lag} day(s) before the target date.", "float32", "past_target_history", uses_target_history=True, default_model_feature=True, leakage_notes="Grouped by store_id and item_id before shifting.")
    for window in (7, 14, 28):
        add(f"sales_roll_mean_{window}", "sales_history", f"Mean of sales from t-{window} through t-1.", "float32", "past_target_history", uses_target_history=True, default_model_feature=True, leakage_notes="Calculated with shift(1) before rolling.")
    for window in (7, 28):
        add(f"sales_roll_std_{window}", "sales_history", f"Sample standard deviation of sales from t-{window} through t-1.", "float32", "past_target_history", uses_target_history=True, default_model_feature=True, leakage_notes="Calculated with shift(1) before rolling.")
    for window in (7, 28):
        add(f"zero_rate_{window}", "sales_history", f"Share of zero-sales observations from t-{window} through t-1.", "float32", "past_target_history", uses_target_history=True, default_model_feature=True, leakage_notes="Missing history stays missing; zero demand is not treated as missing.")
    add("days_since_last_positive_sale", "sales_history", "Calendar days since the latest positive sale strictly before the target date.", "float32", "past_target_history", uses_target_history=True, default_model_feature=True, leakage_notes="The target day's sale is never inspected.")

    add("previous_week_price", "price_history", "Selling price in the immediately preceding item-store M5 week.", "float32", "past_price_history", default_model_feature=True)
    add("price_change_previous_week", "price_history", "Previous-week price minus the price two weeks before the target week.", "float32", "past_price_history", default_model_feature=True)
    add("price_pct_change_previous_week", "price_history", "100 times previous-week price divided by two-weeks-ago price minus one.", "float32", "past_price_history", default_model_feature=True, leakage_notes="No clipping or capping is applied.")
    add("price_vs_4week_median_lagged", "price_history", "Previous-week price divided by the median of the four weeks before the target week, minus one.", "float32", "past_price_history", default_model_feature=True, leakage_notes="Weekly-aware and based only on completed prior weeks.")
    add("known_future_sell_price", "optional_known_future", "Actual price recorded for the target date/week.", "float32", "optional_known_future", known_at_forecast_time=False, leakage_notes="Scenario-only field; excluded from the default forecast-safe feature set.")

    add("demand_band", "analysis_only", "Stage 4 full-history low/medium/high stratum.", "category", "analysis_only", uses_target_history=True, known_at_forecast_time=False, leakage_notes="Full-history target aggregate; prohibited as a model predictor.")
    add("is_feature_ready", "control", "True when every required past-demand and past-price feature is available.", "bool", "control", leakage_notes="Controls row eligibility and is not a predictor.")
    add("sales", "target", "Observed unit sales on the target date.", "uint16", "target", known_at_forecast_time=False, leakage_notes="Untransformed regression target; never a feature for the same row.")
    return features


def build_feature_manifest() -> dict[str, Any]:
    """Build the deterministic, tracked feature manifest."""

    features = feature_specifications()
    return {
        "manifest_version": 1,
        "project_stage": "Stage 6 - Leakage-Safe Feature Engineering & Chronological Validation Design",
        "dataset": "data/processed/smartstock_v1_features.csv",
        "target": "sales",
        "feature_ready_flag": "is_feature_ready",
        "default_model_features": [item["name"] for item in features if item["default_model_feature"]],
        "required_history_features": REQUIRED_HISTORY_FEATURES,
        "features": features,
    }


def build_validation_manifest() -> dict[str, Any]:
    """Return the frozen expanding-window validation design."""

    return {
        "manifest_version": 1,
        "project_stage": "Stage 6 - Leakage-Safe Feature Engineering & Chronological Validation Design",
        "dataset_version": "SmartStock Version 1",
        "target_column": "sales",
        "availability_policy": {
            "primary_proxy": "first known selling-price date per store_id + item_id",
            "fallback": "first positive-sale date only when no price is ever known",
            "prelaunch_targets_excluded": True,
            "active_zero_sales_retained": True,
        },
        "feature_warmup_policy": {
            "maximum_required_demand_history_days": 28,
            "required_completed_price_weeks": 4,
            "rule": "is_feature_ready is true only when all required past-demand and past-price features are non-missing",
            "warmup_rows_are_persisted": True,
        },
        "training_policy": "expanding window; all feature-ready rows strictly before each evaluation window start",
        "random_split_allowed": False,
        "validation_folds": [
            {"name": "validation_fold_1", "start_date": "2016-01-24", "end_date": "2016-02-22"},
            {"name": "validation_fold_2", "start_date": "2016-02-23", "end_date": "2016-03-23"},
            {"name": "validation_fold_3", "start_date": "2016-03-24", "end_date": "2016-04-22"},
        ],
        "locked_final_test": {
            "start_date": "2016-04-23",
            "end_date": "2016-05-22",
            "days": 30,
            "locked": True,
            "policy": "Do not use final-test target performance for feature, model, or hyperparameter selection.",
            "historical_limitation": "Stage 5 descriptive EDA inspected the complete history before this lock; from Stage 6 onward model-selection decisions must not inspect final-test target performance.",
        },
        "forecast_horizons_days": [1, 7, 30],
        "horizon_evaluation": {
            "1_day": "evaluate daily Day+1 forecasts",
            "7_day": "evaluate daily Day+1 through Day+7 errors and aggregate seven-day demand",
            "30_day": "evaluate daily Day+1 through Day+30 errors and aggregate 30-day demand",
        },
        "split_label_policy": "Dates are persisted in this manifest instead of assigning one permanent dataset_role column because training eligibility changes by expanding fold.",
    }


def audit_feature_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Reject default predictors with a prohibited information source."""

    allowed_sources = {"static_identity", "same_day_known_calendar", "availability_proxy", "past_target_history", "past_price_history"}
    default_features = [feature for feature in manifest["features"] if feature["default_model_feature"]]
    prohibited = [feature["name"] for feature in default_features if feature["source_timing"] not in allowed_sources]
    demand_band = next(feature for feature in manifest["features"] if feature["name"] == "demand_band")
    target = next(feature for feature in manifest["features"] if feature["name"] == "sales")
    known_future_price = next(feature for feature in manifest["features"] if feature["name"] == "known_future_sell_price")
    checks = {
        "default_sources_allowed": not prohibited,
        "demand_band_is_analysis_only": demand_band["category"] == "analysis_only" and not demand_band["default_model_feature"],
        "target_is_not_feature": not target["default_model_feature"],
        "current_price_is_scenario_only": known_future_price["category"] == "optional_known_future" and not known_future_price["default_model_feature"],
        "no_duplicate_feature_names": len({feature["name"] for feature in manifest["features"]}) == len(manifest["features"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "allowed_default_source_timings": sorted(allowed_sources),
        "default_feature_count": len(default_features),
        "prohibited_default_features": prohibited,
    }
