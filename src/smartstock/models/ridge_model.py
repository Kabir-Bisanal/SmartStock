"""Frozen global Ridge model and fold-safe preprocessing for SmartStock Stage 8."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROHIBITED_FEATURES = {
    "sales",
    "demand_band",
    "known_future_sell_price",
    "sell_price",
    "previous_week_price",
    "price_change_previous_week",
    "price_pct_change_previous_week",
    "price_vs_4week_median_lagged",
    "zero_rate_7",
    "zero_rate_28",
    "days_since_last_positive_sale",
    "availability_date",
    "is_active",
    "product_age_weeks",
    "state_id",
    "cat_id",
}


def validate_ridge_config(config: dict[str, Any], feature_manifest: dict[str, Any]) -> dict[str, bool]:
    """Confirm that the tracked experiment uses only approved Stage 6 fields."""

    categorical = list(config["categorical_features"])
    numeric = list(config["numeric_features"])
    features = categorical + numeric
    manifest_names = {entry["name"] for entry in feature_manifest["features"]}
    checks = {
        "features_exist_in_stage6_manifest": set(features).issubset(manifest_names),
        "feature_count_matches": len(features) == config["feature_count_before_encoding"],
        "features_are_unique": len(features) == len(set(features)),
        "no_prohibited_feature": not bool(set(features) & PROHIBITED_FEATURES),
        "target_is_sales": config["target"] == "sales",
        "fixed_alpha_one": float(config["model"]["alpha"]) == 1.0,
        "solver_is_lsqr": config["model"]["solver"] == "lsqr",
        "recursive_strategy": config["forecast_policy"]["strategy"] == "recursive",
        "price_features_excluded": not any("price" in feature for feature in features),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Invalid Stage 8 Ridge configuration: {', '.join(failed)}")
    return checks


def prepare_feature_frame(
    frame: pd.DataFrame,
    categorical_features: list[str],
    numeric_features: list[str],
) -> pd.DataFrame:
    """Return only model fields with stable sklearn-friendly dtypes."""

    missing = set(categorical_features + numeric_features) - set(frame.columns)
    if missing:
        raise ValueError(f"Model feature frame is missing columns: {sorted(missing)}")
    result = frame[categorical_features + numeric_features].copy()
    for column in categorical_features:
        result[column] = result[column].astype("object").where(result[column].notna(), np.nan)
    for column in numeric_features:
        result[column] = pd.to_numeric(result[column], errors="raise").astype("float32")
    return result


def build_ridge_pipeline(config: dict[str, Any]) -> Pipeline:
    """Create the fixed preprocessing and Ridge pipeline declared before evaluation."""

    categorical = list(config["categorical_features"])
    numeric = list(config["numeric_features"])
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__NONE__", keep_empty_features=True),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical),
            ("numeric", StandardScaler(), numeric),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    model = Ridge(
        alpha=float(config["model"]["alpha"]),
        solver=config["model"]["solver"],
        fit_intercept=bool(config["model"].get("fit_intercept", True)),
    )
    return Pipeline(steps=[("preprocessing", preprocessing), ("ridge", model)])


def select_training_rows(data: pd.DataFrame, validation_start: pd.Timestamp) -> pd.DataFrame:
    """Select feature-ready supervised rows strictly before validation begins."""

    training = data[data["is_feature_ready"].eq(True) & data["date"].lt(validation_start)].copy()
    if training.empty:
        raise ValueError("No feature-ready Ridge training rows precede validation.")
    if training["date"].ge(validation_start).any() or not training["is_feature_ready"].all():
        raise ValueError("Ridge training-row cutoff or readiness policy was violated.")
    return training


def fit_global_ridge(
    training: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit one global Ridge pipeline and return preprocessing audit evidence."""

    categorical = list(config["categorical_features"])
    numeric = list(config["numeric_features"])
    features = prepare_feature_frame(training, categorical, numeric)
    target = training[config["target"]].to_numpy(dtype="float64")
    pipeline = build_ridge_pipeline(config)
    pipeline.fit(features, target)

    preprocessing = pipeline.named_steps["preprocessing"]
    scaler = preprocessing.named_transformers_["numeric"]
    encoder = preprocessing.named_transformers_["categorical"].named_steps["encoder"]
    sample_count = scaler.n_samples_seen_
    if isinstance(sample_count, np.ndarray):
        sample_count_matches = bool(np.all(sample_count == len(training)))
    else:
        sample_count_matches = int(sample_count) == len(training)
    category_checks = {
        column: set(map(str, learned)).issubset(
            set(training[column].dropna().astype(str).unique()) | {"__NONE__"}
        )
        for column, learned in zip(categorical, encoder.categories_, strict=True)
    }
    audit = {
        "training_rows": len(training),
        "training_date_min": training["date"].min().date().isoformat(),
        "training_date_max": training["date"].max().date().isoformat(),
        "training_series": int(training.groupby(["store_id", "item_id"], observed=True).ngroups),
        "feature_ready_only": bool(training["is_feature_ready"].all()),
        "scaler_fit_row_count_matches_training": sample_count_matches,
        "encoder_categories_come_from_training_only": bool(all(category_checks.values())),
        "encoded_feature_count": int(len(preprocessing.get_feature_names_out())),
        "target_excluded_from_preprocessing": config["target"] not in categorical + numeric,
        "validation_statistics_used_for_fit": False,
        "final_test_statistics_used_for_fit": False,
    }
    if not all(
        audit[name]
        for name in (
            "feature_ready_only",
            "scaler_fit_row_count_matches_training",
            "encoder_categories_come_from_training_only",
            "target_excluded_from_preprocessing",
        )
    ):
        raise RuntimeError("Fold-specific Ridge preprocessing audit failed.")
    return pipeline, audit
