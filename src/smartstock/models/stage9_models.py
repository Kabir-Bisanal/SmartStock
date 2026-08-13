"""Training-only preprocessing and nonlinear global models for SmartStock Stage 9."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

try:
    from smartstock.models.ridge_model import PROHIBITED_FEATURES, prepare_feature_frame
except ModuleNotFoundError:  # Supports direct execution from the models directory.
    from ridge_model import PROHIBITED_FEATURES, prepare_feature_frame


SUPPORTED_MODEL_FAMILIES = {"hist_gradient_boosting", "xgboost"}


def validate_candidate_config(
    config: dict[str, Any], feature_manifest: dict[str, Any]
) -> dict[str, bool]:
    """Validate the controlled Stage 9 feature and candidate-model policy."""

    categorical = list(config["categorical_features"])
    numeric = list(config["numeric_features"])
    features = categorical + numeric
    manifest_names = {entry["name"] for entry in feature_manifest["features"]}
    candidates = set(config["candidate_models"])
    checks = {
        "features_exist_in_stage6_manifest": set(features).issubset(manifest_names),
        "feature_count_matches": len(features) == int(config["feature_count_before_encoding"]),
        "features_are_unique": len(features) == len(set(features)),
        "exact_candidate_families": candidates == SUPPORTED_MODEL_FAMILIES,
        "target_is_sales": config["target"] == "sales",
        "no_prohibited_feature": not bool(set(features) & PROHIBITED_FEATURES),
        "price_features_excluded": not any("price" in feature for feature in features),
        "demand_band_excluded": "demand_band" not in features,
        "optional_intermittency_features_excluded": not bool(
            {"zero_rate_7", "zero_rate_28", "days_since_last_positive_sale"} & set(features)
        ),
        "recursive_strategy": config["forecast_policy"]["strategy"] == "recursive",
        "thirty_day_horizon": int(config["forecast_policy"]["horizon_days"]) == 30,
        "fixed_random_seed": int(config["random_seed"]) == 42,
        "tuning_cap_respected": all(
            len(grid) <= int(config["selection_policy"]["maximum_tuning_configurations"])
            for grid in config["limited_tuning_grids"].values()
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Invalid Stage 9 candidate configuration: {', '.join(failed)}")
    return checks


def candidate_configuration(
    base_config: dict[str, Any],
    model_family: str,
    parameters: dict[str, Any],
    configuration_id: str,
) -> dict[str, Any]:
    """Return a per-experiment config compatible with the recursive engine."""

    if model_family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(f"Unsupported Stage 9 model family: {model_family}")
    result = deepcopy(base_config)
    result["model_family"] = model_family
    result["model_parameters"] = deepcopy(parameters)
    result["configuration_id"] = configuration_id
    result["experiment_name"] = configuration_id
    return result


def _categorical_pipeline(model_family: str) -> Pipeline:
    imputer = SimpleImputer(
        strategy="constant", fill_value="__NONE__", keep_empty_features=True
    )
    if model_family == "hist_gradient_boosting":
        encoder: Any = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=np.float32,
        )
    else:
        encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=True, dtype=np.float32
        )
    return Pipeline(steps=[("imputer", imputer), ("encoder", encoder)])


def build_tree_pipeline(
    config: dict[str, Any], model_family: str, parameters: dict[str, Any]
) -> Pipeline:
    """Build an HGB or XGBoost pipeline without numeric standardization."""

    if model_family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(f"Unsupported Stage 9 model family: {model_family}")
    categorical = list(config["categorical_features"])
    numeric = list(config["numeric_features"])
    preprocessing = ColumnTransformer(
        transformers=[
            ("categorical", _categorical_pipeline(model_family), categorical),
            (
                "numeric",
                SimpleImputer(strategy="median", keep_empty_features=True),
                numeric,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0 if model_family == "hist_gradient_boosting" else 1.0,
    )
    if model_family == "hist_gradient_boosting":
        model: Any = HistGradientBoostingRegressor(
            **parameters,
            categorical_features=list(range(len(categorical))),
        )
    else:
        try:
            from xgboost import XGBRegressor
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific message.
            raise ModuleNotFoundError(
                "Stage 9 requires xgboost. Install the pinned project requirements first."
            ) from exc
        model = XGBRegressor(**parameters)
    return Pipeline(steps=[("preprocessing", preprocessing), ("model", model)])


def select_training_rows_through(
    data: pd.DataFrame, training_cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Select feature-ready rows on or before a frozen training cutoff."""

    cutoff = pd.Timestamp(training_cutoff)
    training = data[
        data["is_feature_ready"].eq(True) & data["date"].le(cutoff)
    ].copy()
    if training.empty:
        raise ValueError("No feature-ready Stage 9 training rows meet the cutoff.")
    if training["date"].gt(cutoff).any() or not training["is_feature_ready"].all():
        raise ValueError("Stage 9 training cutoff or readiness policy was violated.")
    return training


def fit_global_tree(
    training: pd.DataFrame,
    config: dict[str, Any],
    model_family: str,
    parameters: dict[str, Any],
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit one global nonlinear pipeline and return leakage-audit evidence."""

    categorical = list(config["categorical_features"])
    numeric = list(config["numeric_features"])
    features = prepare_feature_frame(training, categorical, numeric)
    target = training[config["target"]].to_numpy(dtype="float64")
    pipeline = build_tree_pipeline(config, model_family, parameters)
    pipeline.fit(features, target)

    preprocessing = pipeline.named_steps["preprocessing"]
    categorical_transformer = preprocessing.named_transformers_["categorical"]
    encoder = categorical_transformer.named_steps["encoder"]
    learned_categories = encoder.categories_
    category_checks = {
        column: set(map(str, learned)).issubset(
            set(training[column].dropna().astype(str).unique()) | {"__NONE__"}
        )
        for column, learned in zip(categorical, learned_categories, strict=True)
    }
    numeric_imputer = preprocessing.named_transformers_["numeric"]
    audit = {
        "model_family": model_family,
        "training_rows": len(training),
        "training_date_min": training["date"].min().date().isoformat(),
        "training_date_max": training["date"].max().date().isoformat(),
        "training_series": int(
            training.groupby(["store_id", "item_id"], observed=True).ngroups
        ),
        "feature_ready_only": bool(training["is_feature_ready"].all()),
        "encoder_categories_come_from_training_only": bool(all(category_checks.values())),
        "numeric_imputer_fitted_from_training": len(numeric_imputer.statistics_) == len(numeric),
        "numeric_standardization_used": False,
        "target_excluded_from_preprocessing": config["target"] not in categorical + numeric,
        "validation_statistics_used_for_fit": False,
        "final_test_statistics_used_for_fit": False,
    }
    required = (
        "feature_ready_only",
        "encoder_categories_come_from_training_only",
        "numeric_imputer_fitted_from_training",
        "target_excluded_from_preprocessing",
    )
    if not all(audit[name] for name in required):
        raise RuntimeError("Stage 9 training-only preprocessing audit failed.")
    return pipeline, audit
