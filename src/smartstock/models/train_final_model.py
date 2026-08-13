"""Reproducibly train and serialize the frozen SmartStock Version 1 forecaster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

try:
    from smartstock.models.baseline_evaluation import file_sha256
    from smartstock.models.production_forecaster import save_forecaster
    from smartstock.models.ridge_model import fit_global_ridge
    from smartstock.models.stage9_models import fit_global_tree, select_training_rows_through
except ModuleNotFoundError:  # Supports direct execution from the models directory.
    from baseline_evaluation import file_sha256
    from production_forecaster import save_forecaster
    from ridge_model import fit_global_ridge
    from stage9_models import fit_global_tree, select_training_rows_through


DEFAULT_FINAL_CONFIG = Path("config/v1_final_model.json")
DEFAULT_FEATURE_DATA = Path("data/processed/smartstock_v1_features.csv")
DEFAULT_MODEL_ARTIFACT = Path("models/smartstock_v1_forecaster.joblib")


class Mean28FeatureForecaster(BaseEstimator, RegressorMixin):
    """Serializable fallback that returns the recursively rebuilt 28-day mean."""

    def fit(self, features: pd.DataFrame, target: np.ndarray | None = None) -> "Mean28FeatureForecaster":
        if "sales_roll_mean_28" not in features.columns:
            raise ValueError("Mean-28 production fallback requires sales_roll_mean_28.")
        self.n_features_in_ = features.shape[1]
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return pd.to_numeric(features["sales_roll_mean_28"], errors="raise").to_numpy(
            dtype="float64"
        )


def train_final_forecaster(
    data: pd.DataFrame,
    config: dict[str, Any],
    *,
    feature_data_path: Path = DEFAULT_FEATURE_DATA,
    artifact_path: Path = DEFAULT_MODEL_ARTIFACT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train exactly the already-frozen family and persist one portable bundle."""

    if config.get("selection_status") != "frozen_before_test":
        raise PermissionError("Production training requires a frozen Stage 9 model config.")
    cutoff = pd.Timestamp(config["final_training_cutoff"])
    training = select_training_rows_through(data, cutoff)
    family = config["selected_model_family"]
    parameters = dict(config["model_parameters"])
    if family in {"hist_gradient_boosting", "xgboost"}:
        pipeline, audit = fit_global_tree(training, config, family, parameters)
    elif family == "ridge":
        ridge_config = dict(config)
        ridge_config["model"] = parameters
        pipeline, audit = fit_global_ridge(training, ridge_config)
    elif family == "mean_28":
        pipeline = Mean28FeatureForecaster().fit(training[config["categorical_features"] + config["numeric_features"]])
        audit = {
            "model_family": family,
            "training_rows": len(training),
            "training_date_min": training["date"].min().date().isoformat(),
            "training_date_max": training["date"].max().date().isoformat(),
            "training_series": int(training.groupby(["store_id", "item_id"], observed=True).ngroups),
            "preprocessing": "none; sales_roll_mean_28 is rebuilt recursively",
            "validation_statistics_used_for_fit": False,
            "final_test_statistics_used_for_fit": False,
        }
    else:
        raise ValueError(f"Unsupported frozen production family: {family}")

    if training["date"].max() != cutoff:
        raise RuntimeError("Final production training does not end at the frozen cutoff.")
    bundle = {
        "artifact_version": 1,
        "pipeline": pipeline,
        "config": config,
        "model_family": family,
        "model_name": config["selected_model_name"],
        "trained_through": cutoff.date().isoformat(),
        "training_rows": len(training),
        "feature_data_sha256": file_sha256(feature_data_path),
    }
    save_forecaster(bundle, artifact_path)
    audit.update(
        {
            "artifact_path": artifact_path.as_posix(),
            "artifact_size_bytes": artifact_path.stat().st_size,
            "artifact_sha256": file_sha256(artifact_path),
            "training_cutoff_matches_freeze": True,
        }
    )
    return bundle, audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_FINAL_CONFIG)
    parser.add_argument("--feature-data", type=Path, default=DEFAULT_FEATURE_DATA)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_MODEL_ARTIFACT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.config.exists():
        raise FileNotFoundError("Run Stage 9 validation selection and freeze first.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        from smartstock.models.stage9_evaluation import load_stage9_data
    except ModuleNotFoundError:
        from stage9_evaluation import load_stage9_data
    data = load_stage9_data(
        args.feature_data, config, through_date=config["final_training_cutoff"]
    )
    _, audit = train_final_forecaster(
        data, config, feature_data_path=args.feature_data, artifact_path=args.artifact
    )
    print("SmartStock final forecaster trained")
    print(f"Model: {config['selected_model_name']}")
    print(f"Training rows: {audit['training_rows']:,}")
    print(f"Trained through: {audit['training_date_max']}")
    print(f"Artifact: {audit['artifact_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
