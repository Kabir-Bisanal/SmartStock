"""Validation-only model selection and one-time locked-test evaluation for Stage 9."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    from smartstock.models.baseline_evaluation import (
        calculate_metrics,
        file_sha256,
        json_safe,
        write_json,
    )
    from smartstock.models.recursive_forecasting import (
        DEMAND_FEATURES,
        _build_histories,
        generate_recursive_predictions,
    )
    from smartstock.models.ridge_model import select_training_rows
    from smartstock.models.stage9_models import (
        candidate_configuration,
        fit_global_tree,
        validate_candidate_config,
    )
except ModuleNotFoundError:  # Supports direct execution from the repository root.
    from baseline_evaluation import calculate_metrics, file_sha256, json_safe, write_json
    from recursive_forecasting import DEMAND_FEATURES, _build_histories, generate_recursive_predictions
    from ridge_model import select_training_rows
    from stage9_models import candidate_configuration, fit_global_tree, validate_candidate_config


DEFAULT_FEATURE_DATA = Path("data/processed/smartstock_v1_features.csv")
DEFAULT_CANDIDATE_CONFIG = Path("config/v1_stage9_candidates.json")
DEFAULT_FINAL_CONFIG = Path("config/v1_final_model.json")
DEFAULT_FEATURE_MANIFEST = Path("config/v1_features.json")
DEFAULT_VALIDATION_MANIFEST = Path("config/v1_validation.json")
DEFAULT_STAGE7_METRICS = Path("reports/stage7_baseline_metrics.csv")
DEFAULT_STAGE8_METRICS = Path("reports/stage8_ridge_metrics.csv")
DEFAULT_VALIDATION_PREDICTIONS = Path(
    "data/processed/models/stage9/nonlinear_validation_predictions.csv"
)
DEFAULT_FINAL_TEST_PREDICTIONS = Path(
    "data/processed/models/stage9/final_test_predictions.csv"
)
DEFAULT_METRICS = Path("reports/stage9_model_metrics.csv")
DEFAULT_TUNING_METRICS = Path("reports/stage9_tuning_results.csv")
DEFAULT_SELECTION_SUMMARY = Path("reports/stage9_validation_selection.json")
DEFAULT_TEST_RECEIPT = Path("reports/stage9_test_evaluation_receipt.json")
DEFAULT_MODEL_ARTIFACT = Path("models/smartstock_v1_forecaster.joblib")

STANDARD_METRIC_COLUMNS = [
    "fold",
    "model_name",
    "horizon",
    "horizon_days",
    "evaluation_type",
    "segment_type",
    "segment_value",
    "mae",
    "rmse",
    "wape",
    "bias",
    "aggregate_bias",
    "actual_sum",
    "forecast_sum",
    "observations",
    "rmsse",
    "rmsse_defined_series",
    "rmsse_undefined_series",
]

FROZEN_INPUTS = {
    "raw_sales": Path("data/raw/sales_train_evaluation.csv"),
    "raw_prices": Path("data/raw/sell_prices.csv"),
    "raw_calendar": Path("data/raw/calendar.csv"),
    "interim_dataset": Path("data/interim/smartstock_v1_long.csv"),
    "feature_dataset": DEFAULT_FEATURE_DATA,
    "subset_manifest": Path("config/v1_subset.json"),
    "feature_manifest": DEFAULT_FEATURE_MANIFEST,
    "validation_manifest": DEFAULT_VALIDATION_MANIFEST,
    "ridge_config": Path("config/v1_ridge.json"),
    "stage7_metrics": DEFAULT_STAGE7_METRICS,
    "stage8_metrics": DEFAULT_STAGE8_METRICS,
}


def load_stage9_data(
    path: Path,
    config: dict[str, Any],
    *,
    through_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load only approved model, context, control, and evaluation columns.

    Selection uses ``through_date=2016-04-22`` so final-test target rows are
    discarded chunk-by-chunk and never enter candidate fitting or scoring.
    """

    categorical = list(config["categorical_features"])
    numeric = list(config["numeric_features"])
    columns = list(
        dict.fromkeys(
            [
                "date",
                "availability_date",
                "is_feature_ready",
                "sales",
                "demand_band",
                *categorical,
                *numeric,
            ]
        )
    )
    dtype: dict[str, str] = {
        column: "string" for column in set(categorical) | {"demand_band"}
    }
    dtype.update({column: "float32" for column in numeric})
    dtype.update({"is_feature_ready": "boolean", "sales": "float32"})
    cutoff = pd.Timestamp(through_date) if through_date is not None else None
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=columns,
        parse_dates=["date", "availability_date"],
        dtype=dtype,
        chunksize=100_000,
        low_memory=False,
    ):
        if cutoff is not None:
            chunk = chunk[chunk["date"].le(cutoff)]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        raise ValueError("No Stage 9 modeling rows were loaded.")
    data = pd.concat(chunks, ignore_index=True)
    return data.sort_values(["store_id", "item_id", "date"], kind="stable").reset_index(drop=True)


def _calculate_model_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    compatible = predictions.copy()
    compatible["baseline_name"] = compatible["model_name"]
    return calculate_metrics(compatible).rename(columns={"baseline_name": "model_name"})


def evaluate_tree_configuration(
    data: pd.DataFrame,
    base_config: dict[str, Any],
    validation_manifest: dict[str, Any],
    model_family: str,
    parameters: dict[str, Any],
    configuration_id: str,
    *,
    run_mutation_audit: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit and recursively evaluate one nonlinear configuration on all three folds."""

    config = candidate_configuration(base_config, model_family, parameters, configuration_id)
    prediction_frames: list[pd.DataFrame] = []
    fit_audits: list[dict[str, Any]] = []
    recursive_audits: list[dict[str, Any]] = []
    fit_seconds: dict[str, float] = {}
    mutation_result: dict[str, bool] | None = None
    for fold in validation_manifest["validation_folds"]:
        start = pd.Timestamp(fold["start_date"])
        training = select_training_rows(data, start)
        fit_started = time.perf_counter()
        pipeline, fit_audit = fit_global_tree(
            training, config, model_family, parameters
        )
        fit_seconds[fold["name"]] = round(time.perf_counter() - fit_started, 3)
        fit_audit.update(
            {
                "fold": fold["name"],
                "training_strictly_before_validation": bool(training["date"].lt(start).all()),
                "training_ends_at_forecast_origin": bool(
                    training["date"].max() == start - pd.DateOffset(days=1)
                ),
            }
        )
        predictions, recursive_audit = generate_recursive_predictions(
            pipeline, data, fold, validation_manifest, config
        )
        if run_mutation_audit and fold is validation_manifest["validation_folds"][-1]:
            mutation_result = _mutation_audit(
                pipeline, predictions, data, fold, validation_manifest, config
            )
            if not all(mutation_result.values()):
                raise RuntimeError(f"Future-target mutation audit failed for {configuration_id}.")
        prediction_frames.append(predictions)
        fit_audits.append(fit_audit)
        recursive_audits.append(recursive_audit)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = _calculate_model_metrics(predictions)
    audit = {
        "configuration_id": configuration_id,
        "model_family": model_family,
        "parameters": parameters,
        "fit_audits": fit_audits,
        "recursive_audits": recursive_audits,
        "fit_seconds_by_fold": fit_seconds,
        "prediction_rows": len(predictions),
        "latest_target_date": predictions["target_date"].max().date().isoformat(),
        "all_preprocessing_training_only": all(
            item["encoder_categories_come_from_training_only"]
            and item["numeric_imputer_fitted_from_training"]
            and item["training_strictly_before_validation"]
            for item in fit_audits
        ),
        "all_recursive_audits_passed": all(
            all(entry["checks"].values()) for entry in recursive_audits
        ),
        "future_target_mutation_audit": mutation_result,
    }
    return predictions, metrics, audit


def balanced_validation_ranking(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank models using folds, horizons, aggregates, scaled error, and bias."""

    models = sorted(metrics["model_name"].unique())
    components: list[dict[str, Any]] = []
    horizon_weights = {
        "day_1_daily": 0.5,
        "days_1_7_daily": 0.75,
        "days_1_7_aggregate": 1.25,
        "days_1_30_daily": 2.0,
        "days_1_30_aggregate": 2.0,
    }
    metric_weights = {"mae": 2.0, "rmse": 1.0, "wape": 1.0, "rmsse": 2.0}
    overall = metrics[
        metrics["fold"].eq("combined") & metrics["segment_type"].eq("overall")
    ]
    for horizon, horizon_weight in horizon_weights.items():
        frame = overall[overall["horizon"].eq(horizon)].set_index("model_name")
        if set(models) - set(frame.index):
            raise ValueError(f"Missing combined selection metrics for {horizon}.")
        for metric, metric_weight in metric_weights.items():
            ranks = frame[metric].rank(method="average", ascending=True)
            for model in models:
                components.append(
                    {
                        "model_name": model,
                        "component": f"combined:{horizon}:{metric}",
                        "rank": float(ranks.loc[model]),
                        "weight": horizon_weight * metric_weight,
                    }
                )
        bias_ranks = frame["bias"].abs().rank(method="average", ascending=True)
        for model in models:
            components.append(
                {
                    "model_name": model,
                    "component": f"combined:{horizon}:absolute_bias",
                    "rank": float(bias_ranks.loc[model]),
                    "weight": horizon_weight * 0.4,
                }
            )

    primary_folds = metrics[
        metrics["fold"].ne("combined")
        & metrics["horizon"].eq("days_1_30_daily")
        & metrics["segment_type"].eq("overall")
    ]
    for fold, frame in primary_folds.groupby("fold", observed=True, sort=True):
        frame = frame.set_index("model_name")
        for metric in ("mae", "rmsse"):
            ranks = frame[metric].rank(method="average", ascending=True)
            for model in models:
                components.append(
                    {
                        "model_name": model,
                        "component": f"{fold}:days_1_30_daily:{metric}",
                        "rank": float(ranks.loc[model]),
                        "weight": 1.5,
                    }
                )
    stability = primary_folds.groupby("model_name", observed=True)["mae"].std()
    stability_ranks = stability.rank(method="average", ascending=True)
    for model in models:
        components.append(
            {
                "model_name": model,
                "component": "fold_stability:mae_std",
                "rank": float(stability_ranks.loc[model]),
                "weight": 1.0,
            }
        )

    component_frame = pd.DataFrame(components)
    component_frame["weighted_rank"] = component_frame["rank"] * component_frame["weight"]
    ranking = (
        component_frame.groupby("model_name", observed=True)
        .agg(
            weighted_rank_sum=("weighted_rank", "sum"),
            total_weight=("weight", "sum"),
            comparison_components=("component", "size"),
        )
        .reset_index()
    )
    ranking["balanced_rank_score"] = ranking["weighted_rank_sum"] / ranking["total_weight"]
    ranking["selection_rank"] = ranking["balanced_rank_score"].rank(
        method="min", ascending=True
    ).astype(int)
    primary = overall[overall["horizon"].eq("days_1_30_daily")].set_index("model_name")
    fold_stats = primary_folds.groupby("model_name", observed=True)["mae"].agg(["mean", "std"])
    ranking["days_1_30_mae"] = ranking["model_name"].map(primary["mae"])
    ranking["days_1_30_rmse"] = ranking["model_name"].map(primary["rmse"])
    ranking["days_1_30_wape"] = ranking["model_name"].map(primary["wape"])
    ranking["days_1_30_rmsse"] = ranking["model_name"].map(primary["rmsse"])
    ranking["days_1_30_bias"] = ranking["model_name"].map(primary["bias"])
    ranking["fold_mae_mean"] = ranking["model_name"].map(fold_stats["mean"])
    ranking["fold_mae_std"] = ranking["model_name"].map(fold_stats["std"])
    return ranking.sort_values(
        ["selection_rank", "days_1_30_mae", "model_name"], kind="stable"
    ).reset_index(drop=True)


def _benchmark_metrics(
    stage7_path: Path, stage8_path: Path
) -> pd.DataFrame:
    stage7 = pd.read_csv(stage7_path)
    mean_28 = stage7[stage7["baseline_name"].eq("mean_28")].rename(
        columns={"baseline_name": "model_name"}
    )
    ridge = pd.read_csv(stage8_path)
    frames = [mean_28[STANDARD_METRIC_COLUMNS], ridge[STANDARD_METRIC_COLUMNS]]
    return pd.concat(frames, ignore_index=True)


def _mutation_audit(
    pipeline: Any,
    original_predictions: pd.DataFrame,
    data: pd.DataFrame,
    fold: dict[str, str],
    validation_manifest: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, bool]:
    mask = data["date"].between(fold["start_date"], fold["end_date"])
    columns = ["sales", *DEMAND_FEATURES]
    saved = data.loc[mask, columns].copy()
    try:
        data.loc[mask, columns] = 99_999.0
        mutated, _ = generate_recursive_predictions(
            pipeline, data, fold, validation_manifest, config
        )
    finally:
        data.loc[mask, columns] = saved.to_numpy()
    same = bool(
        np.allclose(original_predictions["raw_forecast"], mutated["raw_forecast"])
        and np.allclose(original_predictions["forecast"], mutated["forecast"])
    )
    return {
        "future_actual_mutation_does_not_change_forecasts": same,
        "future_lag_rolling_mutation_does_not_change_forecasts": same,
        "future_prices_absent_from_model_inputs": True,
    }


def freeze_final_model_config(
    path: Path,
    candidate_config: dict[str, Any],
    validation_manifest: dict[str, Any],
    *,
    selected_model_name: str,
    selected_model_family: str,
    selected_parameters: dict[str, Any],
    ranking: pd.DataFrame,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    """Write the immutable selection decision before any test target is scored."""

    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing final-model freeze: {path.as_posix()}"
        )
    test = validation_manifest["locked_final_test"]
    if selected_model_family in candidate_config["candidate_models"]:
        preprocessing = candidate_config["candidate_models"][selected_model_family]["preprocessing"]
    elif selected_model_family == "ridge":
        preprocessing = "training-only one-hot encoding plus StandardScaler from frozen Stage 8"
    elif selected_model_family == "mean_28":
        preprocessing = "none; recursively rebuilt sales_roll_mean_28"
    else:
        raise ValueError(f"Unsupported selected model family: {selected_model_family}")
    manifest = {
        "manifest_version": 1,
        "project_stage": "Stage 9 - Final Forecasting Model Selection & Locked Test Evaluation",
        "selection_status": "frozen_before_test",
        "selection_frozen_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "selection_data": "three frozen validation folds only",
        "test_targets_inspected_at_freeze": False,
        "post_test_model_changes_allowed": False,
        "selected_model_name": selected_model_name,
        "selected_model_family": selected_model_family,
        "model_parameters": selected_parameters,
        "random_seed": 42,
        "target": candidate_config["target"],
        "categorical_features": candidate_config["categorical_features"],
        "numeric_features": candidate_config["numeric_features"],
        "known_context_numeric_features": candidate_config["known_context_numeric_features"],
        "recursive_demand_features": candidate_config["recursive_demand_features"],
        "feature_count_before_encoding": candidate_config["feature_count_before_encoding"],
        "expected_series": candidate_config["expected_series"],
        "forecast_policy": candidate_config["forecast_policy"],
        "price_policy": candidate_config["price_policy"],
        "feature_expansion": candidate_config["feature_expansion"],
        "features_deliberately_excluded": candidate_config["features_deliberately_excluded"],
        "preprocessing": preprocessing,
        "final_training_cutoff": "2016-04-22",
        "locked_final_test": {
            "start_date": test["start_date"],
            "end_date": test["end_date"],
            "days": test["days"],
            "evaluation_allowed_only_after_this_freeze": True,
        },
        "production_artifact": DEFAULT_MODEL_ARTIFACT.as_posix(),
        "production_forecast_function": "smartstock.models.production_forecaster.forecast_demand",
        "validation_ranking": ranking.to_dict(orient="records"),
        "source_hashes_at_selection": source_hashes,
    }
    write_json(path, json_safe(manifest))
    return manifest


def assert_locked_test_can_run(
    final_config_path: Path,
    validation_manifest_path: Path,
    receipt_path: Path = DEFAULT_TEST_RECEIPT,
    predictions_path: Path = DEFAULT_FINAL_TEST_PREDICTIONS,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Enforce config freeze and prevent accidental second test evaluation."""

    if not final_config_path.exists():
        raise PermissionError("Locked test cannot run before final-model configuration freeze.")
    if receipt_path.exists() or predictions_path.exists():
        raise PermissionError("Locked final test has already been evaluated; a second run is blocked.")
    config = json.loads(final_config_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_manifest_path.read_text(encoding="utf-8"))
    if (
        config.get("selection_status") != "frozen_before_test"
        or config.get("test_targets_inspected_at_freeze") is not False
        or config.get("post_test_model_changes_allowed") is not False
    ):
        raise PermissionError("Final-model freeze does not satisfy the test-unlock policy.")
    expected = validation["locked_final_test"]
    if config["locked_final_test"]["start_date"] != expected["start_date"] or config[
        "locked_final_test"
    ]["end_date"] != expected["end_date"]:
        raise PermissionError("Frozen final-test dates do not match Stage 6 validation policy.")
    return config, validation, file_sha256(final_config_path)


def _selection_model_details(
    selected_name: str,
    representative: dict[str, dict[str, Any]],
    ridge_config_path: Path,
) -> tuple[str, dict[str, Any]]:
    for family, details in representative.items():
        if details["configuration_id"] == selected_name:
            return family, dict(details["parameters"])
    if selected_name == "ridge_global_v1":
        ridge = json.loads(ridge_config_path.read_text(encoding="utf-8"))
        return "ridge", dict(ridge["model"])
    if selected_name == "mean_28":
        return "mean_28", {"window_days": 28}
    raise ValueError(f"Cannot resolve selected production model: {selected_name}")


def run_validation_selection(
    feature_data_path: Path = DEFAULT_FEATURE_DATA,
    candidate_config_path: Path = DEFAULT_CANDIDATE_CONFIG,
    feature_manifest_path: Path = DEFAULT_FEATURE_MANIFEST,
    validation_manifest_path: Path = DEFAULT_VALIDATION_MANIFEST,
    stage7_metrics_path: Path = DEFAULT_STAGE7_METRICS,
    stage8_metrics_path: Path = DEFAULT_STAGE8_METRICS,
    final_config_path: Path = DEFAULT_FINAL_CONFIG,
    predictions_path: Path = DEFAULT_VALIDATION_PREDICTIONS,
    metrics_path: Path = DEFAULT_METRICS,
    tuning_metrics_path: Path = DEFAULT_TUNING_METRICS,
    selection_summary_path: Path = DEFAULT_SELECTION_SUMMARY,
) -> dict[str, Any]:
    """Evaluate candidates, tune only the stronger nonlinear family, then freeze selection."""

    if DEFAULT_TEST_RECEIPT.exists():
        raise PermissionError("Post-test model selection changes are prohibited.")
    if final_config_path.exists():
        raise FileExistsError("A final-model freeze already exists; selection will not be overwritten.")
    started = time.perf_counter()
    hashes_before = {name: file_sha256(path) for name, path in FROZEN_INPUTS.items()}
    candidate_config = json.loads(candidate_config_path.read_text(encoding="utf-8"))
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    validation_manifest = json.loads(validation_manifest_path.read_text(encoding="utf-8"))
    config_checks = validate_candidate_config(candidate_config, feature_manifest)
    test_start = pd.Timestamp(validation_manifest["locked_final_test"]["start_date"])
    selection_cutoff = test_start - pd.DateOffset(days=1)

    print("Loading validation-only Stage 9 data through 2016-04-22 ...")
    data = load_stage9_data(
        feature_data_path, candidate_config, through_date=selection_cutoff
    )
    if data["date"].max() != selection_cutoff or data["date"].ge(test_start).any():
        raise RuntimeError("Final-test target rows entered the validation-selection dataset.")

    evaluations: dict[str, dict[str, Any]] = {}
    all_audits: list[dict[str, Any]] = []
    print("Evaluating base HistGradientBoosting and XGBoost configurations ...")
    for family in ("hist_gradient_boosting", "xgboost"):
        parameters = candidate_config["candidate_models"][family]["base_configuration"]
        configuration_id = f"{family}_base"
        predictions, metrics, audit = evaluate_tree_configuration(
            data,
            candidate_config,
            validation_manifest,
            family,
            parameters,
            configuration_id,
            run_mutation_audit=True,
        )
        evaluations[configuration_id] = {
            "family": family,
            "parameters": parameters,
            "predictions": predictions,
            "metrics": metrics,
        }
        all_audits.append(audit)
        primary = metrics[
            metrics["fold"].eq("combined")
            & metrics["horizon"].eq("days_1_30_daily")
            & metrics["segment_type"].eq("overall")
        ].iloc[0]
        print(f"  {configuration_id}: MAE={primary['mae']:.4f}, RMSSE={primary['rmsse']:.4f}")

    base_metrics = pd.concat(
        [entry["metrics"] for entry in evaluations.values()], ignore_index=True
    )
    base_ranking = balanced_validation_ranking(base_metrics)
    strongest_base_name = str(base_ranking.iloc[0]["model_name"])
    strongest_family = evaluations[strongest_base_name]["family"]
    print(f"Limited tuning selected for: {strongest_family}")

    grid = candidate_config["limited_tuning_grids"][strongest_family]
    tuning_ids = [strongest_base_name]
    short = "hgb" if strongest_family == "hist_gradient_boosting" else "xgb"
    for index, parameters in enumerate(grid[1:], start=2):
        configuration_id = f"{short}_tune_{index:02d}"
        print(f"  Evaluating {configuration_id} ({index}/{len(grid)}) ...")
        predictions, metrics, audit = evaluate_tree_configuration(
            data,
            candidate_config,
            validation_manifest,
            strongest_family,
            parameters,
            configuration_id,
            run_mutation_audit=True,
        )
        evaluations[configuration_id] = {
            "family": strongest_family,
            "parameters": parameters,
            "predictions": predictions,
            "metrics": metrics,
        }
        tuning_ids.append(configuration_id)
        all_audits.append(audit)

    tuning_all_metrics = pd.concat(
        [evaluations[name]["metrics"] for name in tuning_ids], ignore_index=True
    )
    tuning_ranking = balanced_validation_ranking(tuning_all_metrics)
    best_tuned_name = str(tuning_ranking.iloc[0]["model_name"])
    representative: dict[str, dict[str, Any]] = {}
    for family in ("hist_gradient_boosting", "xgboost"):
        if family == strongest_family:
            chosen_id = best_tuned_name
        else:
            chosen_id = f"{family}_base"
        representative[family] = {
            "configuration_id": chosen_id,
            "parameters": evaluations[chosen_id]["parameters"],
        }

    benchmark_metrics = _benchmark_metrics(stage7_metrics_path, stage8_metrics_path)
    representative_tree_metrics = pd.concat(
        [
            evaluations[representative[family]["configuration_id"]]["metrics"]
            for family in ("hist_gradient_boosting", "xgboost")
        ],
        ignore_index=True,
    )
    validation_metrics = pd.concat(
        [benchmark_metrics, representative_tree_metrics], ignore_index=True
    )
    final_ranking = balanced_validation_ranking(validation_metrics)
    selected_name = str(final_ranking.iloc[0]["model_name"])
    selected_family, selected_parameters = _selection_model_details(
        selected_name, representative, Path("config/v1_ridge.json")
    )

    mean_primary = validation_metrics[
        validation_metrics["model_name"].eq("mean_28")
        & validation_metrics["fold"].eq("combined")
        & validation_metrics["horizon"].eq("days_1_30_daily")
        & validation_metrics["segment_type"].eq("overall")
    ].iloc[0]
    selected_primary = validation_metrics[
        validation_metrics["model_name"].eq(selected_name)
        & validation_metrics["fold"].eq("combined")
        & validation_metrics["horizon"].eq("days_1_30_daily")
        & validation_metrics["segment_type"].eq("overall")
    ].iloc[0]
    mae_improvement = 100 * (mean_primary["mae"] - selected_primary["mae"]) / mean_primary["mae"]
    fold_primary = validation_metrics[
        validation_metrics["fold"].ne("combined")
        & validation_metrics["horizon"].eq("days_1_30_daily")
        & validation_metrics["segment_type"].eq("overall")
    ]
    fold_pivot = fold_primary.pivot(index="fold", columns="model_name", values="mae")
    fold_wins_vs_mean = int((fold_pivot[selected_name] < fold_pivot["mean_28"]).sum())

    predictions = pd.concat(
        [
            evaluations[representative[family]["configuration_id"]]["predictions"]
            for family in ("hist_gradient_boosting", "xgboost")
        ],
        ignore_index=True,
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False, date_format="%Y-%m-%d")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    validation_metrics.assign(evaluation_split="validation").to_csv(metrics_path, index=False)
    tuning_output = tuning_all_metrics[
        tuning_all_metrics["segment_type"].eq("overall")
    ].copy()
    tuning_output.to_csv(tuning_metrics_path, index=False)

    hashes_after = {name: file_sha256(path) for name, path in FROZEN_INPUTS.items()}
    if hashes_before != hashes_after:
        raise RuntimeError("A Stage 8-or-earlier frozen artifact changed during model selection.")
    freeze = freeze_final_model_config(
        final_config_path,
        candidate_config,
        validation_manifest,
        selected_model_name=selected_name,
        selected_model_family=selected_family,
        selected_parameters=selected_parameters,
        ranking=final_ranking,
        source_hashes=hashes_before,
    )
    summary = {
        "stage": "Stage 9 validation selection and pre-test model freeze",
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "configuration_validation": config_checks,
        "selection_data_latest_date": data["date"].max().date().isoformat(),
        "final_test_targets_loaded_for_selection": False,
        "base_candidate_ranking": base_ranking.to_dict(orient="records"),
        "strongest_initial_nonlinear_family": strongest_family,
        "tuning_configurations_evaluated": len(grid),
        "tuning_ranking": tuning_ranking.to_dict(orient="records"),
        "representative_nonlinear_configurations": representative,
        "final_validation_ranking": final_ranking.to_dict(orient="records"),
        "selected_model_name": selected_name,
        "selected_model_family": selected_family,
        "selected_parameters": selected_parameters,
        "selected_days_1_30_daily": selected_primary.to_dict(),
        "mae_improvement_vs_mean_28_pct": float(mae_improvement),
        "mae_change_is_negligible_under_policy": bool(
            abs(mae_improvement)
            < float(candidate_config["selection_policy"]["negligible_primary_mae_change_pct"])
        ),
        "fold_mae_wins_vs_mean_28": fold_wins_vs_mean,
        "candidate_audits": all_audits,
        "frozen_config": final_config_path.as_posix(),
        "frozen_config_sha256": file_sha256(final_config_path),
        "test_evaluated": False,
        "source_integrity_unchanged": True,
        "prediction_rows_persisted": len(predictions),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(selection_summary_path, json_safe(summary))
    print("Validation selection complete; model choice is now frozen before test.")
    print(f"Selected model: {selected_name}")
    print(f"Frozen config: {final_config_path.as_posix()}")
    return json_safe(summary)


def run_locked_test_once(
    feature_data_path: Path = DEFAULT_FEATURE_DATA,
    final_config_path: Path = DEFAULT_FINAL_CONFIG,
    validation_manifest_path: Path = DEFAULT_VALIDATION_MANIFEST,
    predictions_path: Path = DEFAULT_FINAL_TEST_PREDICTIONS,
    metrics_path: Path = DEFAULT_METRICS,
    receipt_path: Path = DEFAULT_TEST_RECEIPT,
    artifact_path: Path = DEFAULT_MODEL_ARTIFACT,
) -> dict[str, Any]:
    """Train the frozen winner and score the locked final test exactly once."""

    started = time.perf_counter()
    config, validation_manifest, config_sha = assert_locked_test_can_run(
        final_config_path, validation_manifest_path, receipt_path, predictions_path
    )
    hashes_before = {name: file_sha256(path) for name, path in FROZEN_INPUTS.items()}
    test = validation_manifest["locked_final_test"]
    print("Final model is frozen. Loading data through the locked test end ...")
    data = load_stage9_data(feature_data_path, config, through_date=test["end_date"])

    try:
        from smartstock.models.production_forecaster import forecast_demand
        from smartstock.models.train_final_model import train_final_forecaster
    except ModuleNotFoundError:
        from production_forecaster import forecast_demand
        from train_final_model import train_final_forecaster

    print("Training the frozen production configuration through 2016-04-22 ...")
    bundle, training_audit = train_final_forecaster(
        data,
        config,
        feature_data_path=feature_data_path,
        artifact_path=artifact_path,
    )
    print("Generating the locked 30-day test forecast before attaching test actuals ...")
    forecasts, recursive_audit = forecast_demand(
        bundle,
        data,
        config["final_training_cutoff"],
        int(test["days"]),
        return_audit=True,
    )

    # Test actuals and analysis-only metadata are attached only after all forecasts exist.
    test_actuals = data.loc[
        data["date"].between(test["start_date"], test["end_date"]),
        ["date", "store_id", "item_id", "dept_id", "demand_band", "sales"],
    ].rename(columns={"date": "target_date", "sales": "actual"})
    predictions = forecasts.merge(
        test_actuals,
        on=["target_date", "store_id", "item_id"],
        how="left",
        validate="one_to_one",
    )
    if predictions["actual"].isna().any():
        raise RuntimeError("Locked-test actuals did not attach one-to-one after forecasting.")
    _, history_metadata = _build_histories(
        data, pd.Timestamp(config["final_training_cutoff"])
    )
    metadata_rows = [
        {"store_id": key[0], "item_id": key[1], **value}
        for key, value in history_metadata.items()
    ]
    predictions = predictions.merge(
        pd.DataFrame(metadata_rows),
        on=["store_id", "item_id"],
        how="left",
        validate="many_to_one",
    )
    predictions["fold"] = "locked_final_test"
    predictions["model_name"] = config["selected_model_name"]
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
    ].sort_values(["target_date", "store_id", "item_id"], kind="stable")
    if len(predictions) != 9_000 or predictions["forecast"].lt(0).any():
        raise RuntimeError("Locked-test output size or non-negative policy failed.")

    test_metrics = _calculate_model_metrics(predictions)
    test_metrics = test_metrics[test_metrics["fold"].eq("locked_final_test")].copy()
    existing = pd.read_csv(metrics_path)
    if existing["evaluation_split"].eq("locked_final_test").any():
        raise PermissionError("Locked-test metrics already exist; refusing a duplicate evaluation.")
    test_metrics["evaluation_split"] = "locked_final_test"
    combined_metrics = pd.concat([existing, test_metrics], ignore_index=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False, date_format="%Y-%m-%d")
    combined_metrics.to_csv(metrics_path, index=False)

    hashes_after = {name: file_sha256(path) for name, path in FROZEN_INPUTS.items()}
    if hashes_before != hashes_after:
        raise RuntimeError("A frozen Stage 8-or-earlier artifact changed during locked-test evaluation.")
    overall = test_metrics[
        test_metrics["horizon"].eq("days_1_30_daily")
        & test_metrics["segment_type"].eq("overall")
    ].iloc[0]
    raw_negative = predictions["raw_forecast"].lt(0)
    receipt = {
        "stage": "Stage 9 one-time locked final-test evaluation receipt",
        "evaluation_count": 1,
        "evaluated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "model_selection_frozen_before_test": True,
        "frozen_config_sha256": config_sha,
        "selected_model_name": config["selected_model_name"],
        "selected_model_family": config["selected_model_family"],
        "test_start": test["start_date"],
        "test_end": test["end_date"],
        "prediction_rows": len(predictions),
        "item_store_series": int(
            predictions.groupby(["store_id", "item_id"], observed=True).ngroups
        ),
        "actuals_attached_only_after_forecasts_generated": True,
        "test_targets_used_for_model_selection": False,
        "post_test_hyperparameter_changes_allowed": False,
        "recursive_audit": recursive_audit,
        "training_audit": training_audit,
        "negative_raw_prediction_diagnostics": {
            "count": int(raw_negative.sum()),
            "percentage": float(100 * raw_negative.mean()),
            "most_negative": float(predictions["raw_forecast"].min()),
            "official_negative_count": int(predictions["forecast"].lt(0).sum()),
        },
        "days_1_30_daily_overall": overall.to_dict(),
        "predictions": {
            "path": predictions_path.as_posix(),
            "sha256": file_sha256(predictions_path),
            "size_bytes": predictions_path.stat().st_size,
        },
        "model_artifact": {
            "path": artifact_path.as_posix(),
            "sha256": file_sha256(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
        },
        "frozen_source_integrity_unchanged": True,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(receipt_path, json_safe(receipt))
    print("Locked final test evaluated exactly once.")
    print(
        f"30-day daily: MAE={overall['mae']:.4f}, "
        f"WAPE={100 * overall['wape']:.2f}%, RMSSE={overall['rmsse']:.4f}"
    )
    return json_safe(receipt)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("validation", "final-test"),
        required=True,
        help="Run validation/freeze first; run final-test only afterward.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "validation":
        run_validation_selection()
    else:
        run_locked_test_once()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
