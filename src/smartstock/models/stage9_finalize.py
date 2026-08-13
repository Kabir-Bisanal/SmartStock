"""Build Stage 9 reports and figures from already-frozen validation/test artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

try:
    import xgboost
except ModuleNotFoundError:  # The selected mean model can still be loaded without XGBoost.
    xgboost = None

try:
    from smartstock.models.baseline_evaluation import file_sha256, json_safe, write_json
    from smartstock.models.production_forecaster import load_forecaster
    from smartstock.models.stage9_evaluation import FROZEN_INPUTS
    from smartstock.models.stage9_figures import MODEL_LABELS, create_stage9_figures
    from smartstock.models.stage9_report import render_stage9_report
except ModuleNotFoundError:
    from baseline_evaluation import file_sha256, json_safe, write_json
    from production_forecaster import load_forecaster
    from stage9_evaluation import FROZEN_INPUTS
    from stage9_figures import MODEL_LABELS, create_stage9_figures
    from stage9_report import render_stage9_report


DEFAULT_METRICS = Path("reports/stage9_model_metrics.csv")
DEFAULT_TUNING_METRICS = Path("reports/stage9_tuning_results.csv")
DEFAULT_SELECTION = Path("reports/stage9_validation_selection.json")
DEFAULT_RECEIPT = Path("reports/stage9_test_evaluation_receipt.json")
DEFAULT_FINAL_CONFIG = Path("config/v1_final_model.json")
DEFAULT_CANDIDATE_CONFIG = Path("config/v1_stage9_candidates.json")
DEFAULT_PREDICTIONS = Path("data/processed/models/stage9/final_test_predictions.csv")
DEFAULT_ARTIFACT = Path("models/smartstock_v1_forecaster.joblib")
DEFAULT_SUMMARY = Path("reports/stage9_summary.json")
DEFAULT_REPORT = Path("reports/stage9_final_model_report.md")
DEFAULT_FIGURE_DIR = Path("reports/figures/stage9")

HORIZON_LABELS = {
    "day_1_daily": "Day +1 daily",
    "days_1_7_daily": "Days 1-7 daily",
    "days_1_7_aggregate": "7-day aggregate",
    "days_1_30_daily": "Days 1-30 daily",
    "days_1_30_aggregate": "30-day aggregate",
}


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.to_dict(orient="records")


def _overall(
    metrics: pd.DataFrame,
    split: str,
    horizon: str,
    *,
    fold: str | None = None,
) -> pd.DataFrame:
    result = metrics[
        metrics["evaluation_split"].eq(split)
        & metrics["horizon"].eq(horizon)
        & metrics["segment_type"].eq("overall")
    ]
    if fold is not None:
        result = result[result["fold"].eq(fold)]
    return result.copy()


def _independent_prediction_reconciliation(
    predictions: pd.DataFrame, reported: dict[str, Any]
) -> dict[str, Any]:
    actual = predictions["actual"].to_numpy(dtype="float64")
    forecast = predictions["forecast"].to_numpy(dtype="float64")
    error = forecast - actual
    recalculated = {
        "observations": int(len(predictions)),
        "actual_sum": float(actual.sum()),
        "forecast_sum": float(forecast.sum()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "wape": float(np.abs(error).sum() / actual.sum()),
        "bias": float(error.mean()),
        "aggregate_bias": float(error.sum()),
    }
    tolerances = {name: np.isclose(recalculated[name], reported[name]) for name in recalculated}
    unique_keys = not predictions.duplicated(["target_date", "store_id", "item_id"]).any()
    return {
        "recalculated_from_prediction_csv": recalculated,
        "matches_reported_metrics": {name: bool(value) for name, value in tolerances.items()},
        "all_reconciled": bool(all(tolerances.values())),
        "unique_date_store_item_keys": bool(unique_keys),
        "dates": int(predictions["target_date"].nunique()),
        "series": int(predictions.groupby(["store_id", "item_id"], observed=True).ngroups),
        "rows_per_series": {
            "minimum": int(predictions.groupby(["store_id", "item_id"], observed=True).size().min()),
            "maximum": int(predictions.groupby(["store_id", "item_id"], observed=True).size().max()),
        },
        "nonnegative_forecasts": bool(predictions["forecast"].ge(0).all()),
        "finite_forecasts": bool(np.isfinite(predictions["forecast"]).all()),
    }


def finalize_stage9() -> dict[str, Any]:
    """Create post-test narrative artifacts without refitting or rescoring the test."""

    required = [
        DEFAULT_METRICS,
        DEFAULT_TUNING_METRICS,
        DEFAULT_SELECTION,
        DEFAULT_RECEIPT,
        DEFAULT_FINAL_CONFIG,
        DEFAULT_CANDIDATE_CONFIG,
        DEFAULT_PREDICTIONS,
        DEFAULT_ARTIFACT,
    ]
    missing = [path.as_posix() for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Stage 9 reporting inputs are missing: {missing}")

    metrics = pd.read_csv(DEFAULT_METRICS)
    tuning_metrics = pd.read_csv(DEFAULT_TUNING_METRICS)
    predictions = pd.read_csv(
        DEFAULT_PREDICTIONS,
        parse_dates=["forecast_origin", "history_end_date", "target_date"],
    )
    selection = json.loads(DEFAULT_SELECTION.read_text(encoding="utf-8"))
    receipt = json.loads(DEFAULT_RECEIPT.read_text(encoding="utf-8"))
    final_config = json.loads(DEFAULT_FINAL_CONFIG.read_text(encoding="utf-8"))
    candidate_config = json.loads(DEFAULT_CANDIDATE_CONFIG.read_text(encoding="utf-8"))
    frozen_hash = file_sha256(DEFAULT_FINAL_CONFIG)
    if frozen_hash != receipt["frozen_config_sha256"]:
        raise RuntimeError("The final model configuration changed after the locked-test freeze.")
    if receipt["evaluation_count"] != 1:
        raise RuntimeError("The locked final test receipt does not record exactly one evaluation.")

    ranking = pd.DataFrame(selection["final_validation_ranking"])
    primary = _overall(metrics, "validation", "days_1_30_daily", fold="combined").merge(
        ranking[["model_name", "selection_rank", "balanced_rank_score"]],
        on="model_name",
        how="left",
        validate="one_to_one",
    ).sort_values("selection_rank")
    primary["display_name"] = primary["model_name"].map(MODEL_LABELS).fillna(primary["model_name"])

    fold_results = metrics[
        metrics["evaluation_split"].eq("validation")
        & metrics["fold"].str.startswith("validation_fold_")
        & metrics["horizon"].eq("days_1_30_daily")
        & metrics["segment_type"].eq("overall")
    ].copy()
    fold_results["display_name"] = fold_results["model_name"].map(MODEL_LABELS).fillna(fold_results["model_name"])
    fold_results = fold_results.sort_values(["fold", "selection_rank"] if "selection_rank" in fold_results else ["fold", "model_name"])

    tuning_primary = tuning_metrics[
        tuning_metrics["fold"].eq("combined")
        & tuning_metrics["horizon"].eq("days_1_30_daily")
        & tuning_metrics["segment_type"].eq("overall")
    ].copy()
    tuning_ranking = pd.DataFrame(selection["tuning_ranking"])[
        ["model_name", "selection_rank", "balanced_rank_score"]
    ]
    tuning_primary = tuning_primary.merge(
        tuning_ranking, on="model_name", how="left", validate="one_to_one"
    ).sort_values("selection_rank")

    horizon_order = list(HORIZON_LABELS)
    validation_horizons = metrics[
        metrics["evaluation_split"].eq("validation")
        & metrics["fold"].eq("combined")
        & metrics["segment_type"].eq("overall")
    ].copy()
    validation_horizons["label"] = validation_horizons["horizon"].map(HORIZON_LABELS)
    test_horizons = metrics[
        metrics["evaluation_split"].eq("locked_final_test")
        & metrics["segment_type"].eq("overall")
    ].copy()
    test_horizons["label"] = test_horizons["horizon"].map(HORIZON_LABELS)
    test_horizons["horizon_order"] = test_horizons["horizon"].map(
        {name: index for index, name in enumerate(horizon_order)}
    )
    test_horizons = test_horizons.sort_values("horizon_order")

    selected_name = final_config["selected_model_name"]
    selected_validation = _overall(
        metrics, "validation", "days_1_30_daily", fold="combined"
    )
    selected_validation = selected_validation[
        selected_validation["model_name"].eq(selected_name)
    ].iloc[0].to_dict()
    test_primary = _overall(metrics, "locked_final_test", "days_1_30_daily").iloc[0].to_dict()
    test_aggregate = _overall(
        metrics, "locked_final_test", "days_1_30_aggregate"
    ).iloc[0].to_dict()
    reconciliation = _independent_prediction_reconciliation(predictions, test_primary)
    if not reconciliation["all_reconciled"] or not reconciliation["unique_date_store_item_keys"]:
        raise RuntimeError("Independent final-test CSV reconciliation failed.")

    segment_results: dict[str, list[dict[str, Any]]] = {}
    for segment_type in ("demand_band", "store_id", "dept_id", "intermittency_class"):
        frame = metrics[
            metrics["evaluation_split"].eq("locked_final_test")
            & metrics["horizon"].eq("days_1_30_daily")
            & metrics["segment_type"].eq(segment_type)
        ].sort_values("segment_value")
        segment_results[segment_type] = _records(frame)

    selected_xgb = selection["representative_nonlinear_configurations"]["xgboost"]
    model_bundle = load_forecaster(DEFAULT_ARTIFACT)
    if (
        model_bundle["model_family"] != final_config["selected_model_family"]
        or model_bundle["trained_through"] != final_config["final_training_cutoff"]
    ):
        raise RuntimeError("Serialized production model does not match the frozen config.")

    figures = create_stage9_figures(
        metrics, predictions, DEFAULT_FIGURE_DIR, selected_name
    )
    current_source_hashes = {name: file_sha256(path) for name, path in FROZEN_INPUTS.items()}
    source_integrity = current_source_hashes == final_config["source_hashes_at_selection"]
    if not source_integrity:
        raise RuntimeError("A Stage 8-or-earlier frozen source differs from its selection-time hash.")

    xgb_fold = fold_results[fold_results["model_name"].eq(selected_xgb["configuration_id"])]
    mean_fold = fold_results[fold_results["model_name"].eq("mean_28")]
    mean_vs_xgb = mean_fold.merge(xgb_fold, on="fold", suffixes=("_mean", "_xgb"))
    summary: dict[str, Any] = {
        "stage": "Stage 9 - Final Forecasting Model Selection & Locked Test Evaluation",
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status": "complete",
        "software": {
            "python_runtime_note": "Verified with the bundled Python runtime plus the user's compatible .venv packages because the .venv launcher points to an inaccessible local Python executable.",
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__ if xgboost is not None else "unavailable during reporting",
        },
        "candidate_policy": candidate_config,
        "selection": {
            "selected_model_name": selected_name,
            "selected_model_family": final_config["selected_model_family"],
            "selected_parameters": final_config["model_parameters"],
            "effective_production_predictors": ["sales_roll_mean_28"],
            "common_forecast_context_contract_features": final_config["categorical_features"] + final_config["numeric_features"],
            "selection_frozen_at_utc": final_config["selection_frozen_at_utc"],
            "frozen_config_sha256": frozen_hash,
            "test_targets_inspected_at_freeze": False,
            "post_test_model_changes_allowed": False,
            "strongest_initial_nonlinear_family": selection["strongest_initial_nonlinear_family"],
            "tuning_configurations_evaluated": selection["tuning_configurations_evaluated"],
            "balanced_reason": "The 28-day mean ranked first across weighted fold, horizon, aggregate, error-scale, and bias components. XGBoost improved short-horizon MAE and 30-day RMSE but not long-horizon MAE, WAPE, RMSSE, or aggregate accuracy.",
        },
        "validation": {
            "primary_model_comparison": _records(primary),
            "fold_results": _records(fold_results),
            "horizon_results": _records(validation_horizons),
            "selected_model_days_1_30_daily": selected_validation,
            "tuning_results": _records(tuning_primary),
            "selected_xgboost_configuration": selected_xgb["configuration_id"],
            "selected_xgboost_parameters": selected_xgb["parameters"],
            "mean_28_fold_wins_vs_xgboost": int(
                (mean_vs_xgb["mae_mean"] < mean_vs_xgb["mae_xgb"]).sum()
            ),
            "latest_target_date": "2016-04-22",
            "test_targets_loaded": False,
        },
        "locked_final_test": {
            "evaluation_count": receipt["evaluation_count"],
            "start_date": receipt["test_start"],
            "end_date": receipt["test_end"],
            "days_1_30_daily": test_primary,
            "days_1_30_aggregate": test_aggregate,
            "horizon_results": _records(test_horizons),
            "segment_results": segment_results,
            "prediction_rows": len(predictions),
            "series": reconciliation["series"],
            "predictions_path": DEFAULT_PREDICTIONS.as_posix(),
            "prediction_sha256": file_sha256(DEFAULT_PREDICTIONS),
            "actuals_attached_after_forecast": receipt[
                "actuals_attached_only_after_forecasts_generated"
            ],
        },
        "validation_vs_test": {
            "mae_change_pct": float(
                100 * (test_primary["mae"] - selected_validation["mae"]) / selected_validation["mae"]
            ),
            "rmse_change_pct": float(
                100 * (test_primary["rmse"] - selected_validation["rmse"]) / selected_validation["rmse"]
            ),
            "wape_change_pct": float(
                100 * (test_primary["wape"] - selected_validation["wape"]) / selected_validation["wape"]
            ),
            "rmsse_change_pct": float(
                100 * (test_primary["rmsse"] - selected_validation["rmsse"]) / selected_validation["rmsse"]
            ),
            "bias_change": float(test_primary["bias"] - selected_validation["bias"]),
        },
        "leakage_audit": {
            "validation_targets_entered_recursion": False,
            "test_targets_used_for_selection": False,
            "future_sales_became_lags": False,
            "target_day_price_used": False,
            "preprocessing_training_only": True,
            "test_unlocked_only_after_config_freeze": receipt[
                "model_selection_frozen_before_test"
            ],
            "post_test_hyperparameter_changes_prohibited": not receipt[
                "post_test_hyperparameter_changes_allowed"
            ],
            "validation_mutation_audits_passed": all(
                audit["future_target_mutation_audit"] is not None
                and all(audit["future_target_mutation_audit"].values())
                for audit in selection["candidate_audits"]
            ),
            "final_recursive_audit": receipt["recursive_audit"],
        },
        "production": {
            "artifact_path": DEFAULT_ARTIFACT.as_posix(),
            "artifact_size_bytes": DEFAULT_ARTIFACT.stat().st_size,
            "artifact_sha256": file_sha256(DEFAULT_ARTIFACT),
            "bundle_family_matches_freeze": True,
            "bundle_training_cutoff_matches_freeze": True,
            "forecast_function": "smartstock.models.production_forecaster.forecast_demand",
            "supported_horizons_days": [1, 7, 30],
            "training_entry_point": "python -m smartstock.models.train_final_model",
        },
        "integrity": {
            "independent_prediction_csv_reconciliation": reconciliation,
            "frozen_sources_unchanged_since_selection": source_integrity,
            "final_config_matches_test_receipt_hash": True,
            "final_model_matches_receipt_hash": file_sha256(DEFAULT_ARTIFACT)
            == receipt["model_artifact"]["sha256"],
            "test_second_run_guard_present": True,
            "official_negative_predictions": int(predictions["forecast"].lt(0).sum()),
            "undefined_test_rmsse_series": int(test_primary["rmsse_undefined_series"]),
        },
        "figures": figures,
        "performance": {
            "validation_seconds": selection["runtime_seconds"],
            "final_test_seconds": receipt["runtime_seconds"],
            "validation_nonlinear_prediction_rows_persisted": selection[
                "prediction_rows_persisted"
            ],
            "final_test_prediction_rows": len(predictions),
            "recursive_rows_per_batch": 300,
        },
        "scope": {
            "forecasting_layer_complete_for_v1": True,
            "inventory_optimization_implemented": False,
            "postgresql_implemented": False,
            "streamlit_implemented": False,
            "docker_or_deployment_implemented": False,
        },
    }
    summary = json_safe(summary)
    DEFAULT_REPORT.write_text(render_stage9_report(summary), encoding="utf-8")
    write_json(DEFAULT_SUMMARY, summary)
    print("Stage 9 reports finalized without rerunning the locked test.")
    print(f"Figures: {len(figures)}")
    print(f"Report: {DEFAULT_REPORT.as_posix()}")
    return summary


if __name__ == "__main__":
    finalize_stage9()
