"""Run SmartStock Stage 8 global Ridge validation with true recursive forecasting."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import sklearn

try:
    from smartstock.models.baseline_evaluation import calculate_metrics, file_sha256, json_safe, write_json
    from smartstock.models.recursive_forecasting import DEMAND_FEATURES, generate_recursive_predictions
    from smartstock.models.ridge_model import fit_global_ridge, select_training_rows, validate_ridge_config
    from smartstock.models.stage8_figures import create_stage8_figures
    from smartstock.models.stage8_report import render_stage8_report
except ModuleNotFoundError:  # Supports direct execution from the repository root.
    from baseline_evaluation import calculate_metrics, file_sha256, json_safe, write_json
    from recursive_forecasting import DEMAND_FEATURES, generate_recursive_predictions
    from ridge_model import fit_global_ridge, select_training_rows, validate_ridge_config
    from stage8_figures import create_stage8_figures
    from stage8_report import render_stage8_report


DEFAULT_FEATURE_DATA = Path("data") / "processed" / "smartstock_v1_features.csv"
DEFAULT_RIDGE_CONFIG = Path("config") / "v1_ridge.json"
DEFAULT_FEATURE_MANIFEST = Path("config") / "v1_features.json"
DEFAULT_VALIDATION_MANIFEST = Path("config") / "v1_validation.json"
DEFAULT_STAGE7_METRICS = Path("reports") / "stage7_baseline_metrics.csv"
DEFAULT_PREDICTIONS = Path("data") / "processed" / "models" / "stage8" / "ridge_validation_predictions.csv"
DEFAULT_METRICS = Path("reports") / "stage8_ridge_metrics.csv"
DEFAULT_SUMMARY = Path("reports") / "stage8_summary.json"
DEFAULT_REPORT = Path("reports") / "stage8_ridge_evaluation_report.md"
DEFAULT_FIGURE_DIR = Path("reports") / "figures" / "stage8"

FROZEN_INPUTS = {
    "raw_sales": Path("data/raw/sales_train_evaluation.csv"),
    "raw_prices": Path("data/raw/sell_prices.csv"),
    "raw_calendar": Path("data/raw/calendar.csv"),
    "interim_dataset": Path("data/interim/smartstock_v1_long.csv"),
    "feature_dataset": DEFAULT_FEATURE_DATA,
    "subset_manifest": Path("config/v1_subset.json"),
    "feature_manifest": DEFAULT_FEATURE_MANIFEST,
    "validation_manifest": DEFAULT_VALIDATION_MANIFEST,
    "stage7_metrics": DEFAULT_STAGE7_METRICS,
    "stage7_summary": Path("reports/stage7_summary.json"),
    "stage7_report": Path("reports/stage7_baseline_evaluation_report.md"),
    "stage7_metric_code": Path("src/smartstock/models/metrics.py"),
    "stage7_evaluation_code": Path("src/smartstock/models/baseline_evaluation.py"),
}


def load_ridge_data(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    """Load only approved Ridge inputs, controls, and evaluation metadata."""

    categorical = list(config["categorical_features"])
    numeric = list(config["numeric_features"])
    columns = [
        "date",
        "availability_date",
        "is_feature_ready",
        "sales",
        "demand_band",
        *categorical,
        *numeric,
    ]
    columns = list(dict.fromkeys(columns))
    string_columns = set(categorical) | {"demand_band"}
    dtype: dict[str, str] = {column: "string" for column in string_columns}
    dtype.update({column: "float32" for column in numeric})
    dtype.update({"is_feature_ready": "boolean", "sales": "float32"})
    data = pd.read_csv(
        path,
        usecols=columns,
        parse_dates=["date", "availability_date"],
        dtype=dtype,
        low_memory=False,
    )
    return data.sort_values(["store_id", "item_id", "date"], kind="stable").reset_index(drop=True)


def calculate_ridge_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Reuse Stage 7 horizon, segment, and metric definitions exactly."""

    compatible = predictions.copy()
    compatible["baseline_name"] = compatible["model_name"]
    metrics = calculate_metrics(compatible).rename(columns={"baseline_name": "model_name"})
    return metrics


def compare_with_stage7(ridge_metrics: pd.DataFrame, stage7_metrics: pd.DataFrame) -> pd.DataFrame:
    """Attach official mean-28 and zero metrics and calculate transparent improvements."""

    keys = ["fold", "horizon", "horizon_days", "evaluation_type", "segment_type", "segment_value"]
    value_columns = [
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
    result = ridge_metrics.copy()
    for baseline in ("mean_28", "zero"):
        source = stage7_metrics[stage7_metrics["baseline_name"].eq(baseline)][keys + value_columns].copy()
        source = source.rename(columns={column: f"{baseline}_{column}" for column in value_columns})
        result = result.merge(source, on=keys, how="left", validate="one_to_one")
    if result["mean_28_mae"].isna().any() or result["zero_mae"].isna().any():
        raise ValueError("Stage 7 comparison rows do not align with Stage 8 metrics.")
    for metric in ("mae", "rmse", "wape", "rmsse"):
        denominator = result[f"mean_28_{metric}"].replace(0, np.nan)
        result[f"{metric}_improvement_vs_mean_28_pct"] = 100 * (denominator - result[metric]) / denominator
        zero_denominator = result[f"zero_{metric}"].replace(0, np.nan)
        result[f"{metric}_improvement_vs_zero_pct"] = 100 * (zero_denominator - result[metric]) / zero_denominator
    mean_abs_bias = result["mean_28_bias"].abs().replace(0, np.nan)
    result["absolute_bias_improvement_vs_mean_28_pct"] = 100 * (mean_abs_bias - result["bias"].abs()) / mean_abs_bias
    return result


def _slice_records(
    comparison: pd.DataFrame,
    *,
    fold: str | None = None,
    horizon: str | None = None,
    segment_type: str | None = None,
) -> list[dict[str, Any]]:
    frame = comparison
    if fold is not None:
        frame = frame[frame["fold"].eq(fold)]
    if horizon is not None:
        frame = frame[frame["horizon"].eq(horizon)]
    if segment_type is not None:
        frame = frame[frame["segment_type"].eq(segment_type)]
    return frame.to_dict(orient="records")


def summarize_results(comparison: pd.DataFrame) -> dict[str, Any]:
    """Create the required fold, horizon, segment, and balanced comparison summaries."""

    primary = comparison[
        comparison["fold"].eq("combined")
        & comparison["horizon"].eq("days_1_30_daily")
        & comparison["segment_type"].eq("overall")
    ].iloc[0].to_dict()
    fold_results = comparison[
        comparison["fold"].ne("combined")
        & comparison["horizon"].eq("days_1_30_daily")
        & comparison["segment_type"].eq("overall")
    ].sort_values("fold").to_dict(orient="records")
    horizon_order = [
        "day_1_daily",
        "days_1_7_daily",
        "days_1_7_aggregate",
        "days_1_30_daily",
        "days_1_30_aggregate",
    ]
    horizon_frame = comparison[
        comparison["fold"].eq("combined") & comparison["segment_type"].eq("overall")
    ].set_index("horizon").reindex(horizon_order).reset_index()
    segments = {
        name: comparison[
            comparison["fold"].eq("combined")
            & comparison["horizon"].eq("days_1_30_daily")
            & comparison["segment_type"].eq(name)
        ].sort_values("segment_value").to_dict(orient="records")
        for name in ("demand_band", "store_id", "dept_id", "intermittency_class")
    }
    improvements = [row["mae_improvement_vs_mean_28_pct"] for row in fold_results]
    consistent = all(value > 0 for value in improvements) or all(value < 0 for value in improvements)
    wins = sum(primary[f"{metric}_improvement_vs_mean_28_pct"] > 0 for metric in ("mae", "rmse", "wape", "rmsse"))
    aggregate_30 = horizon_frame[horizon_frame["horizon"].eq("days_1_30_aggregate")].iloc[0]
    if wins >= 3 and aggregate_30["mae_improvement_vs_mean_28_pct"] > 0:
        conclusion = "Ridge meaningfully outperformed the 28-day mean on most principal validation metrics, although segment and fold trade-offs still matter."
        recommendation = "A controlled nonlinear global model is justified next to test whether interactions and thresholds improve further, while retaining Ridge and the 28-day mean as benchmarks."
    else:
        conclusion = "Ridge did not establish a balanced, consistent improvement over the 28-day mean. Its results remain useful as the first global linear-model benchmark."
        recommendation = "The next reviewed experiment may consider one controlled nonlinear global model because Ridge's additive linear assumptions did not dominate the simple baseline."
    stability = (
        "Ridge's MAE direction versus the 28-day mean was consistent across all three folds."
        if consistent
        else "Ridge's MAE improvement was not consistent across all three folds, so time-period sensitivity is an important warning."
    )
    return {
        "primary_comparison": primary,
        "fold_results": fold_results,
        "horizon_results": horizon_frame.to_dict(orient="records"),
        "segment_results": segments,
        "fold_mae_improvement_range_pct": [float(min(improvements)), float(max(improvements))],
        "fold_improvement_direction_consistent": consistent,
        "fold_stability_statement": stability,
        "principal_metric_wins_out_of_four": int(wins),
        "balanced_conclusion": conclusion,
        "stage9_recommendation": recommendation,
    }


def run_stage8(
    feature_data_path: Path = DEFAULT_FEATURE_DATA,
    ridge_config_path: Path = DEFAULT_RIDGE_CONFIG,
    feature_manifest_path: Path = DEFAULT_FEATURE_MANIFEST,
    validation_manifest_path: Path = DEFAULT_VALIDATION_MANIFEST,
    stage7_metrics_path: Path = DEFAULT_STAGE7_METRICS,
    predictions_path: Path = DEFAULT_PREDICTIONS,
    metrics_path: Path = DEFAULT_METRICS,
    summary_path: Path = DEFAULT_SUMMARY,
    report_path: Path = DEFAULT_REPORT,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
) -> dict[str, Any]:
    """Fit three global Ridge models and evaluate only the frozen validation folds."""

    started = time.perf_counter()
    hashes_before = {name: file_sha256(path) for name, path in FROZEN_INPUTS.items()}
    config = json.loads(ridge_config_path.read_text(encoding="utf-8"))
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    validation_manifest = json.loads(validation_manifest_path.read_text(encoding="utf-8"))
    config_checks = validate_ridge_config(config, feature_manifest)
    if not validation_manifest["locked_final_test"]["locked"]:
        raise ValueError("Stage 8 requires the frozen final test to remain locked.")

    print("Loading approved Stage 8 fields from the frozen feature dataset once ...")
    data = load_ridge_data(feature_data_path, config)
    test_start = pd.Timestamp(validation_manifest["locked_final_test"]["start_date"])
    pretest_data = data[data["date"].lt(test_start)].copy()
    del data
    stage7_metrics = pd.read_csv(stage7_metrics_path)

    fold_predictions: list[pd.DataFrame] = []
    origin_audits: list[dict[str, Any]] = []
    preprocessing_audits: list[dict[str, Any]] = []
    mutation_checks: list[dict[str, Any]] = []
    fit_seconds: dict[str, float] = {}
    print("Fitting one global Ridge pipeline per fold and forecasting recursively ...")
    for fold in validation_manifest["validation_folds"]:
        start = pd.Timestamp(fold["start_date"])
        origin = start + pd.DateOffset(days=-1)
        training = select_training_rows(pretest_data, start)
        fit_started = time.perf_counter()
        pipeline, preprocessing_audit = fit_global_ridge(training, config)
        fit_seconds[fold["name"]] = round(time.perf_counter() - fit_started, 3)
        preprocessing_audit.update(
            {
                "fold": fold["name"],
                "forecast_origin": origin.date().isoformat(),
                "training_ends_at_origin": training["date"].max() == origin,
                "training_strictly_before_validation": training["date"].lt(start).all(),
            }
        )
        predictions, origin_audit = generate_recursive_predictions(
            pipeline, pretest_data, fold, validation_manifest, config
        )

        # Mutation audit: future actuals and persisted future demand features must not affect recursion.
        validation_mask = pretest_data["date"].between(fold["start_date"], fold["end_date"])
        mutation_columns = ["sales", *DEMAND_FEATURES]
        original_values = pretest_data.loc[validation_mask, mutation_columns].copy()
        try:
            pretest_data.loc[validation_mask, mutation_columns] = 9999.0
            mutated, _ = generate_recursive_predictions(
                pipeline, pretest_data, fold, validation_manifest, config
            )
        finally:
            pretest_data.loc[validation_mask, mutation_columns] = original_values.to_numpy()
        forecasts_unchanged = bool(
            np.allclose(predictions["raw_forecast"], mutated["raw_forecast"])
            and np.allclose(predictions["forecast"], mutated["forecast"])
        )
        actuals_changed = not np.array_equal(predictions["actual"], mutated["actual"])
        mutation_checks.append(
            {
                "fold": fold["name"],
                "validation_actual_mutation_does_not_change_forecasts": forecasts_unchanged,
                "precomputed_future_lag_rolling_mutation_does_not_change_forecasts": forecasts_unchanged,
                "mutated_actuals_were_observably_different": actuals_changed,
                "future_price_mutation_irrelevant_because_price_columns_not_loaded": True,
            }
        )
        if not forecasts_unchanged or not actuals_changed:
            raise RuntimeError(f"Stage 8 validation mutation audit failed for {fold['name']}.")
        fold_predictions.append(predictions)
        origin_audits.append(origin_audit)
        preprocessing_audits.append(preprocessing_audit)

    predictions = pd.concat(fold_predictions, ignore_index=True)
    if predictions["target_date"].max() >= test_start:
        raise RuntimeError("Final-test lock violation: Ridge predictions reach the locked test.")
    print("Calculating Stage 7-compatible metrics and comparisons ...")
    ridge_metrics = calculate_ridge_metrics(predictions)
    comparison = compare_with_stage7(ridge_metrics, stage7_metrics)
    results = summarize_results(comparison)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False, date_format="%Y-%m-%d")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(metrics_path, index=False)
    print("Creating Stage 8 figures ...")
    figures = create_stage8_figures(comparison, figure_dir)

    hashes_after = {name: file_sha256(path) for name, path in FROZEN_INPUTS.items()}
    integrity = {
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
        "unchanged": hashes_before == hashes_after,
    }
    if not integrity["unchanged"]:
        raise RuntimeError("A frozen Stage 7 or earlier artifact changed during Stage 8.")

    raw_negative = predictions["raw_forecast"].lt(0)
    negative_diagnostics = {
        "count": int(raw_negative.sum()),
        "percentage": float(100 * raw_negative.mean()),
        "most_negative": float(predictions["raw_forecast"].min()),
        "official_negative_count": int(predictions["forecast"].lt(0).sum()),
        "forecasts_rounded": False,
    }
    preprocessing_all_passed = all(
        audit["feature_ready_only"]
        and audit["scaler_fit_row_count_matches_training"]
        and audit["encoder_categories_come_from_training_only"]
        and audit["target_excluded_from_preprocessing"]
        and audit["training_ends_at_origin"]
        and audit["training_strictly_before_validation"]
        for audit in preprocessing_audits
    )
    mutation_passed = all(all(value for key, value in check.items() if key != "fold") for check in mutation_checks)
    summary: dict[str, Any] = {
        "stage": "Stage 8 - Global Ridge Regression & Recursive Multi-Step Forecasting",
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "question": "Can a fixed global Ridge model outperform the 28-day historical mean under origin-safe recursive validation?",
        "model_configuration": config,
        "config_validation": config_checks,
        "software": {"scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "forecast_origin_audits": origin_audits,
        "mutation_leakage_audit": {"folds": mutation_checks, "passed": mutation_passed},
        "preprocessing_leakage_audit": {"folds": preprocessing_audits, "all_checks_passed": preprocessing_all_passed},
        "negative_prediction_diagnostics": negative_diagnostics,
        "predictions": {
            "rows": len(predictions),
            "folds": int(predictions["fold"].nunique()),
            "models_per_fold": 1,
            "item_store_pairs": int(predictions.groupby(["store_id", "item_id"], observed=True).ngroups),
            "date_min": predictions["target_date"].min().date().isoformat(),
            "date_max": predictions["target_date"].max().date().isoformat(),
            "file": predictions_path.as_posix(),
            "file_size_bytes": predictions_path.stat().st_size,
            "sha256": file_sha256(predictions_path),
        },
        "metrics": {"rows": len(comparison), "file": metrics_path.as_posix(), "file_size_bytes": metrics_path.stat().st_size},
        "results": results,
        "figures": figures,
        "final_test_lock": {
            "start_date": validation_manifest["locked_final_test"]["start_date"],
            "end_date": validation_manifest["locked_final_test"]["end_date"],
            "predictions_generated": False,
            "metrics_calculated": False,
            "latest_evaluated_target_date": predictions["target_date"].max().date().isoformat(),
            "locked_and_untouched": bool(predictions["target_date"].max() < test_start),
        },
        "source_integrity": integrity,
        "performance": {
            "feature_csv_load_count": 1,
            "global_models_fitted": 3,
            "training_rows_by_fold": {audit["fold"]: audit["training_rows"] for audit in preprocessing_audits},
            "fit_seconds_by_fold": fit_seconds,
            "recursive_batch_predictions": 90,
            "rows_per_recursive_batch": 300,
            "runtime_seconds_before_report_write": round(time.perf_counter() - started, 3),
        },
        "scope": {
            "ridge_only": True,
            "hyperparameter_tuning": False,
            "price_scenario": False,
            "final_test_evaluated": False,
            "tree_models_trained": False,
            "inventory_logic_implemented": False,
        },
    }
    summary = json_safe(summary)
    report_path.write_text(render_stage8_report(summary), encoding="utf-8")
    write_json(summary_path, summary)
    print("\nStage 8 Ridge evaluation complete")
    print(f"Prediction rows: {len(predictions):,}")
    print(f"Metric/comparison rows: {len(comparison):,}")
    print(summary["results"]["balanced_conclusion"])
    print(f"Latest evaluated date: {summary['final_test_lock']['latest_evaluated_target_date']}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-data", type=Path, default=DEFAULT_FEATURE_DATA)
    parser.add_argument("--ridge-config", type=Path, default=DEFAULT_RIDGE_CONFIG)
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--stage7-metrics", type=Path, default=DEFAULT_STAGE7_METRICS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_stage8(
        feature_data_path=args.feature_data,
        ridge_config_path=args.ridge_config,
        feature_manifest_path=args.feature_manifest,
        validation_manifest_path=args.validation_manifest,
        stage7_metrics_path=args.stage7_metrics,
        predictions_path=args.predictions,
        metrics_path=args.metrics,
        summary_path=args.summary,
        report_path=args.report,
        figure_dir=args.figure_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
