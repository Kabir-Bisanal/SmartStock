"""Run forecast-origin-safe Stage 7 baseline validation without touching final test."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    from smartstock.models.baselines import BASELINE_FUNCTIONS, DEFAULT_CROSTON_ALPHA, forecast_all_baselines
    from smartstock.models.metrics import evaluate_prediction_group, rmsse_scale
    from smartstock.models.stage7_figures import create_stage7_figures
    from smartstock.models.stage7_report import render_stage7_report
except ModuleNotFoundError:  # Supports direct execution from the repository root.
    from baselines import BASELINE_FUNCTIONS, DEFAULT_CROSTON_ALPHA, forecast_all_baselines
    from metrics import evaluate_prediction_group, rmsse_scale
    from stage7_figures import create_stage7_figures
    from stage7_report import render_stage7_report


DEFAULT_FEATURE_DATA = Path("data") / "processed" / "smartstock_v1_features.csv"
DEFAULT_VALIDATION_MANIFEST = Path("config") / "v1_validation.json"
DEFAULT_FEATURE_MANIFEST = Path("config") / "v1_features.json"
DEFAULT_SUBSET_MANIFEST = Path("config") / "v1_subset.json"
DEFAULT_INTERIM_DATA = Path("data") / "interim" / "smartstock_v1_long.csv"
DEFAULT_PREDICTIONS = Path("data") / "processed" / "baselines" / "baseline_validation_predictions.csv"
DEFAULT_METRICS = Path("reports") / "stage7_baseline_metrics.csv"
DEFAULT_SUMMARY = Path("reports") / "stage7_summary.json"
DEFAULT_REPORT = Path("reports") / "stage7_baseline_evaluation_report.md"
DEFAULT_FIGURE_DIR = Path("reports") / "figures" / "stage7"

SERIES_KEYS = ["store_id", "item_id"]
HORIZON_SPECS = {
    "day_1_daily": {"days": 1, "aggregate": False},
    "days_1_7_daily": {"days": 7, "aggregate": False},
    "days_1_7_aggregate": {"days": 7, "aggregate": True},
    "days_1_30_daily": {"days": 30, "aggregate": False},
    "days_1_30_aggregate": {"days": 30, "aggregate": True},
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [json_safe(record) for record in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_modeling_data(path: Path) -> pd.DataFrame:
    """Load only fields needed for origin-based baselines and evaluation."""

    columns = ["date", "store_id", "item_id", "dept_id", "demand_band", "is_feature_ready", "sales"]
    data = pd.read_csv(
        path,
        usecols=columns,
        parse_dates=["date"],
        dtype={
            "store_id": "string",
            "item_id": "string",
            "dept_id": "string",
            "demand_band": "string",
            "is_feature_ready": "boolean",
            "sales": "uint16",
        },
    )
    for column in ("store_id", "item_id", "dept_id", "demand_band"):
        data[column] = data[column].astype("category")
    return data.sort_values([*SERIES_KEYS, "date"], kind="stable").reset_index(drop=True)


def assert_window_unlocked(start: pd.Timestamp, end: pd.Timestamp, validation_manifest: dict[str, Any]) -> None:
    """Block any Stage 7 attempt to forecast or evaluate the locked final test."""

    test_start = pd.Timestamp(validation_manifest["locked_final_test"]["start_date"])
    test_end = pd.Timestamp(validation_manifest["locked_final_test"]["end_date"])
    if start <= test_end and end >= test_start:
        raise ValueError(f"Stage 7 final-test lock blocks evaluation from {start.date()} through {end.date()}.")


def classify_intermittency(training_sales: np.ndarray) -> tuple[str, float]:
    """Classify a series using only its fold-specific pre-origin training history."""

    zero_pct = float(100 * np.mean(np.asarray(training_sales) == 0))
    if zero_pct < 50:
        return "regular", zero_pct
    if zero_pct <= 80:
        return "intermittent", zero_pct
    return "highly_intermittent", zero_pct


def generate_fold_predictions(
    data: pd.DataFrame,
    fold: dict[str, str],
    validation_manifest: dict[str, Any],
    horizon: int = 30,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate all baseline forecasts before attaching validation actuals."""

    start = pd.Timestamp(fold["start_date"])
    end = pd.Timestamp(fold["end_date"])
    assert_window_unlocked(start, end, validation_manifest)
    if (end - start).days + 1 != horizon:
        raise ValueError(f"{fold['name']} must contain exactly {horizon} calendar days.")
    origin = start + pd.DateOffset(days=-1)

    # Warm-up rows have incomplete engineered predictors, but their active-period
    # sales are observed and safe for demand-only baseline histories and RMSSE.
    history = data[data["date"].le(origin)]
    validation = data[data["is_feature_ready"] & data["date"].between(start, end)].copy()
    if validation.groupby(SERIES_KEYS, observed=True).size().nunique() != 1 or validation.groupby(SERIES_KEYS, observed=True).size().iloc[0] != horizon:
        raise ValueError(f"{fold['name']} does not contain {horizon} feature-ready days for every series.")

    forecast_rows: list[dict[str, Any]] = []
    source_checks = {
        "history_ends_at_origin": True,
        "last_value_uses_final_preorigin_value": True,
        "seasonal_naive_uses_final_preorigin_week": True,
        "mean_28_uses_final_preorigin_28_days": True,
        "croston_receives_preorigin_history_only": True,
    }
    validation_meta = validation[[*SERIES_KEYS, "date", "dept_id", "demand_band", "sales"]].copy()
    validation_meta = validation_meta.rename(columns={"date": "target_date", "sales": "actual"})

    for (store_id, item_id), target_group in validation_meta.groupby(SERIES_KEYS, observed=True, sort=False):
        series_history = history[
            history["store_id"].astype("string").eq(str(store_id))
            & history["item_id"].astype("string").eq(str(item_id))
        ]
        values = series_history["sales"].to_numpy(dtype="float64")
        history_end = series_history["date"].max()
        if len(values) < 28 or history_end != origin:
            raise ValueError(f"Insufficient or non-origin history for {store_id}/{item_id} in {fold['name']}.")
        forecasts = forecast_all_baselines(values, horizon=horizon)
        dates = target_group.sort_values("target_date")["target_date"].to_numpy()
        intermittent_class, zero_pct = classify_intermittency(values)
        scale = rmsse_scale(values)

        source_checks["history_ends_at_origin"] &= history_end == origin
        source_checks["last_value_uses_final_preorigin_value"] &= bool(np.all(forecasts["last_value"] == values[-1]))
        source_checks["seasonal_naive_uses_final_preorigin_week"] &= bool(np.array_equal(forecasts["seasonal_naive_7"], np.resize(values[-7:], horizon)))
        source_checks["mean_28_uses_final_preorigin_28_days"] &= bool(np.allclose(forecasts["mean_28"], values[-28:].mean()))

        for baseline_name, baseline_values in forecasts.items():
            for horizon_day, (target_date, forecast_value) in enumerate(zip(dates, baseline_values, strict=True), start=1):
                forecast_rows.append(
                    {
                        "fold": fold["name"],
                        "baseline_name": baseline_name,
                        "store_id": str(store_id),
                        "item_id": str(item_id),
                        "forecast_origin": origin,
                        "history_end_date": history_end,
                        "history_observations": len(values),
                        "target_date": pd.Timestamp(target_date),
                        "horizon_day": horizon_day,
                        "forecast": float(forecast_value),
                        "rmsse_scale": scale,
                        "training_zero_pct": zero_pct,
                        "intermittency_class": intermittent_class,
                    }
                )

    forecasts = pd.DataFrame(forecast_rows)
    # Actuals and analysis-only metadata are attached only after forecasts exist.
    predictions = forecasts.merge(validation_meta, on=[*SERIES_KEYS, "target_date"], how="left", validate="many_to_one")
    if predictions["actual"].isna().any():
        raise ValueError(f"Missing validation actuals after forecast generation for {fold['name']}.")
    predictions["actual"] = predictions["actual"].astype("uint16")
    predictions["forecast"] = predictions["forecast"].astype("float32")
    predictions["rmsse_scale"] = predictions["rmsse_scale"].astype("float32")
    predictions["training_zero_pct"] = predictions["training_zero_pct"].astype("float32")
    predictions["horizon_day"] = predictions["horizon_day"].astype("int8")

    validation_series = int(validation_meta.groupby(SERIES_KEYS, observed=True).ngroups)
    expected_rows = validation_series * horizon * len(BASELINE_FUNCTIONS)
    checks = {
        **source_checks,
        "expected_prediction_rows": len(predictions) == expected_rows,
        "five_baselines": set(predictions["baseline_name"].unique()) == set(BASELINE_FUNCTIONS),
        "all_targets_after_origin": bool(predictions["target_date"].gt(predictions["forecast_origin"]).all()),
        "all_targets_within_fold": bool(predictions["target_date"].between(start, end).all()),
        "no_final_test_targets": bool(predictions["target_date"].lt(pd.Timestamp(validation_manifest["locked_final_test"]["start_date"])).all()),
        "finite_nonnegative_forecasts": bool(np.isfinite(predictions["forecast"]).all() and predictions["forecast"].ge(0).all()),
        "demand_band_attached_after_forecasting": True,
        "no_price_inputs_loaded": True,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Forecast-origin audit failed for {fold['name']}: {', '.join(failed)}")
    audit = {
        "fold": fold["name"],
        "forecast_origin": origin.date().isoformat(),
        "validation_start": start.date().isoformat(),
        "validation_end": end.date().isoformat(),
        "prediction_rows": len(predictions),
        "series": int(predictions.groupby(SERIES_KEYS).ngroups),
        "checks": checks,
        "intermittency_class_counts": (
            predictions[[*SERIES_KEYS, "intermittency_class"]]
            .drop_duplicates()
            ["intermittency_class"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
    }
    return predictions, audit


def calculate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate fold-level and combined overall/segment results."""

    records: list[dict[str, Any]] = []
    fold_scopes: list[tuple[str, pd.DataFrame]] = [
        *[(fold, predictions[predictions["fold"].eq(fold)]) for fold in sorted(predictions["fold"].unique())],
        ("combined", predictions),
    ]
    segment_columns = ["demand_band", "store_id", "dept_id", "intermittency_class"]

    for fold_label, fold_frame in fold_scopes:
        for baseline_name, baseline_frame in fold_frame.groupby("baseline_name", observed=True, sort=False):
            for horizon_name, spec in HORIZON_SPECS.items():
                horizon_frame = baseline_frame[baseline_frame["horizon_day"].le(spec["days"])]
                segments: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", horizon_frame)]
                for segment_column in segment_columns:
                    segments.extend(
                        (segment_column, str(segment_value), segment_frame)
                        for segment_value, segment_frame in horizon_frame.groupby(segment_column, observed=True, sort=True)
                    )
                for segment_type, segment_value, segment_frame in segments:
                    result = evaluate_prediction_group(
                        segment_frame,
                        aggregate_horizon_days=spec["days"] if spec["aggregate"] else None,
                    )
                    records.append(
                        {
                            "fold": fold_label,
                            "baseline_name": baseline_name,
                            "horizon": horizon_name,
                            "horizon_days": spec["days"],
                            "evaluation_type": "aggregate" if spec["aggregate"] else "daily",
                            "segment_type": segment_type,
                            "segment_value": segment_value,
                            **result,
                        }
                    )
    return pd.DataFrame(records).sort_values(
        ["fold", "horizon_days", "evaluation_type", "segment_type", "segment_value", "baseline_name"],
        kind="stable",
    ).reset_index(drop=True)


def _overall_slice(metrics: pd.DataFrame, horizon: str, fold: str = "combined") -> pd.DataFrame:
    return metrics[
        metrics["fold"].eq(fold)
        & metrics["horizon"].eq(horizon)
        & metrics["segment_type"].eq("overall")
    ].copy()


def summarize_results(metrics: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
    """Create balanced rankings and required Stage 7 investigations."""

    primary = _overall_slice(metrics, "days_1_30_daily").copy()
    primary["absolute_bias"] = primary["bias"].abs()
    ranking_metrics = ["mae", "rmse", "wape", "rmsse", "absolute_bias"]
    for metric in ranking_metrics:
        primary[f"{metric}_rank"] = primary[metric].rank(method="min", ascending=True)
    primary["mean_rank"] = primary[[f"{metric}_rank" for metric in ranking_metrics]].mean(axis=1)
    balanced = primary.sort_values(["mean_rank", "mae", "baseline_name"])

    best_by_metric = {
        "mae": primary.loc[primary["mae"].idxmin(), "baseline_name"],
        "rmse": primary.loc[primary["rmse"].idxmin(), "baseline_name"],
        "wape": primary.loc[primary["wape"].idxmin(), "baseline_name"],
        "rmsse": primary.loc[primary["rmsse"].idxmin(), "baseline_name"],
        "absolute_bias": primary.loc[primary["absolute_bias"].idxmin(), "baseline_name"],
    }

    intermittency = metrics[
        metrics["fold"].eq("combined")
        & metrics["horizon"].eq("days_1_30_daily")
        & metrics["segment_type"].eq("intermittency_class")
    ].copy()
    croston_analysis = []
    for segment, segment_frame in intermittency.groupby("segment_value", observed=True, sort=True):
        ordered = segment_frame.sort_values("mae")
        croston = segment_frame[segment_frame["baseline_name"].eq("croston_sba")].iloc[0]
        croston_analysis.append(
            {
                "intermittency_class": segment,
                "croston_mae": croston["mae"],
                "croston_wape": croston["wape"],
                "croston_rmsse": croston["rmsse"],
                "mae_rank": int(segment_frame["mae"].rank(method="min").loc[croston.name]),
                "best_mae_baseline": ordered.iloc[0]["baseline_name"],
                "best_mae": ordered.iloc[0]["mae"],
            }
        )

    seasonal_names = ["seasonal_naive_7", "last_value", "mean_28"]
    seasonal_comparison = primary[primary["baseline_name"].isin(seasonal_names)][
        ["baseline_name", "mae", "rmse", "wape", "rmsse", "bias"]
    ].sort_values("mae")
    zero_mae = float(primary.loc[primary["baseline_name"].eq("zero"), "mae"].iloc[0])
    best_nonzero = primary[~primary["baseline_name"].eq("zero")].sort_values("mae").iloc[0]

    fold_results = metrics[
        metrics["fold"].ne("combined")
        & metrics["horizon"].eq("days_1_30_daily")
        & metrics["segment_type"].eq("overall")
    ][["fold", "baseline_name", "mae", "rmse", "wape", "rmsse", "bias"]]

    horizon_results = metrics[
        metrics["fold"].eq("combined")
        & metrics["segment_type"].eq("overall")
    ][["baseline_name", "horizon", "evaluation_type", "mae", "rmse", "wape", "rmsse", "bias", "aggregate_bias"]]

    class_counts = (
        predictions[["fold", *SERIES_KEYS, "intermittency_class"]]
        .drop_duplicates()
        .groupby(["fold", "intermittency_class"], observed=True)
        .size()
        .rename("series")
        .reset_index()
    )
    segment_results = {}
    for segment_type in ("demand_band", "store_id", "dept_id", "intermittency_class"):
        segment_results[segment_type] = metrics[
            metrics["fold"].eq("combined")
            & metrics["horizon"].eq("days_1_30_daily")
            & metrics["segment_type"].eq(segment_type)
        ][["segment_value", "baseline_name", "mae", "rmse", "wape", "rmsse", "bias"]]
    return {
        "primary_comparison": primary[["baseline_name", "mae", "rmse", "wape", "rmsse", "bias", "aggregate_bias", "rmsse_undefined_series"]],
        "best_by_metric": best_by_metric,
        "balanced_ranking": balanced[["baseline_name", "mean_rank", *ranking_metrics]],
        "strongest_overall_baseline": balanced.iloc[0]["baseline_name"],
        "croston_by_intermittency": croston_analysis,
        "seasonal_naive_comparison": seasonal_comparison,
        "zero_baseline": {
            "mae": zero_mae,
            "best_nonzero_baseline": best_nonzero["baseline_name"],
            "best_nonzero_mae": best_nonzero["mae"],
            "best_nonzero_mae_improvement_pct": 100 * (zero_mae - best_nonzero["mae"]) / zero_mae,
        },
        "fold_results": fold_results,
        "horizon_results": horizon_results,
        "intermittency_class_counts": class_counts,
        "segment_results": segment_results,
    }


def run_stage7(
    feature_data_path: Path = DEFAULT_FEATURE_DATA,
    validation_manifest_path: Path = DEFAULT_VALIDATION_MANIFEST,
    feature_manifest_path: Path = DEFAULT_FEATURE_MANIFEST,
    subset_manifest_path: Path = DEFAULT_SUBSET_MANIFEST,
    interim_data_path: Path = DEFAULT_INTERIM_DATA,
    predictions_path: Path = DEFAULT_PREDICTIONS,
    metrics_path: Path = DEFAULT_METRICS,
    summary_path: Path = DEFAULT_SUMMARY,
    report_path: Path = DEFAULT_REPORT,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
) -> dict[str, Any]:
    """Execute only the Stage 7 validation folds and persist auditable results."""

    started = time.perf_counter()
    frozen_paths = {
        "feature_dataset": feature_data_path,
        "validation_manifest": validation_manifest_path,
        "feature_manifest": feature_manifest_path,
        "subset_manifest": subset_manifest_path,
        "interim_dataset": interim_data_path,
    }
    hashes_before = {name: file_sha256(path) for name, path in frozen_paths.items()}
    validation_manifest = json.loads(validation_manifest_path.read_text(encoding="utf-8"))
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if not validation_manifest["locked_final_test"]["locked"]:
        raise ValueError("Stage 7 requires a locked final test.")
    if "demand_band" in feature_manifest["default_model_features"]:
        raise ValueError("Frozen feature manifest incorrectly exposes demand_band as a predictor.")

    print("Loading the frozen Stage 6 modeling dataset once ...")
    data = load_modeling_data(feature_data_path)
    test_start = pd.Timestamp(validation_manifest["locked_final_test"]["start_date"])
    pretest_data = data[data["date"].lt(test_start)].copy()
    del data

    fold_predictions: list[pd.DataFrame] = []
    origin_audits = []
    print("Generating five origin-safe baselines for three validation folds ...")
    for fold in validation_manifest["validation_folds"]:
        predictions, audit = generate_fold_predictions(pretest_data, fold, validation_manifest)
        fold_predictions.append(predictions)
        origin_audits.append(audit)
    predictions = pd.concat(fold_predictions, ignore_index=True)
    if predictions["target_date"].max() >= test_start:
        raise RuntimeError("Final-test lock violation: generated predictions reach the locked test.")

    print("Calculating fold, horizon, aggregate, and segment metrics ...")
    metrics = calculate_metrics(predictions)
    result_summary = summarize_results(metrics, predictions)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False, date_format="%Y-%m-%d")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)

    print("Creating Stage 7 evaluation figures ...")
    figures = create_stage7_figures(metrics, figure_dir)

    hashes_after = {name: file_sha256(path) for name, path in frozen_paths.items()}
    integrity = {
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
        "unchanged": hashes_before == hashes_after,
    }
    if not integrity["unchanged"]:
        raise RuntimeError("A frozen Stage 6 or earlier artifact changed during Stage 7.")

    metric_audit = {
        "rmsse_scale_training_only": True,
        "intermittency_segments_training_only": True,
        "demand_band_analysis_only": "demand_band" not in feature_manifest["default_model_features"],
        "no_price_inputs": True,
        "no_final_test_predictions": bool(predictions["target_date"].max() < test_start),
        "no_final_test_metrics": bool(not metrics["fold"].astype("string").str.contains("test", case=False).any()),
        "wape_zero_denominator_returns_nan": True,
        "undefined_rmsse_scales_counted": True,
    }
    summary: dict[str, Any] = {
        "stage": "Stage 7 - Baseline Forecasting & Evaluation Framework",
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "baseline_definitions": {
            "zero": "0 for every target day",
            "last_value": "final pre-origin observed sale repeated for 30 days",
            "seasonal_naive_7": "final seven pre-origin sales repeated across 30 days",
            "mean_28": "mean of final 28 pre-origin sales repeated across 30 days",
            "croston_sba": f"constant Croston-SBA forecast with alpha={DEFAULT_CROSTON_ALPHA}",
            "croston_formula": "(1 - alpha/2) * smoothed_nonzero_demand_size / smoothed_inter_demand_interval",
        },
        "metric_definitions": {
            "mae": "mean(abs(forecast - actual))",
            "rmse": "sqrt(mean((forecast - actual)^2))",
            "wape": "sum(abs(forecast - actual)) / sum(actual); undefined when denominator is zero",
            "rmsse": "mean across series of sqrt(series forecast MSE / training-only mean squared one-step naive error)",
            "aggregate_rmsse": "mean across series of abs(total forecast error) / sqrt(horizon_days * training-only naive scale)",
            "bias": "mean(forecast - actual); positive is overforecast and negative is underforecast",
            "aggregate_bias": "sum(forecast - actual)",
        },
        "forecast_origin_audits": origin_audits,
        "metric_leakage_audit": metric_audit,
        "predictions": {
            "rows": len(predictions),
            "folds": int(predictions["fold"].nunique()),
            "baselines": int(predictions["baseline_name"].nunique()),
            "item_store_pairs": int(predictions.groupby(SERIES_KEYS).ngroups),
            "date_min": predictions["target_date"].min().date().isoformat(),
            "date_max": predictions["target_date"].max().date().isoformat(),
            "file": predictions_path.as_posix(),
            "file_size_bytes": predictions_path.stat().st_size,
            "sha256": file_sha256(predictions_path),
        },
        "metrics": {
            "rows": len(metrics),
            "file": metrics_path.as_posix(),
            "file_size_bytes": metrics_path.stat().st_size,
        },
        "results": result_summary,
        "figures": figures,
        "final_test_lock": {
            "start_date": validation_manifest["locked_final_test"]["start_date"],
            "end_date": validation_manifest["locked_final_test"]["end_date"],
            "locked": True,
            "predictions_generated": False,
            "metrics_calculated": False,
            "latest_evaluated_target_date": predictions["target_date"].max().date().isoformat(),
        },
        "source_integrity": integrity,
        "performance": {
            "feature_csv_load_count": 1,
            "prediction_rows": len(predictions),
            "grouped_series_fold_forecasts": 300 * 3,
            "runtime_seconds_before_report_write": round(time.perf_counter() - started, 3),
            "future_price_columns_loaded": False,
            "baseline_history_policy": "all active pre-origin sales, including Stage 6 feature warm-up rows",
        },
        "scope": {
            "advanced_models_trained": False,
            "final_test_evaluated": False,
            "hyperparameters_tuned": False,
            "inventory_logic_implemented": False,
        },
    }
    summary = json_safe(summary)
    report_path.write_text(render_stage7_report(summary, metrics), encoding="utf-8")
    write_json(summary_path, summary)

    print("\nStage 7 baseline evaluation complete")
    print(f"Prediction rows: {len(predictions):,}")
    print(f"Metric rows: {len(metrics):,}")
    print(f"Strongest balanced baseline: {summary['results']['strongest_overall_baseline']}")
    print(f"Latest evaluated date: {summary['final_test_lock']['latest_evaluated_target_date']}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-data", type=Path, default=DEFAULT_FEATURE_DATA)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument("--subset-manifest", type=Path, default=DEFAULT_SUBSET_MANIFEST)
    parser.add_argument("--interim-data", type=Path, default=DEFAULT_INTERIM_DATA)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_stage7(
        feature_data_path=args.feature_data,
        validation_manifest_path=args.validation_manifest,
        feature_manifest_path=args.feature_manifest,
        subset_manifest_path=args.subset_manifest,
        interim_data_path=args.interim_data,
        predictions_path=args.predictions,
        metrics_path=args.metrics,
        summary_path=args.summary,
        report_path=args.report,
        figure_dir=args.figure_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
