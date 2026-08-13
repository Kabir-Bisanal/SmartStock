"""Build SmartStock Stage 10 deployment forecasts and inventory recommendations."""

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
    from smartstock.inventory.engine import recommend_inventory
    from smartstock.inventory.policy import load_policy
    from smartstock.inventory.stage10_figures import create_stage10_figures
    from smartstock.inventory.stage10_report import render_stage10_report
    from smartstock.inventory.synthetic import generate_synthetic_inventory
    from smartstock.inventory.uncertainty import calibrate_validation_errors
    from smartstock.models.baseline_evaluation import file_sha256, json_safe, write_json
    from smartstock.models.deployment_forecaster import (
        create_deployment_artifact,
        forecast_deployment_demand,
    )
except ModuleNotFoundError:
    from engine import recommend_inventory
    from policy import load_policy
    from stage10_figures import create_stage10_figures
    from stage10_report import render_stage10_report
    from synthetic import generate_synthetic_inventory
    from uncertainty import calibrate_validation_errors
    from baseline_evaluation import file_sha256, json_safe, write_json
    from deployment_forecaster import create_deployment_artifact, forecast_deployment_demand


DEFAULT_FEATURE_DATA = Path("data/processed/smartstock_v1_features.csv")
DEFAULT_VALIDATION_PREDICTIONS = Path(
    "data/processed/baselines/baseline_validation_predictions.csv"
)
DEFAULT_POLICY = Path("config/v1_inventory_policy.json")
DEFAULT_FINAL_CONFIG = Path("config/v1_final_model.json")
DEFAULT_EVALUATION_ARTIFACT = Path("models/smartstock_v1_forecaster.joblib")
DEFAULT_DEPLOYMENT_ARTIFACT = Path("models/smartstock_v1_deployment_forecaster.joblib")
DEFAULT_FORECAST = Path("data/processed/production/smartstock_v1_30day_forecast.csv")
DEFAULT_CALIBRATION = Path("data/processed/inventory/smartstock_v1_error_calibration.csv")
DEFAULT_SNAPSHOT = Path("data/simulated/smartstock_v1_inventory_snapshot.csv")
DEFAULT_RECOMMENDATIONS = Path(
    "data/processed/inventory/smartstock_v1_inventory_recommendations.csv"
)
DEFAULT_SUMMARY = Path("reports/stage10_summary.json")
DEFAULT_REPORT = Path("reports/stage10_inventory_optimization_report.md")
DEFAULT_FIGURES = Path("reports/figures/stage10")

FROZEN_STAGE9 = {
    "evaluation_forecaster": DEFAULT_EVALUATION_ARTIFACT,
    "final_model_config": DEFAULT_FINAL_CONFIG,
    "test_receipt": Path("reports/stage9_test_evaluation_receipt.json"),
    "test_metrics": Path("reports/stage9_model_metrics.csv"),
    "stage9_summary": Path("reports/stage9_summary.json"),
    "stage9_report": Path("reports/stage9_final_model_report.md"),
}


def _group_results(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for value, group in frame.groupby(column, observed=True, sort=True):
        records.append(
            {
                column: str(value),
                "series": len(group),
                "requiring_reorder": int(group["recommended_order_qty"].gt(0).sum()),
                "recommended_units": int(group["recommended_order_qty"].sum()),
                "cost_optimized_units": int(group["cost_optimized_order_qty"].sum()),
                "expected_shortage_without_order": float(
                    group["expected_shortage_without_order"].sum()
                ),
                "expected_shortage_with_recommendation": float(
                    group["expected_shortage_with_recommendation"].sum()
                ),
                "estimated_cost_without_order": float(
                    group["estimated_cost_without_order"].sum()
                ),
                "estimated_cost_with_recommendation": float(
                    group["estimated_cost_with_recommendation"].sum()
                ),
                "estimated_cost_savings": float(group["estimated_cost_savings"].sum()),
            }
        )
    return records


def _recommendation_totals(recommendations: pd.DataFrame) -> dict[str, Any]:
    statuses = ["STOCKOUT", "CRITICAL", "REORDER NOW", "LOW", "HEALTHY", "OVERSTOCK"]
    priorities = ["URGENT", "HIGH", "MEDIUM", "LOW", "NONE"]
    status_counts = recommendations["stock_status"].value_counts().reindex(statuses, fill_value=0)
    priority_counts = recommendations["priority_label"].value_counts().reindex(priorities, fill_value=0)
    shortage_before = float(recommendations["expected_shortage_without_order"].sum())
    shortage_after = float(recommendations["expected_shortage_with_recommendation"].sum())
    return {
        "series": len(recommendations),
        "stock_status_counts": {name: int(value) for name, value in status_counts.items()},
        "priority_counts": {name: int(value) for name, value in priority_counts.items()},
        "stockout_items": int(status_counts["STOCKOUT"]),
        "critical_items": int(status_counts["CRITICAL"]),
        "requiring_reorder": int(recommendations["recommended_order_qty"].gt(0).sum()),
        "healthy_items": int(status_counts["HEALTHY"]),
        "overstock_items": int(status_counts["OVERSTOCK"]),
        "cost_optimized_reorders": int(
            recommendations["cost_optimized_order_qty"].gt(0).sum()
        ),
        "total_recommended_units": int(recommendations["recommended_order_qty"].sum()),
        "total_cost_optimized_units": int(
            recommendations["cost_optimized_order_qty"].sum()
        ),
        "expected_shortage_without_order": shortage_before,
        "expected_shortage_with_recommendation": shortage_after,
        "expected_shortage_reduction": shortage_before - shortage_after,
        "expected_excess_without_order": float(
            recommendations["expected_excess_without_order"].sum()
        ),
        "expected_excess_with_recommendation": float(
            recommendations["expected_excess_with_recommendation"].sum()
        ),
        "estimated_cost_without_order": float(
            recommendations["estimated_cost_without_order"].sum()
        ),
        "estimated_cost_with_recommendation": float(
            recommendations["estimated_cost_with_recommendation"].sum()
        ),
        "estimated_cost_savings": float(recommendations["estimated_cost_savings"].sum()),
    }


def _integrity_checks(
    forecasts: pd.DataFrame,
    calibration: pd.DataFrame,
    snapshot: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> dict[str, bool]:
    keys = ["store_id", "item_id"]
    checks = {
        "forecast_has_9000_rows": len(forecasts) == 9_000,
        "forecast_has_300_series": forecasts.groupby(keys, observed=True).ngroups == 300,
        "forecast_has_30_days": forecasts["target_date"].nunique() == 30,
        "forecast_keys_unique": not forecasts.duplicated(["target_date", *keys]).any(),
        "forecast_nonnegative": forecasts["forecast"].ge(0).all(),
        "calibration_has_300_unique_series": len(calibration) == 300
        and not calibration.duplicated(keys).any(),
        "calibration_is_validation_only": calibration["calibration_source"].eq(
            "validation_only_mean_28"
        ).all(),
        "snapshot_has_300_unique_series": len(snapshot) == 300
        and not snapshot.duplicated(keys).any(),
        "snapshot_is_explicitly_synthetic": snapshot["inventory_source"].eq(
            "synthetic_demo"
        ).all(),
        "recommendations_have_300_unique_series": len(recommendations) == 300
        and not recommendations.duplicated(keys).any(),
        "inventory_position_formula": np.allclose(
            recommendations["inventory_position"],
            recommendations["on_hand"]
            + recommendations["on_order"]
            - recommendations["backorders"],
        ),
        "reorder_point_formula": np.allclose(
            recommendations["reorder_point"],
            recommendations["lead_time_demand"] + recommendations["safety_stock"],
        ),
        "reorder_raw_formula": np.allclose(
            recommendations["recommended_order_qty_raw"],
            np.maximum(
                recommendations["target_stock_level"]
                - recommendations["inventory_position"],
                0,
            ),
        ),
        "rounded_orders_cover_raw": recommendations["recommended_order_qty"].ge(
            recommendations["recommended_order_qty_raw"] - 1e-9
        ).all(),
        "stockout_risk_bounded": recommendations["stockout_risk"].between(0, 1).all(),
        "nonnegative_orders": recommendations["recommended_order_qty"].ge(0).all()
        and recommendations["cost_optimized_order_qty"].ge(0).all(),
        "optimizer_never_worse_than_no_order": recommendations[
            "estimated_cost_with_recommendation"
        ].le(recommendations["estimated_cost_without_order"] + 1e-9).all(),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Stage 10 integrity checks failed: {', '.join(failed)}")
    return {name: bool(value) for name, value in checks.items()}


def run_stage10(
    feature_data_path: Path = DEFAULT_FEATURE_DATA,
    validation_predictions_path: Path = DEFAULT_VALIDATION_PREDICTIONS,
    policy_path: Path = DEFAULT_POLICY,
    final_config_path: Path = DEFAULT_FINAL_CONFIG,
    evaluation_artifact_path: Path = DEFAULT_EVALUATION_ARTIFACT,
    deployment_artifact_path: Path = DEFAULT_DEPLOYMENT_ARTIFACT,
    forecast_path: Path = DEFAULT_FORECAST,
    calibration_path: Path = DEFAULT_CALIBRATION,
    snapshot_path: Path = DEFAULT_SNAPSHOT,
    recommendations_path: Path = DEFAULT_RECOMMENDATIONS,
    summary_path: Path = DEFAULT_SUMMARY,
    report_path: Path = DEFAULT_REPORT,
    figure_dir: Path = DEFAULT_FIGURES,
) -> dict[str, Any]:
    """Execute the complete Stage 10 recommendation pipeline without Stage 9 mutation."""

    started = time.perf_counter()
    frozen_before = {name: file_sha256(path) for name, path in FROZEN_STAGE9.items()}
    receipt = json.loads(FROZEN_STAGE9["test_receipt"].read_text(encoding="utf-8"))
    if receipt["evaluation_count"] != 1:
        raise RuntimeError("Stage 9 final-test receipt is not frozen at one evaluation.")
    if frozen_before["evaluation_forecaster"] != receipt["model_artifact"]["sha256"]:
        raise RuntimeError("Stage 9 evaluation artifact hash does not match its receipt.")
    if frozen_before["final_model_config"] != receipt["frozen_config_sha256"]:
        raise RuntimeError("Stage 9 final config hash does not match its receipt.")
    policy = load_policy(policy_path)

    print("Creating a separate deployment refit through 2016-05-22 ...")
    deployment, deployment_audit = create_deployment_artifact(
        feature_data_path,
        final_config_path,
        deployment_artifact_path,
        trained_through="2016-05-22",
    )
    forecasts = forecast_deployment_demand(deployment, 30)
    forecast_path.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(forecast_path, index=False, date_format="%Y-%m-%d")

    print("Calibrating forecast uncertainty from validation-only mean-28 residuals ...")
    calibration, calibration_audit = calibrate_validation_errors(
        validation_predictions_path,
        minimum_series_observations=int(
            policy["uncertainty"]["minimum_item_store_observations"]
        ),
    )
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(calibration_path, index=False)

    print("Generating the deterministic synthetic/demo inventory snapshot ...")
    snapshot = generate_synthetic_inventory(forecasts, calibration, policy)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(snapshot_path, index=False)

    print("Calculating service-level and cost-aware recommendations ...")
    recommendations = recommend_inventory(forecasts, snapshot, calibration, policy)
    recommendations_path.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(recommendations_path, index=False)

    checks = _integrity_checks(forecasts, calibration, snapshot, recommendations)
    frozen_after = {name: file_sha256(path) for name, path in FROZEN_STAGE9.items()}
    if frozen_before != frozen_after:
        raise RuntimeError("A frozen Stage 9 artifact changed during Stage 10.")
    totals = _recommendation_totals(recommendations)
    print("Generating Stage 10 business figures ...")
    figures = create_stage10_figures(recommendations, figure_dir)

    summary: dict[str, Any] = {
        "stage": "Stage 10 - Inventory Optimization & Reorder Decision Engine",
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status": "complete",
        "data_disclosure": policy["data_disclosure"],
        "policy": {
            "path": policy_path.as_posix(),
            "sha256": file_sha256(policy_path),
            "service_level": policy["service_level"],
            "lead_time_days": policy["lead_time_days"],
            "review_period_days": policy["review_period_days"],
            "low_stock_days": policy["low_stock_days"],
            "overstock_multiplier": policy["overstock_multiplier"],
            "holding_cost_per_unit_per_day": policy[
                "holding_cost_per_unit_per_day"
            ],
            "stockout_cost_per_unit": policy["stockout_cost_per_unit"],
            "fixed_order_cost": policy["fixed_order_cost"],
            "scenario_count": policy["scenario_count"],
            "random_seed": policy["random_seed"],
            "all_business_values_are_demo_defaults": True,
        },
        "stage9_integrity": {
            "hashes_before": frozen_before,
            "hashes_after": frozen_after,
            "unchanged": frozen_before == frozen_after,
            "evaluation_artifact_trained_through": "2016-04-22",
            "evaluation_artifact_overwritten": False,
            "final_test_metrics_recomputed": False,
            "final_test_used_for_calibration": False,
        },
        "deployment_forecast": {
            **deployment_audit,
            "trained_through": deployment["trained_through"],
            "model_family": deployment["model_family"],
            "rows": len(forecasts),
            "series": int(forecasts.groupby(["store_id", "item_id"], observed=True).ngroups),
            "date_min": forecasts["target_date"].min().date().isoformat(),
            "date_max": forecasts["target_date"].max().date().isoformat(),
            "forecast_1d_total": float(
                forecasts.loc[forecasts["horizon_day"].eq(1), "forecast"].sum()
            ),
            "forecast_7d_total": float(
                forecasts.loc[forecasts["horizon_day"].le(7), "forecast"].sum()
            ),
            "forecast_30d_total": float(forecasts["forecast"].sum()),
            "output_path": forecast_path.as_posix(),
            "output_sha256": file_sha256(forecast_path),
            "output_size_bytes": forecast_path.stat().st_size,
            "future_event_snap_price_features_invented": False,
        },
        "uncertainty_calibration": {
            **calibration_audit,
            "output_path": calibration_path.as_posix(),
            "output_sha256": file_sha256(calibration_path),
            "mean_sigma_daily": float(calibration["sigma_daily"].mean()),
            "median_sigma_daily": float(calibration["sigma_daily"].median()),
            "mean_residual_bias_actual_minus_forecast": float(
                calibration["residual_mean_used"].mean()
            ),
        },
        "synthetic_inventory": {
            "rows": len(snapshot),
            "source": "synthetic_demo_not_walmart",
            "random_seed": policy["random_seed"],
            "profile_counts": {
                name: int(value)
                for name, value in snapshot["synthetic_profile_plan"].value_counts().sort_index().items()
            },
            "lead_time_counts": {
                str(name): int(value)
                for name, value in snapshot["lead_time_days"].value_counts().sort_index().items()
            },
            "service_level_counts": {
                str(name): int(value)
                for name, value in snapshot["service_level"].value_counts().sort_index().items()
            },
            "output_path": snapshot_path.as_posix(),
            "output_sha256": file_sha256(snapshot_path),
        },
        "recommendation_results": totals,
        "store_results": _group_results(recommendations, "store_id"),
        "department_results": _group_results(recommendations, "dept_id"),
        "priority_results": _group_results(recommendations, "priority_label"),
        "recommendation_output": {
            "path": recommendations_path.as_posix(),
            "sha256": file_sha256(recommendations_path),
            "size_bytes": recommendations_path.stat().st_size,
            "rows": len(recommendations),
            "columns": len(recommendations.columns),
        },
        "integrity_checks": checks,
        "figures": figures,
        "performance": {
            "runtime_seconds_before_report_write": round(time.perf_counter() - started, 3),
            "demand_scenarios_per_series": int(policy["scenario_count"]),
            "series_optimized": len(recommendations),
            "maximum_candidate_points": int(
                policy["cost_optimization"]["maximum_candidate_points"]
            ),
        },
        "scope": {
            "historical_policy_simulation": "deferred_optional_component",
            "postgresql": False,
            "streamlit": False,
            "docker": False,
            "deployment": False,
        },
    }
    summary = json_safe(summary)
    report_path.write_text(render_stage10_report(summary), encoding="utf-8")
    write_json(summary_path, summary)
    print("\nStage 10 inventory recommendations complete")
    print(f"Recommendation rows: {len(recommendations):,}")
    print(f"Service-level recommended units: {totals['total_recommended_units']:,}")
    print(f"Estimated scenario cost savings: {totals['estimated_cost_savings']:,.2f}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_stage10(policy_path=args.policy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
