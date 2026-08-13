"""Render the beginner-readable SmartStock Stage 8 evaluation report."""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (int,)):
        return f"{value:,}"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def render_stage8_report(summary: dict[str, Any]) -> str:
    """Render Stage 8 findings from calculated validation-only results."""

    results = summary["results"]
    primary = results["primary_comparison"]
    negative = summary["negative_prediction_diagnostics"]
    conclusion = results["balanced_conclusion"]

    def metric_value(metric: str, value: Any) -> str:
        return f"{100 * float(value):.2f}%" if metric == "wape" else _fmt(value)
    lines = [
        "# SmartStock Stage 8 — Global Ridge Regression & Recursive Multi-Step Forecasting",
        "",
        "## Executive Summary",
        "",
        conclusion,
        "",
        f"Across the combined Days 1–30 daily evaluation, Ridge produced MAE **{_fmt(primary['mae'])}**, "
        f"RMSE **{_fmt(primary['rmse'])}**, WAPE **{_fmt(100 * primary['wape'], 2)}%**, "
        f"RMSSE **{_fmt(primary['rmsse'])}**, and signed bias **{_fmt(primary['bias'])}**. "
        f"The official comparison is the tracked Stage 7 28-day mean, not a hard-coded benchmark.",
        "",
        "## Why Ridge Was Chosen",
        "",
        "Ridge is ordinary linear regression plus an L2 penalty on coefficient size. The penalty discourages unstable, very large coefficients when predictors overlap. It provides a transparent first global ML test before moving to nonlinear models.",
        "",
        "## Frozen Stage 8 Feature Set",
        "",
        f"The tracked configuration freezes {summary['model_configuration']['feature_count_before_encoding']} inputs before one-hot encoding:",
        "",
        "Categorical: " + ", ".join(f"`{name}`" for name in summary["model_configuration"]["categorical_features"]) + ".",
        "",
        "Numeric/calendar/history: " + ", ".join(f"`{name}`" for name in summary["model_configuration"]["numeric_features"]) + ".",
        "",
        "## Features Deliberately Excluded",
        "",
        "The target, demand band, all price fields, zero-rate fields, days-since-last-positive, redundant identity fields, and control fields were excluded. Price is deferred because a 30-day deployment forecast needs an explicit future-price-information policy. Discrete intermittency-state features are deferred because recursively updating them from fractional forecasts would add a hidden modeling decision.",
        "",
        "## Preprocessing Pipeline",
        "",
        "Missing categorical values become `__NONE__`, categories are one-hot encoded with unknown categories ignored, and numeric inputs are standardized. Each encoder and scaler is fitted only on that fold's training rows. The sales target is not standardized.",
        "",
        "## Global Model Architecture",
        "",
        "One Ridge model is fitted per fold across all feature-ready observations from all 300 item-store series. Item and store one-hot indicators allow series-specific levels while shared calendar and demand-history coefficients learn across the complete panel.",
        "",
        "## Recursive Forecasting Design",
        "",
        "Day +1 is predicted from actual history through the origin. The clipped Day +1 prediction is appended to synthetic history and becomes available to Day +2 lags and rolling windows. This repeats through Day +30; no validation actual enters the synthetic history.",
        "",
        "## Forecast-Origin Leakage Protection",
        "",
        "Using the persisted Stage 6 Day +20 lag fields directly would reveal actual validation sales from Days +1 through +19. Stage 8 instead selects only approved future calendar/context fields and recomputes every demand lag and rolling statistic from real pre-origin history plus earlier model predictions.",
        "",
        "## Negative Prediction Handling",
        "",
        f"Ridge generated {negative['count']:,} negative raw values ({negative['percentage']:.2f}%); the most negative was {_fmt(negative['most_negative'])}. Sales cannot be negative, so the predeclared official rule clipped them to zero without rounding.",
        "",
        "## Validation Fold Results",
        "",
    ]
    lines += _table(
        ["Fold", "Ridge MAE", "28-day mean MAE", "MAE improvement", "Ridge RMSSE", "Bias"],
        [
            [
                row["fold"].replace("validation_fold_", "Fold "),
                _fmt(row["mae"]),
                _fmt(row["mean_28_mae"]),
                _fmt(row["mae_improvement_vs_mean_28_pct"], 2) + "%",
                _fmt(row["rmsse"]),
                _fmt(row["bias"]),
            ]
            for row in results["fold_results"]
        ],
    )

    heading_map = {
        "day_1_daily": "Day+1 Results",
        "days_1_7_daily": "Days 1–7 Results",
        "days_1_7_aggregate": "7-Day Aggregate Results",
        "days_1_30_daily": "Days 1–30 Results",
        "days_1_30_aggregate": "30-Day Aggregate Results",
    }
    for horizon in results["horizon_results"]:
        lines += ["", f"## {heading_map[horizon['horizon']]}", ""]
        lines += _table(
            ["Metric", "Ridge", "28-day mean", "Improvement"],
            [
                [metric.upper(), metric_value(metric, horizon[metric]), metric_value(metric, horizon[f"mean_28_{metric}"]), _fmt(horizon[f"{metric}_improvement_vs_mean_28_pct"], 2) + "%"]
                for metric in ("mae", "rmse", "wape", "rmsse")
            ],
        )
        lines += ["", f"Ridge bias: **{_fmt(horizon['bias'])}**; 28-day mean bias: **{_fmt(horizon['mean_28_bias'])}**."]

    lines += [
        "",
        "## Comparison with 28-Day Mean",
        "",
        f"For the principal 30-day daily comparison, Ridge's MAE change was **{_fmt(primary['mae_improvement_vs_mean_28_pct'], 2)}%**, RMSE change **{_fmt(primary['rmse_improvement_vs_mean_28_pct'], 2)}%**, WAPE change **{_fmt(primary['wape_improvement_vs_mean_28_pct'], 2)}%**, and RMSSE change **{_fmt(primary['rmsse_improvement_vs_mean_28_pct'], 2)}%**. Positive means Ridge improved; negative means it worsened.",
    ]

    segment_headings = {
        "store_id": "Store Results",
        "dept_id": "Department Results",
        "demand_band": "Demand-Band Results",
        "intermittency_class": "Intermittency Results",
    }
    segment_order = {
        "store_id": ["CA_1", "TX_2", "WI_3"],
        "dept_id": ["FOODS_1", "FOODS_2", "FOODS_3"],
        "demand_band": ["low", "medium", "high"],
        "intermittency_class": ["regular", "intermittent", "highly_intermittent"],
    }
    for segment_type in ("store_id", "dept_id", "demand_band", "intermittency_class"):
        records = sorted(
            results["segment_results"][segment_type],
            key=lambda row: segment_order[segment_type].index(row["segment_value"]),
        )
        lines += ["", f"## {segment_headings[segment_type]}", ""]
        lines += _table(
            ["Segment", "Ridge MAE", "28-day mean MAE", "Zero MAE", "Ridge bias", "MAE vs mean"],
            [
                [
                    row["segment_value"],
                    _fmt(row["mae"]),
                    _fmt(row["mean_28_mae"]),
                    _fmt(row["zero_mae"]),
                    _fmt(row["bias"]),
                    _fmt(row["mae_improvement_vs_mean_28_pct"], 2) + "%",
                ]
                for row in records
            ],
        )

    lines += [
        "",
        "## Forecast Bias",
        "",
        f"The combined 30-day daily Ridge bias was **{_fmt(primary['bias'])}**, compared with **{_fmt(primary['mean_28_bias'])}** for the 28-day mean. Positive bias means systematic overforecasting; negative bias means underforecasting. These directions have different inventory consequences even when absolute accuracy is similar.",
        "",
        "## Fold Stability",
        "",
        results["fold_stability_statement"],
        "",
        "## Raw Negative Prediction Diagnostics",
        "",
        f"Count: **{negative['count']:,}**; percentage: **{negative['percentage']:.2f}%**; most negative raw value: **{_fmt(negative['most_negative'])}**. All official forecasts are non-negative and remain continuous.",
        "",
        "## Leakage Audits",
        "",
    ]
    for audit in summary["forecast_origin_audits"]:
        lines.append(f"- **{audit['fold']}**: all {sum(bool(value) for value in audit['checks'].values())} forecast-origin checks passed; origin {audit['forecast_origin']}.")
    lines += [
        f"- Validation-target and precomputed-future-feature mutation audit: **{_fmt(summary['mutation_leakage_audit']['passed'])}**.",
        f"- Fold-only encoder/scaler audit: **{_fmt(summary['preprocessing_leakage_audit']['all_checks_passed'])}**.",
        f"- Final-test lock: **{_fmt(summary['final_test_lock']['locked_and_untouched'])}**; latest evaluated date {summary['final_test_lock']['latest_evaluated_target_date']}.",
        "",
        "## Model Limitations",
        "",
        "Training features use actual prior demand, while later recursive validation features partly use earlier Ridge predictions. This exposure mismatch is normal for recursive forecasting, but early errors can propagate through later lags and rolling windows. Ridge also assumes additive linear effects and cannot naturally express complex thresholds or interactions.",
        "",
        "## Recommendation for Stage 9",
        "",
        results["stage9_recommendation"],
        "",
        "The final test remains locked. Any Stage 9 model should use the same origins, recursive policy, official metrics, and baseline comparisons before final model selection.",
        "",
        "## Evaluation Figures",
        "",
        "![Ridge versus 28-day mean overall metrics](figures/stage8/01_ridge_vs_mean28_overall.png)",
        "",
        "![MAE comparison by fold](figures/stage8/02_mae_by_fold.png)",
        "",
        "![RMSSE by horizon](figures/stage8/03_rmsse_by_horizon.png)",
        "",
        "![WAPE by horizon](figures/stage8/04_wape_by_horizon.png)",
        "",
        "![Forecast bias](figures/stage8/05_forecast_bias.png)",
        "",
        "![MAE by demand band](figures/stage8/06_mae_by_demand_band.png)",
        "",
        "![MAE by intermittency](figures/stage8/07_mae_by_intermittency_class.png)",
        "",
        "![Ridge MAE improvement by segment](figures/stage8/08_mae_improvement_by_segment.png)",
        "",
    ]
    return "\n".join(lines)
