"""Render the calculated SmartStock Stage 7 baseline report."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import pandas as pd


BASELINE_LABELS = {
    "zero": "Zero",
    "last_value": "Last value",
    "seasonal_naive_7": "7-day seasonal naive",
    "mean_28": "28-day mean",
    "croston_sba": "Croston-SBA",
}


def display(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, bool):
        return "Passed" if value else "FAILED"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.4f}"
    return str(value)


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    def escape(value: Any) -> str:
        return display(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(escape(value) for value in row) + " |" for row in rows)
    return lines


def figure(filename: str, alt: str) -> list[str]:
    return ["", f"![{alt}](figures/stage7/{filename})", ""]


def metric_rows(records: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            BASELINE_LABELS.get(row["baseline_name"], row["baseline_name"]),
            row["mae"],
            row["rmse"],
            row["wape"],
            row["rmsse"],
            row["bias"],
        )
        for row in records
    ]


def render_stage7_report(summary: dict[str, Any], metrics: pd.DataFrame) -> str:
    """Render all required Stage 7 sections using validation results only."""

    results = summary["results"]
    primary = results["primary_comparison"]
    best = results["best_by_metric"]
    lock = summary["final_test_lock"]
    zero = results["zero_baseline"]
    primary_map = {row["baseline_name"]: row for row in primary}
    seasonal_gain_vs_last = 100 * (primary_map["last_value"]["mae"] - primary_map["seasonal_naive_7"]["mae"]) / primary_map["last_value"]["mae"]
    mean_gain_vs_seasonal = 100 * (primary_map["seasonal_naive_7"]["mae"] - primary_map["mean_28"]["mae"]) / primary_map["seasonal_naive_7"]["mae"]

    def horizon_records(name: str) -> list[dict[str, Any]]:
        return [row for row in results["horizon_results"] if row["horizon"] == name]

    lines: list[str] = [
        "# SmartStock Stage 7 — Baseline Forecasting & Evaluation Framework",
        "",
        f"Generated at `{summary['generated_at_utc']}` using the three frozen validation folds. The final test was not forecast or evaluated.",
        "",
        "## Executive Summary",
        "",
        f"Stage 7 generated **{summary['predictions']['rows']:,} daily validation predictions**: 300 item-store series × three folds × 30 days × five baselines. The balanced validation ranking identifies **{BASELINE_LABELS[results['strongest_overall_baseline']]}** as the strongest overall baseline, while individual metrics favor different methods.",
        "",
        f"The best non-zero rule reduced 30-day daily MAE by **{zero['best_nonzero_mae_improvement_pct']:.2f}%** relative to the explicit zero sanity baseline. The latest evaluated target date is `{lock['latest_evaluated_target_date']}`, one day before the locked final test begins.",
        "",
        "## Baseline Definitions",
        "",
        "- **Zero:** forecast zero every day. This is an essential floor because active demand is nearly 50% zero.",
        "- **Last value:** repeat the final observed pre-origin sale for all 30 days.",
        "- **7-day seasonal naive:** repeat only the final seven pre-origin observations, preserving their weekly position.",
        "- **28-day mean:** repeat the continuous mean of the final 28 pre-origin days.",
        f"- **Croston-SBA:** use alpha `{0.1}` and forecast `(1 - alpha/2) × smoothed positive-demand size / smoothed inter-demand interval`. The first positive demand and its 1-based interval initialize the recursion; all-zero history returns zero.",
        "",
        "A future forecasting model must beat simple rules before its extra complexity is useful. Persistence is meaningful because recent demand often carries information; seasonal naive tests the weekday pattern found in Stage 5; and Croston explicitly separates positive demand sizes from the intervals between them.",
        "",
        "## Forecast-Origin Safety",
        "",
        "Each fold makes one 30-day forecast from the day immediately before validation. All five methods receive only active sales observations ending at that origin. Stage 6 warm-up rows are valid here because their sales are observed; `is_feature_ready = false` means engineered ML predictors are incomplete, not that demand history is unknown. Validation targets remain feature-ready. Forecasts are fully created before validation actuals and analysis-only metadata are attached.",
        "",
        "Day +2 cannot use actual Day +1 demand: at the real forecast origin, Day +1 has not happened. Reusing validation sales through precomputed row lags would simulate 30 one-day forecasts rather than one genuine 30-day forecast and would leak future information.",
        "",
    ]
    lines.extend(table(["Fold", "Forecast origin", "Validation", "Prediction rows", "Series"], [
        (audit["fold"], audit["forecast_origin"], f"{audit['validation_start']} through {audit['validation_end']}", audit["prediction_rows"], audit["series"])
        for audit in summary["forecast_origin_audits"]
    ]))
    lines.extend([
        "",
        "## Metric Definitions",
        "",
        "- **MAE:** average absolute unit error. It is easy to explain operationally.",
        "- **RMSE:** square root of average squared error. Squaring makes large misses count more heavily than in MAE.",
        "- **WAPE:** total absolute error divided by total actual demand. It remains usable when individual actual days are zero, unlike ordinary MAPE. A zero group denominator returns undefined rather than an artificial epsilon result.",
        "- **RMSSE:** each series' forecast RMSE divided by its training-only one-step naive scale, then averaged equally across defined series. This permits comparison between low- and high-volume products. Zero-scale series remain undefined and are counted.",
        "- **Bias:** mean of `forecast - actual`. Positive values mean systematic overforecasting; negative values mean underforecasting. Those directions imply different inventory risks: excess stock versus shortages.",
        "",
        "For aggregate 7- and 30-day RMSSE, each series' total error is divided by `sqrt(horizon × training scale)` before series scores are averaged.",
        "",
        "## Fold-Level Results",
        "",
        "Days 1–30 daily metrics by fold:",
        "",
    ])
    fold_rows = []
    for row in results["fold_results"]:
        fold_rows.append((row["fold"], BASELINE_LABELS[row["baseline_name"]], row["mae"], row["rmse"], row["wape"], row["rmsse"], row["bias"]))
    lines.extend(table(["Fold", "Baseline", "MAE", "RMSE", "WAPE", "RMSSE", "Bias"], fold_rows))
    lines.extend(figure("08_fold_stability_mae.png", "Fold stability comparison"))

    lines.extend(["## 1-Day Horizon Results", ""])
    lines.extend(table(["Baseline", "MAE", "RMSE", "WAPE", "RMSSE", "Bias"], metric_rows(horizon_records("day_1_daily"))))
    lines.extend(["", "Day +1 is the only horizon where the immediate pre-origin observation is literally yesterday. Persistence can therefore be more competitive here than later in the same 30-day forecast."])

    lines.extend(["", "## 7-Day Horizon Results", "", "Daily errors across Days 1–7:", ""])
    lines.extend(table(["Baseline", "MAE", "RMSE", "WAPE", "RMSSE", "Bias"], metric_rows(horizon_records("days_1_7_daily"))))
    lines.extend(["", f"Across the complete 30-day validation horizons, seasonal naive reduced MAE by **{seasonal_gain_vs_last:.2f}%** relative to last value, suggesting useful weekly recurrence. However, the 28-day mean reduced MAE by another **{mean_gain_vs_seasonal:.2f}%** relative to seasonal naive. Weekday repetition helped versus persistence but was not the strongest smoothing rule; this is association, not a causal weekday claim."])

    lines.extend(["", "## 30-Day Horizon Results", "", "Daily errors across Days 1–30:", ""])
    lines.extend(table(["Baseline", "MAE", "RMSE", "WAPE", "RMSSE", "Bias"], metric_rows(horizon_records("days_1_30_daily"))))
    lines.extend(figure("01_mae_by_baseline.png", "MAE by baseline"))
    lines.extend(figure("02_rmsse_by_baseline.png", "RMSSE by baseline"))
    lines.extend(figure("03_wape_by_baseline.png", "WAPE by baseline"))
    lines.extend(figure("05_daily_mae_by_horizon.png", "Daily MAE by forecast horizon"))

    lines.extend(["## Aggregate Horizon Results", "", "Seven-day total-demand metrics:", ""])
    lines.extend(table(["Baseline", "MAE", "RMSE", "WAPE", "RMSSE", "Bias"], metric_rows(horizon_records("days_1_7_aggregate"))))
    lines.extend(["", "Thirty-day total-demand metrics:", ""])
    lines.extend(table(["Baseline", "MAE", "RMSE", "WAPE", "RMSSE", "Bias"], metric_rows(horizon_records("days_1_30_aggregate"))))
    lines.extend(["", "Aggregate errors matter for inventory planning because replenishment decisions often depend on total demand across a lead-time horizon, even if individual daily forecasts are imperfect. The 28-day mean has the lowest 30-day aggregate MAE, WAPE, and RMSSE, while seasonal naive has a slightly lower aggregate RMSE; this is one example of why no single metric should decide the benchmark."])

    for section, segment_key, title in [
        ("Demand-Band Results", "demand_band", "Demand band"),
        ("Store Results", "store_id", "Store"),
        ("Department Results", "dept_id", "Department"),
        ("Intermittency Results", "intermittency_class", "Training intermittency"),
    ]:
        lines.extend(["", f"## {section}", ""])
        records = results["segment_results"][segment_key]
        rows = [
            (row["segment_value"], BASELINE_LABELS[row["baseline_name"]], row["mae"], row["wape"], row["rmsse"], row["bias"])
            for row in records
        ]
        lines.extend(table([title, "Baseline", "MAE", "WAPE", "RMSSE", "Bias"], rows))
        if segment_key == "demand_band":
            low_records = [row for row in records if row["segment_value"] == "low"]
            low_best = min(low_records, key=lambda row: row["mae"])
            lines.extend(["", f"Demand bands use full-history information for analysis only. They were attached after forecasts and never influenced a baseline. The **{BASELINE_LABELS[low_best['baseline_name']]}** rule has the lowest low-band MAE. A zero rule leading this sparse segment is a serious warning: future models must show that improvements are real rather than merely predicting small positive quantities on many zero days."])
            lines.extend(figure("06_wape_by_demand_band.png", "WAPE by demand band"))
        if segment_key == "intermittency_class":
            lines.extend(["", "Each classification was recomputed separately using only training history available at that fold's origin."])
            lines.extend(figure("07_mae_by_intermittency.png", "MAE by training-history intermittency"))

    lines.extend(["## Croston-SBA Analysis", ""])
    lines.extend(table(["Training class", "Croston MAE", "Croston WAPE", "Croston RMSSE", "MAE rank", "Best MAE baseline", "Best MAE"], [
        (row["intermittency_class"], row["croston_mae"], row["croston_wape"], row["croston_rmsse"], row["mae_rank"], BASELINE_LABELS[row["best_mae_baseline"]], row["best_mae"])
        for row in results["croston_by_intermittency"]
    ]))
    croston_wins = [row for row in results["croston_by_intermittency"] if row["best_mae_baseline"] == "croston_sba"]
    highly = next(row for row in results["croston_by_intermittency"] if row["intermittency_class"] == "highly_intermittent")
    croston_conclusion = "Croston did not achieve the lowest MAE in any training-intermittency class." if not croston_wins else f"Croston achieved the lowest MAE in {len(croston_wins)} class(es)."
    lines.extend(["", f"{croston_conclusion} For highly intermittent series it ranked **{highly['mae_rank']} of 5**, while zero had the best MAE. Croston remains informative as a specialized benchmark, but the validation evidence does not support assuming it wins simply because demand is intermittent."])

    lines.extend(["", "## Forecast Bias", ""])
    lines.extend(table(["Baseline", "Mean bias", "Aggregate bias"], [
        (BASELINE_LABELS[row["baseline_name"]], row["bias"], row["aggregate_bias"])
        for row in primary
    ]))
    lines.extend(["", f"All five 30-day daily baselines underforecast on average. Croston-SBA is closest to zero bias at **{primary_map['croston_sba']['bias']:.4f} units/day**, while the zero rule is necessarily most negative. Low average bias does not by itself mean low absolute error."])
    lines.extend(figure("04_bias_by_baseline.png", "Forecast bias by baseline"))

    lines.extend(["## Fold Stability", "", "The 28-day mean has the lowest MAE in each of the three folds, although its error level changes across time. This consistency supports using it as the main simple benchmark. A baseline that is slightly better on average but unstable would be a weaker comparison standard for Stage 8.", ""])

    lines.extend(["## Best Baseline Trade-offs", ""])
    lines.extend(table(["Metric", "Best baseline"], [(metric, BASELINE_LABELS[name]) for metric, name in best.items()]))
    lines.extend(["", "Balanced ranking across MAE, RMSE, WAPE, RMSSE, and absolute bias:", ""])
    lines.extend(table(["Baseline", "Mean rank", "MAE", "RMSE", "WAPE", "RMSSE", "Absolute bias"], [
        (BASELINE_LABELS[row["baseline_name"]], row["mean_rank"], row["mae"], row["rmse"], row["wape"], row["rmsse"], row["absolute_bias"])
        for row in results["balanced_ranking"]
    ]))
    lines.extend(["", f"No single metric decides the winner. **{BASELINE_LABELS[results['strongest_overall_baseline']]}** has the strongest balanced validation rank and becomes the principal simple benchmark for future models, while the metric-specific leaders remain important comparators."])

    lines.extend(["", "## Implications for ML Models", ""])
    lines.extend([
        "- A future ML model must improve on these baselines across multiple folds and horizons, not merely one metric.",
        "- The zero baseline remains a required check for sparse segments.",
        "- Global models should be evaluated by store, department, demand band, and fold-specific intermittency, even though the latter two are not predictors.",
        "- Daily predictions must remain origin-safe when aggregated into 7- and 30-day business totals.",
    ])

    lines.extend(["", "## Final-Test Lock Confirmation", ""])
    lines.extend([
        f"The locked period is `{lock['start_date']}` through `{lock['end_date']}`. No Stage 7 prediction or metric was generated for it. The latest evaluated date is `{lock['latest_evaluated_target_date']}`.",
        "",
        "The code rejects any evaluation window that overlaps the lock, and automated tests exercise that safeguard.",
    ])

    lines.extend(["", "## Limitations / Open Questions", ""])
    lines.extend([
        "- Baseline rules do not use price, event, SNAP, product-age, or hierarchy predictors.",
        "- Croston alpha is fixed at 0.1 and was not tuned.",
        "- Aggregate RMSSE uses a documented horizon-scaled extension rather than an official M5 competition aggregate metric.",
        "- Demand bands remain full-history analysis labels.",
        "- Sales can be censored by unobserved stockouts.",
        "- Stage 8 still needs an agreed first ML model, preprocessing policy, and metric-based acceptance criteria.",
        "",
        "No advanced model, final-test evaluation, hyperparameter tuning, database, inventory logic, dashboard, or deployment work was performed.",
    ])
    return "\n".join(lines).rstrip() + "\n"
