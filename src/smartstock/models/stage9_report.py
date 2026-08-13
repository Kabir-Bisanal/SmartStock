"""Render the calculated SmartStock Stage 9 completion report."""

from __future__ import annotations

from typing import Any


def _metric_row(row: dict[str, Any]) -> str:
    return (
        f"MAE {row['mae']:.3f}, RMSE {row['rmse']:.3f}, "
        f"WAPE {100 * row['wape']:.2f}%, RMSSE {row['rmsse']:.3f}, "
        f"bias {row['bias']:+.3f}"
    )


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def render_stage9_report(summary: dict[str, Any]) -> str:
    """Return a detailed, beginner-readable Markdown report using actual values."""

    validation = summary["validation"]
    test = summary["locked_final_test"]
    selection = summary["selection"]
    model_rows = []
    for row in validation["primary_model_comparison"]:
        model_rows.append(
            [
                row["display_name"],
                f"{row['mae']:.3f}",
                f"{row['rmse']:.3f}",
                f"{100 * row['wape']:.2f}%",
                f"{row['rmsse']:.3f}",
                f"{row['bias']:+.3f}",
                str(row["selection_rank"]),
            ]
        )
    horizon_rows = [
        [
            row["label"],
            f"{row['mae']:.3f}",
            f"{row['rmse']:.3f}",
            f"{100 * row['wape']:.2f}%",
            f"{row['rmsse']:.3f}",
            f"{row['bias']:+.3f}",
        ]
        for row in test["horizon_results"]
    ]
    fold_rows = [
        [
            row["fold"].replace("validation_fold_", "Fold "),
            row["display_name"],
            f"{row['mae']:.3f}",
            f"{row['rmse']:.3f}",
            f"{100 * row['wape']:.2f}%",
            f"{row['rmsse']:.3f}",
            f"{row['bias']:+.3f}",
        ]
        for row in validation["fold_results"]
    ]
    tuning_rows = [
        [
            row["model_name"],
            f"{row['mae']:.3f}",
            f"{row['rmse']:.3f}",
            f"{100 * row['wape']:.2f}%",
            f"{row['rmsse']:.3f}",
            f"{row['bias']:+.3f}",
            str(row["selection_rank"]),
        ]
        for row in validation["tuning_results"]
    ]

    segment_sections = []
    for segment_type, title in (
        ("demand_band", "Demand Band"),
        ("store_id", "Store"),
        ("dept_id", "Department"),
        ("intermittency_class", "Intermittency"),
    ):
        rows = []
        for row in test["segment_results"][segment_type]:
            rows.append(
                [
                    str(row["segment_value"]),
                    f"{row['mae']:.3f}",
                    f"{row['rmse']:.3f}",
                    f"{100 * row['wape']:.2f}%",
                    f"{row['rmsse']:.3f}",
                    f"{row['bias']:+.3f}",
                ]
            )
        segment_sections.append(
            f"### {title}\n\n"
            + _table([title, "MAE", "RMSE", "WAPE", "RMSSE", "Bias"], rows)
        )

    selected_validation = validation["selected_model_days_1_30_daily"]
    selected_test = test["days_1_30_daily"]
    report = f"""# SmartStock Stage 9 — Final Forecasting Model Report

## Executive Summary

Stage 9 completed the Version 1 forecasting layer. HistGradientBoosting and XGBoost were evaluated with true 30-day recursive forecasting on the three frozen chronological folds. XGBoost was the stronger nonlinear family and received the permitted six-configuration tuning pass. Neither nonlinear model established a balanced improvement over the 28-Day Historical Mean, so the simple mean was frozen as the production model before the locked test was opened.

The one-time locked test covered **{test['start_date']} through {test['end_date']}**. The selected model produced {_metric_row(selected_test)} on all 9,000 daily predictions. This result is an unbiased final report, not a reason to reopen model selection.

## Candidate Models and Information Policy

- **HistGradientBoostingRegressor:** nonlinear scikit-learn tree boosting with training-only ordinal categorical encoding.
- **XGBoost Regressor:** nonlinear boosted trees with training-only one-hot categorical encoding and histogram tree construction.
- **Global Ridge:** the frozen Stage 8 linear benchmark.
- **28-Day Historical Mean:** the strongest Stage 7 simple baseline.

The nonlinear candidates used the 26 approved core fields from Stage 8. Price fields, `demand_band`, and final-test information were excluded. Numeric inputs were median-imputed from training rows and were not standardized because tree splits do not require scaling. The optional zero-rate expansion was deliberately omitted to avoid adding a new predicted zero/non-zero threshold to recursive behavior after the core nonlinear comparison already answered the Version 1 question.

The selected 28-day mean has one effective predictor: the recursively rebuilt `sales_roll_mean_28`. It does not use identity, calendar, event, SNAP, price, or demand-band fields when calculating the forecast. The common production interface retains the broader context contract so later stages can call the same daily engine safely.

## Recursive Forecast-Origin Safety

For each validation fold and the final test, actual demand was available only through the forecast origin. Day +1 was forecast, clipped at zero, appended to synthetic history, and then used to rebuild Day +2 lags and rolling statistics. This continued through Day +30. Actual future sales and persisted future lag/rolling columns were attached or inspected only after the full forecast existed.

The mutation audit changed future actual demand and future precomputed demand features to extreme values without changing any nonlinear validation forecast. The production audit separately confirmed recursive lag-1, lag-7, and rolling updates. No target-day price was loaded as a predictor.

## Metric Definitions

- **MAE:** average absolute unit error; easy to interpret operationally.
- **RMSE:** square-root mean squared error; penalizes large misses more heavily than MAE.
- **WAPE:** total absolute error divided by total actual demand. It remains usable when individual actual days are zero, unlike ordinary MAPE. A zero group denominator remains undefined rather than being replaced with an arbitrary epsilon.
- **RMSSE:** each series' error scaled by its one-step naive error from training history only, then averaged across defined series. This makes differently sized series more comparable.
- **Bias:** mean `(forecast - actual)`. Positive means systematic overforecasting; negative means systematic underforecasting. Aggregate bias is the signed total error.

## Validation Model Comparison — Days 1–30 Daily

{_table(['Model', 'MAE', 'RMSE', 'WAPE', 'RMSSE', 'Bias', 'Balanced rank'], model_rows)}

XGBoost improved RMSE versus both Ridge and the mean, showing that nonlinear trees reduced some large errors. It did not improve 30-day MAE, WAPE, RMSSE, aggregate accuracy, or balanced rank. The 28-day mean therefore remained the defensible production choice. Its advantage was not inferred from the locked test.

## Limited XGBoost Tuning

XGBoost was the stronger base nonlinear family. Exactly {selection['tuning_configurations_evaluated']} manually declared configurations were evaluated across all three validation folds; no automated or large search was performed.

{_table(['Configuration', 'MAE', 'RMSE', 'WAPE', 'RMSSE', 'Bias', 'Balanced rank'], tuning_rows)}

The validation-selected XGBoost configuration was **{validation['selected_xgboost_configuration']}**. A different configuration had the lowest isolated 30-day MAE, but the selected configuration ranked better across the full horizon, aggregate, scale, bias, and stability policy. This is why no single metric was allowed to determine tuning.

## Validation Fold Stability — Days 1–30 Daily

{_table(['Fold', 'Model', 'MAE', 'RMSE', 'WAPE', 'RMSSE', 'Bias'], fold_rows)}

XGBoost was notably stable across folds, but stability alone did not overcome its weaker overall MAE/WAPE and aggregate results. The production mean beat the selected XGBoost configuration on 30-day MAE in {validation['mean_28_fold_wins_vs_xgboost']} of 3 folds.

## Validation Horizon Behavior

Nonlinear models were strongest at Day +1: XGBoost recorded the best validation Day +1 MAE, followed by HistGradientBoosting and Ridge. As their own predictions were recursively fed back, that advantage weakened. At Days 1–30, the 28-day mean had the lowest MAE/WAPE, while XGBoost had the lowest RMSE. This directly answers the Stage 9 question: nonlinear boosting improved short-horizon and spike-sensitive error, but did not solve recursive long-horizon drift broadly enough for Version 1 selection.

## Final Model Freeze

- Selected model: **{selection['selected_model_name']}**
- Family: **{selection['selected_model_family']}**
- Parameters: `{selection['selected_parameters']}`
- Selection frozen: **{selection['selection_frozen_at_utc']}**
- Frozen configuration hash: `{selection['frozen_config_sha256']}`
- Test targets inspected at freeze: **No**
- Post-test tuning or family changes allowed: **No**

The 28-day forecast is recalculated recursively. At each future day it predicts the mean of the latest 28 values, where earlier horizon predictions replace unknown future actuals. Forecasts are continuous, non-negative, and not rounded.

## Locked Final-Test Results

{_table(['Horizon', 'MAE', 'RMSE', 'WAPE', 'RMSSE', 'Bias'], horizon_rows)}

The 30-day aggregate forecast had total-demand MAE **{test['days_1_30_aggregate']['mae']:.3f} units per item-store**, WAPE **{100 * test['days_1_30_aggregate']['wape']:.2f}%**, RMSSE **{test['days_1_30_aggregate']['rmsse']:.3f}**, and aggregate bias **{test['days_1_30_aggregate']['aggregate_bias']:+.1f} units** across the 300 series totals.

## Validation vs Locked Test

Validation Days 1–30 daily performance was {_metric_row(selected_validation)}. Locked-test performance was {_metric_row(selected_test)}. Test MAE was **{summary['validation_vs_test']['mae_change_pct']:+.2f}%** relative to validation, while test WAPE changed by **{summary['validation_vs_test']['wape_change_pct']:+.2f}%**. The test period was somewhat easier by MAE/RMSE/WAPE, but RMSSE was higher and underforecast bias was more negative. These differences are reported without reopening tuning.

## Locked-Test Segment Results

{chr(10).join(segment_sections)}

The low-demand and highly intermittent segments have WAPE above 100% even when their unit MAE is small. That is expected when the denominator contains very little actual demand. High-demand and regular series dominate unit error, while percentage metrics expose the relative difficulty of sparse series.

## Production Artifacts and API

- Frozen configuration: `config/v1_final_model.json`
- Serialized model bundle: `{summary['production']['artifact_path']}` ({summary['production']['artifact_size_bytes']:,} bytes)
- Final-test predictions: `{test['predictions_path']}` ({test['prediction_rows']:,} rows)
- Forecast API: `smartstock.models.production_forecaster.forecast_demand`
- Training command: `python -m smartstock.models.train_final_model`

The forecast function accepts horizon 1, 7, or 30, requires real calendar/context rows for the requested dates, and returns tidy item/store/date/horizon forecasts. It will not invent future calendar data.

## Why Baselines Matter

A learned model is useful only when it reliably beats simple rules under the same evaluation conditions. Persistence tests whether yesterday alone is enough. Seasonal naive tests whether last week's weekday pattern repeats. Croston addresses non-zero demand sizes and gaps for intermittent products. Stage 9 demonstrated the same principle at model selection: greater algorithmic complexity did not automatically produce a better inventory-horizon forecast, so the simpler validated method was chosen.

## Integrity and Test-Lock Confirmation

- Final-test evaluation count: **1**
- Config existed and was hashed before test scoring: **Yes**
- Test target used in model selection: **No**
- Test actual attached only after all 30-day forecasts existed: **Yes**
- Preprocessing fitted on training rows only: **Yes**
- Frozen Stage 8-and-earlier hashes unchanged: **Yes**
- Official negative forecasts: **0**
- Undefined final-test RMSSE series: **0**
- Second final-test run is blocked by existing receipt/prediction artifacts: **Yes**

## Performance

Validation selection and tuning took **{summary['performance']['validation_seconds']:.1f} seconds** in the compatible verification runtime. The final training, recursive forecast, scoring, and serialization phase took **{summary['performance']['final_test_seconds']:.1f} seconds**. Candidate evaluation loaded the feature CSV once, reused grouped histories, and predicted each recursive day as a 300-row batch.

## Limitations and Open Questions

- The final model is price-agnostic because future price schedules are not guaranteed in deployment.
- The 28-day mean is robust but cannot explicitly learn events, promotions, trends, or cross-series relationships.
- Recursive mean forecasts become smoother across a long horizon and may underreact to sudden demand shifts; the final test bias was negative.
- Descriptive Stage 5 work predates the Stage 6 lock, as already recorded in the validation manifest. From Stage 6 onward, selection obeyed the lock.
- The test contains only one 30-day period. It is an unbiased final snapshot, not evidence for further Version 1 tuning.

## Deliberately Deferred

No inventory optimization, safety stock, reorder point, cost model, PostgreSQL integration, Streamlit interface, Docker packaging, or deployment work was implemented in Stage 9. Those belong to Stages 10–12.
"""
    return report
