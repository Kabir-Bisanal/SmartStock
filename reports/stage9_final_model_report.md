# SmartStock Stage 9 — Final Forecasting Model Report

## Executive Summary

Stage 9 completed the Version 1 forecasting layer. HistGradientBoosting and XGBoost were evaluated with true 30-day recursive forecasting on the three frozen chronological folds. XGBoost was the stronger nonlinear family and received the permitted six-configuration tuning pass. Neither nonlinear model established a balanced improvement over the 28-Day Historical Mean, so the simple mean was frozen as the production model before the locked test was opened.

The one-time locked test covered **2016-04-23 through 2016-05-22**. The selected model produced MAE 1.382, RMSE 2.387, WAPE 61.20%, RMSSE 0.764, bias -0.108 on all 9,000 daily predictions. This result is an unbiased final report, not a reason to reopen model selection.

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

| Model | MAE | RMSE | WAPE | RMSSE | Bias | Balanced rank |
| --- | --- | --- | --- | --- | --- | --- |
| 28-Day Mean | 1.462 | 2.801 | 65.86% | 0.736 | -0.066 | 1 |
| Global Ridge | 1.475 | 2.722 | 66.44% | 0.762 | +0.024 | 2 |
| XGBoost (selected config) | 1.493 | 2.664 | 67.27% | 0.753 | +0.135 | 3 |
| HistGradientBoosting | 1.519 | 2.665 | 68.46% | 0.758 | +0.237 | 4 |

XGBoost improved RMSE versus both Ridge and the mean, showing that nonlinear trees reduced some large errors. It did not improve 30-day MAE, WAPE, RMSSE, aggregate accuracy, or balanced rank. The 28-day mean therefore remained the defensible production choice. Its advantage was not inferred from the locked test.

## Limited XGBoost Tuning

XGBoost was the stronger base nonlinear family. Exactly 6 manually declared configurations were evaluated across all three validation folds; no automated or large search was performed.

| Configuration | MAE | RMSE | WAPE | RMSSE | Bias | Balanced rank |
| --- | --- | --- | --- | --- | --- | --- |
| xgb_tune_03 | 1.493 | 2.664 | 67.27% | 0.753 | +0.135 | 1 |
| xgb_tune_04 | 1.478 | 2.656 | 66.59% | 0.759 | +0.009 | 2 |
| xgb_tune_02 | 1.498 | 2.698 | 67.51% | 0.751 | +0.124 | 3 |
| xgboost_base | 1.500 | 2.713 | 67.57% | 0.751 | +0.123 | 4 |
| xgb_tune_06 | 1.500 | 2.700 | 67.60% | 0.753 | +0.122 | 5 |
| xgb_tune_05 | 1.524 | 2.757 | 68.69% | 0.749 | +0.219 | 6 |

The validation-selected XGBoost configuration was **xgb_tune_03**. A different configuration had the lowest isolated 30-day MAE, but the selected configuration ranked better across the full horizon, aggregate, scale, bias, and stability policy. This is why no single metric was allowed to determine tuning.

## Validation Fold Stability — Days 1–30 Daily

| Fold | Model | MAE | RMSE | WAPE | RMSSE | Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Fold 1 | HistGradientBoosting | 1.498 | 2.551 | 69.53% | 0.762 | +0.252 |
| Fold 1 | 28-Day Mean | 1.382 | 2.630 | 64.13% | 0.729 | -0.175 |
| Fold 1 | Global Ridge | 1.407 | 2.524 | 65.27% | 0.754 | -0.054 |
| Fold 1 | XGBoost (selected config) | 1.499 | 2.626 | 69.55% | 0.764 | +0.173 |
| Fold 2 | HistGradientBoosting | 1.541 | 2.797 | 67.42% | 0.761 | +0.176 |
| Fold 2 | 28-Day Mean | 1.514 | 3.034 | 66.26% | 0.739 | -0.130 |
| Fold 2 | Global Ridge | 1.498 | 2.895 | 65.55% | 0.764 | -0.019 |
| Fold 2 | XGBoost (selected config) | 1.487 | 2.724 | 65.09% | 0.748 | +0.050 |
| Fold 3 | HistGradientBoosting | 1.519 | 2.641 | 68.49% | 0.751 | +0.283 |
| Fold 3 | 28-Day Mean | 1.489 | 2.722 | 67.14% | 0.739 | +0.108 |
| Fold 3 | Global Ridge | 1.519 | 2.734 | 68.51% | 0.768 | +0.145 |
| Fold 3 | XGBoost (selected config) | 1.493 | 2.641 | 67.30% | 0.748 | +0.180 |

XGBoost was notably stable across folds, but stability alone did not overcome its weaker overall MAE/WAPE and aggregate results. The production mean beat the selected XGBoost configuration on 30-day MAE in 2 of 3 folds.

## Validation Horizon Behavior

Nonlinear models were strongest at Day +1: XGBoost recorded the best validation Day +1 MAE, followed by HistGradientBoosting and Ridge. As their own predictions were recursively fed back, that advantage weakened. At Days 1–30, the 28-day mean had the lowest MAE/WAPE, while XGBoost had the lowest RMSE. This directly answers the Stage 9 question: nonlinear boosting improved short-horizon and spike-sensitive error, but did not solve recursive long-horizon drift broadly enough for Version 1 selection.

## Final Model Freeze

- Selected model: **mean_28**
- Family: **mean_28**
- Parameters: `{'window_days': 28}`
- Selection frozen: **2026-08-13T11:06:45+00:00**
- Frozen configuration hash: `4bcc1542c44434903d9f0be14715bf83c012b8caf1ceddef6ebb8ad979e4a543`
- Test targets inspected at freeze: **No**
- Post-test tuning or family changes allowed: **No**

The 28-day forecast is recalculated recursively. At each future day it predicts the mean of the latest 28 values, where earlier horizon predictions replace unknown future actuals. Forecasts are continuous, non-negative, and not rounded.

## Locked Final-Test Results

| Horizon | MAE | RMSE | WAPE | RMSSE | Bias |
| --- | --- | --- | --- | --- | --- |
| Day +1 daily | 1.345 | 2.121 | 58.40% | 0.574 | -0.090 |
| Days 1-7 daily | 1.324 | 2.250 | 66.00% | 0.667 | +0.185 |
| 7-day aggregate | 4.632 | 9.858 | 32.98% | 0.700 | +1.295 |
| Days 1-30 daily | 1.382 | 2.387 | 61.20% | 0.764 | -0.108 |
| 30-day aggregate | 15.002 | 28.076 | 22.14% | 1.097 | -3.233 |

The 30-day aggregate forecast had total-demand MAE **15.002 units per item-store**, WAPE **22.14%**, RMSSE **1.097**, and aggregate bias **-969.8 units** across the 300 series totals.

## Validation vs Locked Test

Validation Days 1–30 daily performance was MAE 1.462, RMSE 2.801, WAPE 65.86%, RMSSE 0.736, bias -0.066. Locked-test performance was MAE 1.382, RMSE 2.387, WAPE 61.20%, RMSSE 0.764, bias -0.108. Test MAE was **-5.45%** relative to validation, while test WAPE changed by **-7.08%**. The test period was somewhat easier by MAE/RMSE/WAPE, but RMSSE was higher and underforecast bias was more negative. These differences are reported without reopening tuning.

## Locked-Test Segment Results

### Demand Band

| Demand Band | MAE | RMSE | WAPE | RMSSE | Bias |
| --- | --- | --- | --- | --- | --- |
| high | 2.308 | 3.663 | 48.22% | 0.683 | -0.179 |
| low | 0.739 | 1.125 | 103.25% | 0.849 | -0.097 |
| medium | 1.107 | 1.585 | 85.11% | 0.761 | -0.049 |
### Store

| Store | MAE | RMSE | WAPE | RMSSE | Bias |
| --- | --- | --- | --- | --- | --- |
| CA_1 | 1.501 | 2.595 | 58.16% | 0.763 | -0.198 |
| TX_2 | 1.301 | 2.250 | 60.40% | 0.704 | -0.129 |
| WI_3 | 1.344 | 2.303 | 65.88% | 0.827 | +0.004 |
### Department

| Department | MAE | RMSE | WAPE | RMSSE | Bias |
| --- | --- | --- | --- | --- | --- |
| FOODS_1 | 1.491 | 2.436 | 72.18% | 0.871 | -0.451 |
| FOODS_2 | 1.301 | 2.263 | 74.65% | 0.766 | +0.035 |
| FOODS_3 | 1.393 | 2.433 | 54.38% | 0.736 | -0.087 |
### Intermittency

| Intermittency | MAE | RMSE | WAPE | RMSSE | Bias |
| --- | --- | --- | --- | --- | --- |
| highly_intermittent | 0.421 | 0.661 | 133.20% | 0.839 | -0.042 |
| intermittent | 0.907 | 1.519 | 81.15% | 0.807 | -0.050 |
| regular | 1.984 | 3.131 | 54.23% | 0.711 | -0.175 |

The low-demand and highly intermittent segments have WAPE above 100% even when their unit MAE is small. That is expected when the denominator contains very little actual demand. High-demand and regular series dominate unit error, while percentage metrics expose the relative difficulty of sparse series.

## Production Artifacts and API

- Frozen configuration: `config/v1_final_model.json`
- Serialized model bundle: `models/smartstock_v1_forecaster.joblib` (4,743 bytes)
- Final-test predictions: `data/processed/models/stage9/final_test_predictions.csv` (9,000 rows)
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

Validation selection and tuning took **881.4 seconds** in the compatible verification runtime. The final training, recursive forecast, scoring, and serialization phase took **12.8 seconds**. Candidate evaluation loaded the feature CSV once, reused grouped histories, and predicted each recursive day as a 300-row batch.

## Limitations and Open Questions

- The final model is price-agnostic because future price schedules are not guaranteed in deployment.
- The 28-day mean is robust but cannot explicitly learn events, promotions, trends, or cross-series relationships.
- Recursive mean forecasts become smoother across a long horizon and may underreact to sudden demand shifts; the final test bias was negative.
- Descriptive Stage 5 work predates the Stage 6 lock, as already recorded in the validation manifest. From Stage 6 onward, selection obeyed the lock.
- The test contains only one 30-day period. It is an unbiased final snapshot, not evidence for further Version 1 tuning.

## Deliberately Deferred

No inventory optimization, safety stock, reorder point, cost model, PostgreSQL integration, Streamlit interface, Docker packaging, or deployment work was implemented in Stage 9. Those belong to Stages 10–12.
