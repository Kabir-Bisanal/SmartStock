# Forecasting methodology

## Modeling population

SmartStock V1 models 300 daily series: 100 FOOD items across `CA_1`, `TX_2`, and `WI_3`. A product becomes available at its first known selling price. Pre-launch dates are not modeling targets, while genuine zero sales during the active period remain part of demand.

## Leakage controls

All splits are chronological. Three 30-day expanding-window validation folds precede the locked final test of 2016-04-23 through 2016-05-22. For a forecast made at origin `T`, only sales through `T` are known. Day +1 is appended to a synthetic history, then Day +2 features are recomputed from that history, continuing through Day +30. Actual demand inside validation or test never becomes a future lag.

Price is excluded from the V1 production forecaster because future price schedules are not guaranteed at deployment time. Demand band is evaluation metadata, never a predictor. RMSSE scales use training history only.

## Models compared

- Baselines: zero, last value, 7-day seasonal naive, 28-day historical mean, and Croston-SBA.
- Global Ridge: one-hot categorical identity plus scaled numeric/calendar/history inputs.
- HistGradientBoosting and XGBoost: nonlinear global tabular candidates with restrained validation-only tuning.

Every candidate was judged across MAE, RMSE, WAPE, RMSSE, signed bias, fold stability, daily horizons, and 7-/30-day aggregates. MAPE was avoided because many daily actuals are zero.

## Why the 28-day mean won

The 28-day mean had the strongest balanced validation ranking. Ridge improved Day +1 but degraded over recursive 30-day paths; nonlinear boosting reduced some spike-sensitive errors but did not deliver a consistent improvement across folds and horizons. Selecting the simpler method is an evidence-based result, not a failure to use more complex models.

The model configuration was frozen before opening the final test. The selected model was then evaluated once:

| Locked-test view | MAE | RMSE | WAPE | RMSSE |
|---|---:|---:|---:|---:|
| Daily, Days 1–30 | 1.382 | 2.387 | 61.20% | 0.764 |
| 30-day aggregate | 15.002 | 28.076 | 22.14% | 1.097 |

Daily accuracy is difficult for intermittent retail demand; aggregation substantially lowers percentage error and is directly useful for replenishment planning.

## Deployment refit

The evaluation artifact remains frozen through 2016-04-22. A separate deployment copy of the same selected 28-day-mean family was refit through 2016-05-22 and produced 9,000 daily forecasts for 2016-05-23 through 2016-06-21. This refit did not recompute final-test metrics or change model selection.

## Limitations

The final forecaster is intentionally simple, point-based, price-agnostic, and trained on a limited 100-item/three-store subset. Observed sales can be censored by historical stockouts. V1 does not model probabilistic demand, promotions, or direct multi-horizon objectives.
