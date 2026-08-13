# SmartStock Stage 6 — Leakage-Safe Feature Engineering & Chronological Validation Design

Generated at `2026-08-13T06:13:08+00:00` from the frozen SmartStock Version 1 interim dataset.

> Stage 6 builds modeling inputs and a validation design only. No forecast model is fitted, scored, or selected.

## Executive Summary

Stage 6 excluded **101,794** unavailable pre-launch rows, retained **480,506** active rows, and persisted **472,106** fully feature-ready rows plus their warm-up rows. The processed file contains **480,506 rows × 49 columns** across 100 products, three stores, and 300 item-store series.

Every default model feature passed the manifest leakage audit. Three expanding 30-day validation folds precede a locked 30-day final test from `2016-04-23` through `2016-05-22`.

## Modeling Policies

- Pre-launch rows are not demand targets.
- Active-period zero sales remain legitimate observations.
- First known price is the availability proxy; it is not an explicit inventory record.
- Future models are intended to be global across the 300 series.
- `demand_band` is analysis-only because it uses full-history demand.
- Evaluation must be chronological; random splitting is prohibited.
- Historical demand and price features use information strictly before the target date or week.

## Active-Period Filtering

| Measure | Result |
| --- | --- |
| Input rows | 582,300 |
| Pre-launch rows excluded | 101,794 |
| Pre-launch share | 17.48% |
| Active rows retained | 480,506 |
| Active zero-sales rows retained | 235,769 |
| Active zero rate | 49.07% |
| Price-based availability pairs | 300 |
| Positive-sale fallback pairs | 0 |

Filtering before feature creation prevents unavailable assortment periods from teaching a model that a not-yet-offered product had genuine zero demand.

## Calendar Features

Calendar predictors are known for the target date in advance: `day_of_week`, `day_of_month`, `month`, `quarter`, `year`, `week_of_year`, and `is_weekend`. Parsed dates were checked against the original M5 weekday, month, and year fields.

The sine/cosine encodings for weekday and month put the end and beginning of each cycle close together. For example, December and January are neighbors on a circle even though their integer labels are 12 and 1.

## Event / SNAP Features

`is_event` accompanies the four raw event identity fields. Empty event fields mean no event in that slot; no sales-derived target encoding was used. There are **39,012** active event rows.

A single `snap_active` field maps `CA_1 → snap_CA`, `TX_2 → snap_TX`, and `WI_3 → snap_WI`. The unrelated state flags are not persisted as predictors.

## Product-Age Features

`product_age_days` equals target date minus availability date and starts at zero. `product_age_weeks` stores completed weeks. Observed active ages range from **0** to **1,940 days**. Recently launched products can behave differently from mature products even when they share a department or store.

## Sales Lag Features

| Feature | Exact definition |
| --- | --- |
| sales_lag_1 | sales(t-1) |
| sales_lag_7 | sales(t-7) |
| sales_lag_14 | sales(t-14) |
| sales_lag_28 | sales(t-28) |

Every shift is calculated independently within `(store_id, item_id)`. Yesterday's observed sales may help predict today, but today's target cannot be used to predict itself.

## Rolling Features

The rolling means use windows of 7, 14, and 28 prior observations. Rolling sample standard deviations use 7 and 28 prior observations. All formulas use the equivalent of `sales.shift(1).rolling(window)`, so the target day's sale is outside its own window.

Without `shift(1)`, a rolling average would contain `sales(t)`. A model evaluated with that feature could appear artificially accurate because part of the answer was included in the input.

## Intermittency Features

`zero_rate_7` and `zero_rate_28` measure the share of zero-demand days in strictly historical windows. `days_since_last_positive_sale` looks backward from `t-1`. Missing prior history remains missing and is never confused with an observed zero.

## Price-Feature Policy

Default forecast-safe price features use completed prior M5 weeks:

- `previous_week_price`: price in the preceding item-store week.
- `price_change_previous_week`: preceding-week price minus the price two weeks before the target week.
- `price_pct_change_previous_week`: the same change as a percentage of the earlier price.
- `price_vs_4week_median_lagged`: preceding-week price divided by the median price of the preceding four weeks, minus one.

`known_future_sell_price` preserves the actual target-week price only for explicitly defined known-future-price experiments. It is excluded from the default feature list. A price appearing in a historical holdout file does not prove that a production forecasting system would know it before making the forecast.

## Extreme Price-Change Review

The weekly review found **1,022** non-zero sequential changes. The maximum absolute percentage change is **1012.00%**, from **0.25** to **2.78**. The very low previous-price denominator is the primary reason the percentage is so large.

| Store | Item | Week | Previous | New | Absolute change | % change |
| --- | --- | --- | --- | --- | --- | --- |
| TX_2 | FOODS_3_658 | 11,411 | 0.2500 | 2.7800 | 2.5300 | 1,012.0000 |
| TX_2 | FOODS_2_367 | 11,536 | 1.0000 | 4.9700 | 3.9700 | 397.0000 |
| TX_2 | FOODS_3_724 | 11,210 | 2.0000 | 6.4800 | 4.4800 | 224.0000 |
| TX_2 | FOODS_2_170 | 11,227 | 1.0000 | 2.8300 | 1.8300 | 183.0000 |
| TX_2 | FOODS_2_170 | 11,435 | 1.0000 | 2.8300 | 1.8300 | 183.0000 |
| TX_2 | FOODS_3_539 | 11,226 | 1.0000 | 2.7800 | 1.7800 | 178.0000 |
| TX_2 | FOODS_3_539 | 11,405 | 1.0000 | 2.7800 | 1.7800 | 178.0000 |
| TX_2 | FOODS_3_539 | 11,413 | 1.0000 | 2.7800 | 1.7800 | 178.0000 |
| TX_2 | FOODS_1_017 | 11,222 | 0.9900 | 2.6800 | 1.6900 | 170.7071 |
| TX_2 | FOODS_1_017 | 11,235 | 0.9900 | 2.6800 | 1.6900 | 170.7071 |

No price was deleted, capped, or transformed. Whether later modeling clips or transforms percentage changes must be decided using training and validation periods only.

## Feature Missingness / Warm-up

| Required history feature | Missing rows | Missing % |
| --- | --- | --- |
| sales_lag_1 | 300 | 0.0624 |
| sales_lag_7 | 2,100 | 0.4370 |
| sales_lag_14 | 4,200 | 0.8741 |
| sales_lag_28 | 8,400 | 1.7482 |
| sales_roll_mean_7 | 2,100 | 0.4370 |
| sales_roll_mean_14 | 4,200 | 0.8741 |
| sales_roll_mean_28 | 8,400 | 1.7482 |
| sales_roll_std_7 | 2,100 | 0.4370 |
| sales_roll_std_28 | 8,400 | 1.7482 |
| zero_rate_7 | 2,100 | 0.4370 |
| zero_rate_28 | 8,400 | 1.7482 |
| days_since_last_positive_sale | 910 | 0.1894 |
| previous_week_price | 2,100 | 0.4370 |
| price_change_previous_week | 4,200 | 0.8741 |
| price_pct_change_previous_week | 4,200 | 0.8741 |
| price_vs_4week_median_lagged | 8,400 | 1.7482 |

A row is feature-ready only when all required demand-history and price-history fields are present. **8,400 rows** remain persisted but are flagged not ready; **472,106 rows (98.25%)** are ready. First readiness occurs between product ages **28 and 28 days** across the 300 series.

## Final Feature Dataset

| Measure | Result |
| --- | --- |
| Path | data/processed/smartstock_v1_features.csv |
| Rows | 480,506 |
| Columns | 49 |
| Feature-ready rows | 472,106 |
| Products | 100 |
| Stores | 3 |
| Item-store pairs | 300 |
| Date range | 2011-01-29 through 2016-05-22 |
| CSV size | 123.41 MiB |
| Feature DataFrame memory | 61.51 MiB |

Warm-up rows are kept with `is_feature_ready = false`; nothing is silently dropped beyond the agreed pre-launch exclusion. Split labels are not embedded because an expanding-window row can be training data in one fold and validation data in an earlier fold.

Constant persisted columns are: `cat_id`, `is_active`. They are retained for context or auditability but excluded from default modeling when they add no variation.

## Chronological Validation Design

Training expands through all feature-ready history strictly before each validation start. Random splitting is prohibited because it would allow later retail behavior to influence evaluation of predictions for earlier dates.

Multiple recent origins reduce the chance of selecting a model because it happened to perform well in one unusual month.

## Locked Final Test

The final test is locked from **2016-04-23 through 2016-05-22**, containing **9,000 feature-ready rows** across all 300 series. From Stage 6 onward its target performance must not guide feature or model choices.

Stage 5 descriptively inspected the full history before this lock. That limitation is documented honestly; the remedy is strict non-inspection of final-test model performance from this point forward.

## Rolling Validation Folds

| Fold | Training end | Training rows | Validation dates | Validation rows |
| --- | --- | --- | --- | --- |
| validation_fold_1 | 2016-01-23 | 436,106 | 2016-01-24 through 2016-02-22 | 9,000 |
| validation_fold_2 | 2016-02-22 | 445,106 | 2016-02-23 through 2016-03-23 | 9,000 |
| validation_fold_3 | 2016-03-23 | 454,106 | 2016-03-24 through 2016-04-22 | 9,000 |

## Horizon Evaluation Plan

- 1-day: evaluate the Day+1 daily forecast.
- 7-day: evaluate daily Day+1 through Day+7 errors and the aggregate seven-day total.
- 30-day: evaluate daily Day+1 through Day+30 errors and the aggregate 30-day total.

Future models should generate daily forecasts; the 7-day and 30-day business totals are aggregations of those daily predictions.

## Leakage Audit

| Audit check | Result |
| --- | --- |
| default_sources_allowed | Passed |
| demand_band_is_analysis_only | Passed |
| target_is_not_feature | Passed |
| current_price_is_scenario_only | Passed |
| no_duplicate_feature_names | Passed |

The **39 default features** come only from static identity, known same-day calendar information, the availability proxy, past demand, or past prices. No default feature uses future sales, full-history target aggregates, optional current price, or final-test statistics.

## Features Deliberately Excluded

- `demand_band` as a predictor: it uses full-history sales and is analysis-only.
- Current target-week price from the default set: retained only as `known_future_sell_price` for explicit scenarios.
- All three raw state SNAP columns: replaced with the store-relevant flag.
- Target encodings and full-history product averages.
- Future sales, centered rolling windows, target-day rolling values, and final-test-derived statistics.
- A 56-day feature family: excluded to keep the first model foundation compact and preserve more history.

## Risks / Open Questions

- First known price is an availability proxy rather than explicit assortment or inventory status.
- Recorded sales may understate unconstrained demand during unobserved stockouts.
- Future price availability must be defined for each deployment scenario.
- Event and SNAP features are calendar associations, not causal effects.
- Extreme price percentages are denominator-sensitive.
- Categorical handling, baseline metrics, and the precise global-model training protocol remain Stage 7 decisions.

## Recommended Next Stage

Review and freeze a Stage 7 baseline and metric strategy before fitting anything. That stage should compare leakage-safe chronological baselines across the three validation folds, preserve the final-test lock, and report results by store, department, demand band, and intermittency segment without using those analysis-only groups as leaked predictors.

## Scope Confirmation

No forecasting model, model score, hyperparameter search, database, inventory calculation, dashboard, Docker configuration, or deployment artifact was created in Stage 6.
