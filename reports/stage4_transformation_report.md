# SmartStock Stage 4 — Version 1 Subset Creation & Controlled Data Transformation

Generated at `2026-08-13T02:28:14+00:00` from the local official M5 raw files.

> Stage 4 selected a reproducible 100-item FOODS subset, transformed only its 300 item-store rows to daily long format, and joined raw calendar and weekly price attributes. No forecasting features or price imputation were added.

## Executive Summary

Version 1 contains **100 FOODS items** across **CA_1, TX_2, WI_3**, producing **300 item-store series** and **582,300 daily rows** from 2011-01-29 through 2016-05-22. The tracked manifest fixes the item list and seed; the generated working CSV remains Git-ignored under `data/interim/`.

All calendar keys matched and neither join multiplied rows. Prices remain missing in **101,794 rows (17.48%)**; those values were deliberately not filled.

## Store Selection

The frozen cross-state stores were used exactly as specified:

| State | Store |
| --- | --- |
| California | CA_1 |
| Texas | TX_2 |
| Wisconsin | WI_3 |

## Eligibility Filtering

The starting population contained **1,437 FOODS items**. **1,340** passed all four rules and **97** failed at least one. **5** items failed more than one rule.

Rule failures are counted independently, so a product failing multiple rules appears in multiple rows. These counts must not be added as though filtering were sequential.

| Eligibility rule | Independent failures | Exact policy |
| --- | --- | --- |
| A — Required stores | 0 | Structural sales row in CA_1, TX_2, and WI_3 |
| B — Extreme sparsity | 10 | Aggregated selected-store zero prevalence must be < 95% |
| C — Useful history | 92 | 1941 − earliest positive-sale day must be at least 730 |
| D — Price coverage | 0 | At least 70% active-period weekly price coverage in each selected store |

Price coverage is measured separately for every item-store pair. Its denominator is the distinct calendar weeks from that store's first positive sale through `d_1941`; weeks before launch are not penalized. Each of the three stores must reach 70%.

## Demand Stratification

The demand metric is **mean daily units across all three selected stores and all 1,941 days**:

```text
mean_daily_demand = total selected-store units / (3 × 1,941)
```

Tertiles were calculated from the eligible FOODS population. Boundary policy: low includes the lower threshold; medium is above the lower and includes the upper threshold; high is above the upper threshold.

| Band | Mean daily demand definition | Eligible items |
| --- | --- | --- |
| Low | ≤ 0.5889289599 | 447 |
| Medium | > 0.5889289599 and ≤ 1.3705421032 | 446 |
| High | > 1.3705421032 | 447 |

### Quota availability before sampling

| Department | Demand band | Available eligible items | Required |
| --- | --- | --- | --- |
| FOODS_1 | low | 86 | 5 |
| FOODS_1 | medium | 73 | 5 |
| FOODS_1 | high | 49 | 5 |
| FOODS_2 | low | 148 | 9 |
| FOODS_2 | medium | 128 | 10 |
| FOODS_2 | high | 87 | 9 |
| FOODS_3 | low | 213 | 19 |
| FOODS_3 | medium | 245 | 19 |
| FOODS_3 | high | 311 | 19 |

## Final Product Selection

| Department | Selected items |
| --- | --- |
| FOODS_1 | 15 |
| FOODS_2 | 28 |
| FOODS_3 | 57 |

| Demand band | Selected items |
| --- | --- |
| high | 33 |
| low | 33 |
| medium | 34 |

Selection used `random_seed = 42` inside each department × demand-band cell. Sorting each candidate pool by `item_id` before sampling ensures the same source data produces the same 100 IDs. The exact list is stored in `config/v1_subset.json`.

Stratified sampling preserves low-, medium-, and high-demand products rather than choosing only best sellers. This creates a more realistic forecasting challenge because retail portfolios contain intermittent and slow-moving items as well as fast movers.

## Transformation

```text
1,437 selected-store FOODS candidates
    -> eligibility + stratified selection
100 items × 3 stores = 300 wide rows
    -> melt d_1 ... d_1941
582,300 daily rows
    -> calendar join on d
    -> price join on store_id + item_id + wm_yr_wk
```

Filtering before melting is the key performance decision. Melting the full M5 sales table would create about 59.2 million rows; melting 300 selected rows creates only 582,300.

## Final Dataset

| Measure | Result |
| --- | --- |
| Output | data/interim/smartstock_v1_long.csv |
| Rows | 582,300 |
| Columns | 23 |
| Unique items | 100 |
| Stores | CA_1, TX_2, WI_3 |
| Item-store pairs | 300 |
| Date range | 2011-01-29 to 2016-05-22 |
| In-memory size (MiB) | 22.3400 |
| CSV size (MiB) | 70.5400 |

## Missing Prices

| Measure | Rows |
| --- | --- |
| Total missing prices | 101,794 |
| Missing percentage | 17.481367% |
| Before first known price | 101,794 |
| On/after first known price | 0 |
| After last known price | 0 |
| Before first positive sale | 101,794 |
| On/after first positive sale | 0 |
| On/after launch with positive sales | 0 |
| Products with at least one missing price | 62 |
| Products without missing prices | 38 |

### Missing prices by store

| Store | Rows | Missing | Missing percentage |
| --- | --- | --- | --- |
| CA_1 | 194,100 | 34,699 | 17.8769% |
| TX_2 | 194,100 | 32,095 | 16.5353% |
| WI_3 | 194,100 | 35,000 | 18.0319% |

### Missing prices by selected product

| Item | Rows | Missing | Missing percentage |
| --- | --- | --- | --- |
| FOODS_2_209 | 5,823 | 4,130 | 70.9256% |
| FOODS_3_601 | 5,823 | 3,787 | 65.0352% |
| FOODS_2_040 | 5,823 | 3,647 | 62.6309% |
| FOODS_3_661 | 5,823 | 3,549 | 60.9480% |
| FOODS_2_178 | 5,823 | 3,493 | 59.9863% |
| FOODS_2_095 | 5,823 | 3,402 | 58.4235% |
| FOODS_3_340 | 5,823 | 3,318 | 56.9809% |
| FOODS_3_482 | 5,823 | 3,206 | 55.0575% |
| FOODS_2_301 | 5,823 | 3,073 | 52.7735% |
| FOODS_3_641 | 5,823 | 2,905 | 49.8884% |
| FOODS_3_532 | 5,823 | 2,884 | 49.5277% |
| FOODS_3_795 | 5,823 | 2,884 | 49.5277% |
| FOODS_3_709 | 5,823 | 2,821 | 48.4458% |
| FOODS_2_028 | 5,823 | 2,793 | 47.9650% |
| FOODS_3_135 | 5,823 | 2,730 | 46.8830% |
| FOODS_3_354 | 5,823 | 2,702 | 46.4022% |
| FOODS_1_089 | 5,823 | 2,646 | 45.4405% |
| FOODS_3_658 | 5,823 | 2,464 | 42.3150% |
| FOODS_3_640 | 5,823 | 2,436 | 41.8341% |
| FOODS_3_312 | 5,823 | 2,429 | 41.7139% |
| FOODS_3_053 | 5,823 | 2,422 | 41.5937% |
| FOODS_1_099 | 5,823 | 2,359 | 40.5118% |
| FOODS_1_082 | 5,823 | 2,310 | 39.6703% |
| FOODS_1_210 | 5,823 | 2,303 | 39.5501% |
| FOODS_3_074 | 5,823 | 2,303 | 39.5501% |
| FOODS_2_180 | 5,823 | 2,212 | 37.9873% |
| FOODS_3_240 | 5,823 | 2,149 | 36.9054% |
| FOODS_3_550 | 5,823 | 2,142 | 36.7852% |
| FOODS_3_329 | 5,823 | 2,023 | 34.7415% |
| FOODS_1_155 | 5,823 | 1,939 | 33.2990% |
| FOODS_1_042 | 5,823 | 1,932 | 33.1788% |
| FOODS_3_321 | 5,823 | 1,652 | 28.3703% |
| FOODS_3_048 | 5,823 | 1,260 | 21.6383% |
| FOODS_1_017 | 5,823 | 1,225 | 21.0373% |
| FOODS_1_156 | 5,823 | 1,211 | 20.7968% |
| FOODS_1_175 | 5,823 | 1,211 | 20.7968% |
| FOODS_3_539 | 5,823 | 1,064 | 18.2724% |
| FOODS_3_179 | 5,823 | 1,050 | 18.0319% |
| FOODS_3_223 | 5,823 | 1,008 | 17.3107% |
| FOODS_3_801 | 5,823 | 924 | 15.8681% |
| FOODS_2_336 | 5,823 | 889 | 15.2670% |
| FOODS_3_263 | 5,823 | 833 | 14.3053% |
| FOODS_3_511 | 5,823 | 714 | 12.2617% |
| FOODS_3_327 | 5,823 | 574 | 9.8575% |
| FOODS_2_136 | 5,823 | 455 | 7.8138% |
| FOODS_2_184 | 5,823 | 427 | 7.3330% |
| FOODS_2_302 | 5,823 | 420 | 7.2128% |
| FOODS_2_170 | 5,823 | 329 | 5.6500% |
| FOODS_3_068 | 5,823 | 329 | 5.6500% |
| FOODS_1_214 | 5,823 | 217 | 3.7266% |
| FOODS_3_479 | 5,823 | 140 | 2.4043% |
| FOODS_3_648 | 5,823 | 140 | 2.4043% |
| FOODS_1_086 | 5,823 | 70 | 1.2021% |
| FOODS_3_635 | 5,823 | 70 | 1.2021% |
| FOODS_2_367 | 5,823 | 49 | 0.8415% |
| FOODS_3_360 | 5,823 | 49 | 0.8415% |
| FOODS_2_339 | 5,823 | 21 | 0.3606% |
| FOODS_3_099 | 5,823 | 21 | 0.3606% |
| FOODS_3_026 | 5,823 | 14 | 0.2404% |
| FOODS_3_383 | 5,823 | 14 | 0.2404% |
| FOODS_3_470 | 5,823 | 14 | 0.2404% |
| FOODS_3_724 | 5,823 | 7 | 0.1202% |
| FOODS_1_020 | 5,823 | 0 | 0.0000% |
| FOODS_1_112 | 5,823 | 0 | 0.0000% |
| FOODS_1_135 | 5,823 | 0 | 0.0000% |
| FOODS_1_172 | 5,823 | 0 | 0.0000% |
| FOODS_2_021 | 5,823 | 0 | 0.0000% |
| FOODS_2_024 | 5,823 | 0 | 0.0000% |
| FOODS_2_059 | 5,823 | 0 | 0.0000% |
| FOODS_2_072 | 5,823 | 0 | 0.0000% |
| FOODS_2_116 | 5,823 | 0 | 0.0000% |
| FOODS_2_139 | 5,823 | 0 | 0.0000% |
| FOODS_2_146 | 5,823 | 0 | 0.0000% |
| FOODS_2_150 | 5,823 | 0 | 0.0000% |
| FOODS_2_218 | 5,823 | 0 | 0.0000% |
| FOODS_2_251 | 5,823 | 0 | 0.0000% |
| FOODS_2_304 | 5,823 | 0 | 0.0000% |
| FOODS_2_322 | 5,823 | 0 | 0.0000% |
| FOODS_2_357 | 5,823 | 0 | 0.0000% |
| FOODS_2_360 | 5,823 | 0 | 0.0000% |
| FOODS_3_089 | 5,823 | 0 | 0.0000% |
| FOODS_3_096 | 5,823 | 0 | 0.0000% |
| FOODS_3_113 | 5,823 | 0 | 0.0000% |
| FOODS_3_123 | 5,823 | 0 | 0.0000% |
| FOODS_3_187 | 5,823 | 0 | 0.0000% |
| FOODS_3_249 | 5,823 | 0 | 0.0000% |
| FOODS_3_276 | 5,823 | 0 | 0.0000% |
| FOODS_3_319 | 5,823 | 0 | 0.0000% |
| FOODS_3_387 | 5,823 | 0 | 0.0000% |
| FOODS_3_432 | 5,823 | 0 | 0.0000% |
| FOODS_3_467 | 5,823 | 0 | 0.0000% |
| FOODS_3_480 | 5,823 | 0 | 0.0000% |
| FOODS_3_501 | 5,823 | 0 | 0.0000% |
| FOODS_3_520 | 5,823 | 0 | 0.0000% |
| FOODS_3_558 | 5,823 | 0 | 0.0000% |
| FOODS_3_611 | 5,823 | 0 | 0.0000% |
| FOODS_3_616 | 5,823 | 0 | 0.0000% |
| FOODS_3_666 | 5,823 | 0 | 0.0000% |
| FOODS_3_677 | 5,823 | 0 | 0.0000% |
| FOODS_3_817 | 5,823 | 0 | 0.0000% |

Missing prices remain `NaN`. Replacing an unknown price with zero would falsely imply the product was free, while forward-fill or mean imputation would introduce assumptions that require a later cleaning decision.

## Validation

| Integrity check | Result |
| --- | --- |
| exactly_100_items | Passed |
| exact_frozen_stores | Passed |
| foods_only | Passed |
| department_quotas | Passed |
| demand_band_quotas | Passed |
| exactly_300_item_store_pairs | Passed |
| expected_rows | Passed |
| complete_day_history_per_pair | Passed |
| unique_date_store_item_key | Passed |
| no_negative_sales | Passed |
| all_calendar_rows_matched | Passed |

Calendar join: 582,300 rows before and 582,300 after; 0 unmatched; multiplier 1.0.

Price join: 582,300 rows before and 582,300 after; 0 duplicate source keys; multiplier 1.0.

Raw source SHA-256 hashes were equal before and after execution: **True**.

## Decisions and Trade-offs

- **Filter before melting:** avoids a needless 59.2-million-row intermediate and keeps laptop memory use controlled.
- **Stratified selection:** gives all FOODS departments and all demand bands explicit representation; pure high-volume sampling would be biased.
- **Fixed seed 42:** pseudorandom selection remains reproducible while avoiding manual cherry-picking.
- **Composite price key:** week alone repeats across all products and stores. `(store_id, item_id, wm_yr_wk)` identifies the intended weekly price without multiplying rows.
- **Missing prices preserved:** zero is a real price value with a different meaning from unknown; no unsupported imputation was introduced.
- **Raw data immutable:** transformations can always be reproduced or corrected because official source files remain unchanged.
- **Safe dtype reduction:** repeated identifiers and labels used pandas categorical dtype in memory, reducing the final working DataFrame to 22.34 MiB. Their CSV values remain ordinary text.

## Scope Confirmation

No EDA, database work, feature engineering, price-change or promotion feature creation, holiday features, train/test splitting, modeling, forecasting, inventory logic, simulation, dashboard, Docker, or deployment work was performed.
