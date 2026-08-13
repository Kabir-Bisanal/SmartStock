# SmartStock Stage 3 — M5 Data Understanding & Profiling

Generated from the local raw M5 files at `2026-08-13T01:40:02+00:00`.

> This is a structural and data-quality profile, not an exploratory analysis, cleaning pipeline, final subset, or production join. The raw files were read but not modified.

## Executive Summary

The sales table contains **30,490 item-store rows**, **1,941 daily columns**, **3,049 items**, and **10 stores**. Its day columns cover `d_1` through `d_1941`. The price table contains **6,841,121 weekly item-store price records** across **282 weeks**. The calendar maps day labels and weeks to **1,969 dates** from **2011-01-29 through 2016-06-19**.

All 1,941 sales day labels match calendar day keys, all 282 price weeks match calendar weeks, and all 30,490 sales item-store combinations have at least one price record. The composite price key `(store_id, item_id, wm_yr_wk)` is unique.

Demand is strongly zero-heavy: **68.00%** of daily cells contain zero. This is expected retail behavior, not automatically missing data. The report proposes three subset strategies but deliberately does not select the final stores or products.

## Sales Dataset

### Structure and cardinality

| Measure | Calculated value |
| --- | --- |
| File | sales_train_evaluation.csv |
| CSV file size (MiB) | 116.10 |
| Rows | 30,490 |
| Columns | 1,947 |
| Identifier columns | id, item_id, dept_id, cat_id, store_id, state_id |
| Daily sales columns | 1,941 |
| First / last daily column | d_1 / d_1941 |
| Daily sequence contiguous | Yes |
| Unique items | 3,049 |
| Unique stores | 10 |
| Unique states | 3 |
| Unique categories | 3 |
| Unique departments | 7 |
| Unique item-store combinations | 30,490 |
| `id` unique | Yes |
| `(item_id, store_id)` unique | Yes |
| Default-inferred daily dtype | int64 |
| Estimated default DataFrame memory (MiB) | 455.26 |
| Estimated int32 DataFrame memory (MiB) | 229.50 |

**Wide format** means that each item-store series is one row and each date is a separate column. This makes the raw file compact on disk but awkward for normal relational joins and potentially expensive to reshape.

**Cardinality** means the number of distinct values in a field or key. A **composite key** uses more than one column; here `(item_id, store_id)` uniquely identifies each sales series.

### Identifier missing values

| Identifier | Count |
| --- | --- |
| id | 0 |
| item_id | 0 |
| dept_id | 0 |
| cat_id | 0 |
| store_id | 0 |
| state_id | 0 |

### Category counts

| Category | Sales rows | Unique items |
| --- | --- | --- |
| HOBBIES | 5,650 | 565 |
| HOUSEHOLD | 10,470 | 1,047 |
| FOODS | 14,370 | 1,437 |

### Department counts

| Department | Sales rows | Unique items |
| --- | --- | --- |
| HOBBIES_1 | 4,160 | 416 |
| HOBBIES_2 | 1,490 | 149 |
| HOUSEHOLD_1 | 5,320 | 532 |
| HOUSEHOLD_2 | 5,150 | 515 |
| FOODS_1 | 2,160 | 216 |
| FOODS_2 | 3,980 | 398 |
| FOODS_3 | 8,230 | 823 |

### Stores and states

| Store | Item-store rows |
| --- | --- |
| CA_1 | 3,049 |
| CA_2 | 3,049 |
| CA_3 | 3,049 |
| CA_4 | 3,049 |
| TX_1 | 3,049 |
| TX_2 | 3,049 |
| TX_3 | 3,049 |
| WI_1 | 3,049 |
| WI_2 | 3,049 |
| WI_3 | 3,049 |

| State | Count |
| --- | --- |
| CA | 12,196 |
| TX | 9,147 |
| WI | 9,147 |

### Daily sales-value validation

| Measure | Calculated value |
| --- | --- |
| Daily cells scanned | 59,181,090 |
| Missing daily values | 0 |
| Negative sales values | 0 |
| Zero sales values | 40,241,819 |
| Zero prevalence | 67.9978% |
| Minimum | 0 |
| Maximum | 763 |
| Profiler scan dtype | int32 |
| Items at least 95% zero | 72 |
| FOODS items at least 95% zero | 8 |

The CSV has no stored datatype; pandas normally infers these integer-valued day columns as `int64`. The profiler reads them as `int32` in chunks after validating their observed range. This reduces temporary memory without changing values.

### Store-level sales structure

| Store | State | Items | Total units | Zero prevalence |
| --- | --- | --- | --- | --- |
| CA_1 | CA | 3,049 | 7,832,248 | 63.76% |
| CA_2 | CA | 3,049 | 5,818,395 | 68.81% |
| CA_3 | CA | 3,049 | 11,363,540 | 59.40% |
| CA_4 | CA | 3,049 | 4,182,534 | 72.00% |
| TX_1 | TX | 3,049 | 5,692,823 | 70.71% |
| TX_2 | TX | 3,049 | 7,329,642 | 66.31% |
| TX_3 | TX | 3,049 | 6,205,940 | 69.72% |
| WI_1 | WI | 3,049 | 5,261,506 | 68.75% |
| WI_2 | WI | 3,049 | 6,697,988 | 70.48% |
| WI_3 | WI | 3,049 | 6,542,557 | 70.03% |

## Price Dataset

| Measure | Calculated value |
| --- | --- |
| File | sell_prices.csv |
| CSV file size (MiB) | 193.97 |
| Rows | 6,841,121 |
| Columns | 4 |
| Optimized DataFrame memory (MiB) | 72.01 |
| Unique stores | 10 |
| Unique items | 3,049 |
| Unique item-store pairs | 30,490 |
| Unique composite price records | 6,841,121 |
| Unique weeks | 282 |
| Earliest / latest week | 11101 / 11621 |
| Minimum price | 0.01 |
| Maximum price | 107.32 |
| Mean price | 4.41 |
| Median price | 3.47 |
| Negative prices | 0 |
| Zero prices | 0 |
| Duplicate rows | 0 |
| Duplicate composite keys | 0 |
| Possible raw item-store-week positions | 8,598,180 |
| Missing raw item-store-week positions | 1,757,059 |
| Overall raw week-grid coverage | 79.56% |

### Column datatypes

| Column | Default inference | Profiler dtype |
| --- | --- | --- |
| store_id | str | category |
| item_id | str | category |
| wm_yr_wk | int64 | int32 |
| sell_price | float64 | float32 |

### Missing values

| Column | Count |
| --- | --- |
| store_id | 0 |
| item_id | 0 |
| wm_yr_wk | 0 |
| sell_price | 0 |

### Price coverage by store

| Store | Records | Items | Weeks | Median weeks/item | Raw week-grid coverage |
| --- | --- | --- | --- | --- | --- |
| CA_1 | 698,412 | 3,049 | 282 | 273.00 | 81.23% |
| CA_2 | 637,395 | 3,049 | 282 | 239.00 | 74.13% |
| CA_3 | 693,990 | 3,049 | 282 | 271.00 | 80.71% |
| CA_4 | 679,025 | 3,049 | 282 | 256.00 | 78.97% |
| TX_1 | 699,796 | 3,049 | 282 | 271.00 | 81.39% |
| TX_2 | 701,214 | 3,049 | 282 | 273.00 | 81.55% |
| TX_3 | 691,112 | 3,049 | 282 | 269.00 | 80.38% |
| WI_1 | 665,912 | 3,049 | 282 | 225.00 | 77.45% |
| WI_2 | 678,171 | 3,049 | 282 | 231.00 | 78.87% |
| WI_3 | 696,094 | 3,049 | 282 | 271.00 | 80.96% |

**Raw week-grid coverage** divides observed price rows by every possible item × calendar-week position for the relevant group. It is intentionally conservative: a pre-launch week is counted as uncovered, even when the absence is legitimate.

### Price coverage by category

| Category | Records | Items | Stores | Weeks | Raw week-grid coverage |
| --- | --- | --- | --- | --- | --- |
| FOODS | 3,181,789 | 1,437 | 10 | 282 | 78.52% |
| HOBBIES | 1,283,905 | 565 | 10 | 282 | 80.58% |
| HOUSEHOLD | 2,375,427 | 1,047 | 10 | 282 | 80.45% |

### Distinct prices per item

| Measure | Calculated value |
| --- | --- |
| Minimum | 1 |
| Median | 5.00 |
| Mean | 5.64 |
| Maximum | 52 |
| Items with one distinct price | 316 |
| Items with price changes | 2,733 |

## Calendar Dataset

| Measure | Calculated value |
| --- | --- |
| Rows | 1,969 |
| Columns | 14 |
| Memory (MiB) | 0.15 |
| Date range | 2011-01-29 to 2016-06-19 |
| Unique `d` values | 1,969 |
| `d` is unique | Yes |
| Unique weeks | 282 |
| Week range | 11101 to 11621 |
| Years | 2011, 2012, 2013, 2014, 2015, 2016 |
| Months | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 |
| Weekdays | Saturday, Sunday, Monday, Tuesday, Wednesday, Thursday, Friday |
| Days with at least one event | 162 |
| Days with two events | 5 |

### Missing values

| Column | Count |
| --- | --- |
| date | 0 |
| wm_yr_wk | 0 |
| weekday | 0 |
| wday | 0 |
| month | 0 |
| year | 0 |
| d | 0 |
| event_name_1 | 1,807 |
| event_type_1 | 1,807 |
| event_name_2 | 1,964 |
| event_type_2 | 1,964 |
| snap_CA | 0 |
| snap_TX | 0 |
| snap_WI | 0 |

### Event-field value counts

#### `event_name_1`

| Value | Count |
| --- | --- |
| Chanukah End | 5 |
| Christmas | 5 |
| Cinco De Mayo | 5 |
| ColumbusDay | 5 |
| Easter | 5 |
| Eid al-Fitr | 5 |
| EidAlAdha | 5 |
| Father's day | 4 |
| Halloween | 5 |
| IndependenceDay | 5 |
| LaborDay | 5 |
| LentStart | 6 |
| LentWeek2 | 6 |
| MartinLutherKingDay | 5 |
| MemorialDay | 6 |
| Mother's day | 6 |
| NBAFinalsEnd | 6 |
| NBAFinalsStart | 6 |
| NewYear | 5 |
| OrthodoxChristmas | 5 |
| OrthodoxEaster | 5 |
| Pesach End | 6 |
| PresidentsDay | 6 |
| Purim End | 6 |
| Ramadan starts | 6 |
| StPatricksDay | 6 |
| SuperBowl | 6 |
| Thanksgiving | 5 |
| ValentinesDay | 6 |
| VeteransDay | 5 |

#### `event_type_1`

| Value | Count |
| --- | --- |
| Cultural | 37 |
| National | 52 |
| Religious | 55 |
| Sporting | 18 |

#### `event_name_2`

| Value | Count |
| --- | --- |
| Cinco De Mayo | 1 |
| Easter | 1 |
| Father's day | 2 |
| OrthodoxEaster | 1 |

#### `event_type_2`

| Value | Count |
| --- | --- |
| Cultural | 4 |
| Religious | 1 |

### Combined event-type distribution

| Event type | Count |
| --- | --- |
| Cultural | 41 |
| National | 52 |
| Religious | 56 |
| Sporting | 18 |

### SNAP flag coverage

| State | SNAP-enabled days | Calendar coverage |
| --- | --- | --- |
| California | 650 | 33.01% |
| Texas | 650 | 33.01% |
| Wisconsin | 650 | 33.01% |

## Retail Hierarchy

The real data demonstrates two linked dimensions rather than one single chain:

```text
Geography: state_id -> store_id
Product:   cat_id -> dept_id -> item_id
Sales:     store_id × item_id × daily columns
```

An item does not belong to a store in the product hierarchy; the sales table crosses the product dimension with the store dimension.

| Verified mapping | Result |
| --- | --- |
| Each store maps to one state | Yes |
| Each department maps to one category | Yes |
| Each item maps to one department | Yes |
| Each item maps to one category | Yes |

Stores per state:

| State | Count |
| --- | --- |
| CA | 4 |
| TX | 3 |
| WI | 3 |

## FOODS Category Analysis

| Measure | Calculated value |
| --- | --- |
| Unique FOODS items | 1,437 |
| FOODS departments | FOODS_1, FOODS_2, FOODS_3 |
| FOODS items appearing in multiple stores | 1,437 |
| FOODS items appearing in every store | 1,437 |
| Minimum stores per FOODS item | 10 |
| Maximum stores per FOODS item | 10 |
| FOODS items first positive after day 365 | 525 |
| FOODS items never positive | 0 |

### FOODS products per department

| Department | Count |
| --- | --- |
| FOODS_1 | 216 |
| FOODS_2 | 398 |
| FOODS_3 | 823 |

### FOODS product availability by store

| Store | Count |
| --- | --- |
| CA_1 | 1,437 |
| CA_2 | 1,437 |
| CA_3 | 1,437 |
| CA_4 | 1,437 |
| TX_1 | 1,437 |
| TX_2 | 1,437 |
| TX_3 | 1,437 |
| WI_1 | 1,437 |
| WI_2 | 1,437 |
| WI_3 | 1,437 |

Every FOODS item has a sales row in every store. This verifies structural availability in the sales matrix, but it does not prove continuous selling or price coverage. Some item-store series start selling late or contain mostly zeros.

### Highest-zero FOODS items

| Item | Zero prevalence | First positive day | Total units |
| --- | --- | --- | --- |
| FOODS_3_220 | 96.18% | 1,009 | 885 |
| FOODS_3_472 | 95.80% | 1,441 | 1,183 |
| FOODS_3_296 | 95.78% | 1,301 | 1,797 |
| FOODS_2_071 | 95.38% | 1,147 | 1,229 |
| FOODS_2_073 | 95.37% | 1,233 | 1,127 |
| FOODS_1_079 | 95.31% | 6 | 1,734 |
| FOODS_3_171 | 95.05% | 1,176 | 1,142 |
| FOODS_2_252 | 95.03% | 1,408 | 1,429 |
| FOODS_3_260 | 94.87% | 1,424 | 1,558 |
| FOODS_3_350 | 94.68% | 1,106 | 1,805 |

## Dataset Relationships

### Sales day columns to calendar

```text
sales.d_x <-> calendar.d -> calendar.date and calendar.wm_yr_wk
```

| Check | Result |
| --- | --- |
| Sales day columns | 1,941 |
| Matched sales day labels | 1,941 |
| Unmatched sales day labels | 0 |
| Calendar labels beyond sales | 28 |
| Observed sales date range | 2011-01-29 to 2016-05-22 |
| Cardinality | one-to-one for day labels; many sales observations to one calendar day after reshaping |

### Weekly prices to calendar

```text
prices.wm_yr_wk <-> calendar.wm_yr_wk
```

| Check | Result |
| --- | --- |
| Price weeks | 282 |
| Matched price weeks | 282 |
| Unmatched price weeks | 0 |
| Calendar weeks without prices | 0 |
| Calendar days per week | 2 to 7 |
| Cardinality | many-to-many on wm_yr_wk alone; one weekly item-store price expands to multiple calendar days |

Joining prices directly to calendar on week alone is technically many-to-many because both tables repeat the week. The safe later sequence is to map each sales day to its week, then join one price row using `(store_id, item_id, wm_yr_wk)`.

### Sales identifiers to prices

```text
sales.(store_id, item_id) <-> prices.(store_id, item_id)
prices unique row key: (store_id, item_id, wm_yr_wk)
```

| Check | Result |
| --- | --- |
| Matched items | 3,049 |
| Sales items without prices | 0 |
| Price items without sales | 0 |
| Matched stores | 10 |
| Sales stores without prices | 0 |
| Price stores without sales | 0 |
| Matched item-store pairs | 30,490 |
| Sales pairs without prices | 0 |
| Price pairs without sales | 0 |
| Composite price key unique | Yes |
| Cardinality | one sales item-store row to many weekly price rows; after adding week to daily sales, many sales rows to one composite price key |

## Data Quality Observations

| Classification | Observation | Why it matters |
| --- | --- | --- |
| Expected Dataset Behavior | Daily demand is zero-heavy: 68.00% of sales cells are zero. | Zero usually means no recorded unit sales, not missing data; later modeling must preserve it. |
| Expected Dataset Behavior | Event fields are sparse: event_name_1 is null on 1,807 of 1,969 days and event_name_2 is null on 1,964 days. | Null event fields normally mean no named event and should not be filled blindly. |
| Expected Dataset Behavior | The price table covers 79.56% of the raw item-store-week grid, leaving 1,757,059 positions without a price row; store coverage varies because items enter or leave assortment over time. | A missing weekly price may be pre-launch or out-of-assortment behavior, not a defective record. |
| Requires Investigation | 525 FOODS items first record positive demand after day 365 across the full store set. | Product introduction timing should influence the final subset and later train/test design. |
| Requires Investigation | 2,733 items have more than one distinct observed price. | Price changes may later support promotion/price features, but no promotion effect is inferred in Stage 3. |
| Requires Investigation | The calendar contains 28 day labels beyond the evaluation sales horizon. | Later joins must distinguish the observed sales period from future calendar rows. |
| Requires Investigation | The wide sales table has 1,941 day columns and an estimated default pandas footprint of about 455.3 MiB. | Later reshaping can multiply row count dramatically and needs an explicit memory-aware design. |
| Expected Dataset Behavior | Negative sales values found: 0. | No correction was made; any nonzero result would require source investigation. |
| Expected Dataset Behavior | Duplicate (store_id, item_id, wm_yr_wk) price keys found: 0. | The future daily sales-to-price join is safe only when this composite key is unique. |

## Version 1 Subset Recommendations

These are candidate strategies, not a final subset. Product-screen counts use transparent diagnostics: a product must have a sales row in all three stores; ‘active by day 365’ means positive demand occurred in each store within the first year; ‘below 95% zero’ is checked in each store; and 80% price coverage is measured against all calendar weeks. That price denominator penalizes legitimate late introductions, so it is a conservative screen rather than a cleaning rule.

Selecting only the highest-volume products would bias the project toward fast movers and make results less representative. A final ~100-item subset should therefore be stratified across `FOODS_1`, `FOODS_2`, and `FOODS_3`, and ideally across demand/coverage bands rather than volume alone.

### A. Cross-state representative stores

Candidate stores: **CA_1, TX_2, WI_3**

Rationale: Uses the best raw price-coverage store from each state, adding geographic and SNAP-policy diversity.

Trade-off: State effects become mixed with store effects, so direct store comparisons are less controlled.

| Store | State | Raw price coverage | Sales zero prevalence |
| --- | --- | --- | --- |
| CA_1 | CA | 81.23% | 63.76% |
| TX_2 | TX | 81.55% | 66.31% |
| WI_3 | WI | 80.96% | 70.03% |

Indicative FOODS product screen:

| Screen | Items passing |
| --- | --- |
| Present in all three sales stores | 1,437 |
| Any price in all three stores | 1,437 |
| At least 80% raw price-week coverage in every store | 864 |
| Active by day 365 in every store | 842 |
| Below 95% zero in every store | 1,378 |
| All indicative screens | 840 |

### B. Same-state comparable stores (TX)

Candidate stores: **TX_2, TX_1, TX_3**

Rationale: Keeps state-level conditions and the state-specific SNAP flag constant while comparing stores.

Trade-off: It reduces geographic diversity and may not generalize as well to the other states.

| Store | State | Raw price coverage | Sales zero prevalence |
| --- | --- | --- | --- |
| TX_1 | TX | 81.39% | 70.71% |
| TX_2 | TX | 81.55% | 66.31% |
| TX_3 | TX | 80.38% | 69.72% |

Indicative FOODS product screen:

| Screen | Items passing |
| --- | --- |
| Present in all three sales stores | 1,437 |
| Any price in all three stores | 1,437 |
| At least 80% raw price-week coverage in every store | 862 |
| Active by day 365 in every store | 833 |
| Below 95% zero in every store | 1,352 |
| All indicative screens | 825 |

### C. Highest raw price-coverage stores

Candidate stores: **TX_2, TX_1, CA_1**

Rationale: Prioritizes complete weekly price histories and therefore simpler later joins.

Trade-off: Coverage-driven store selection can overrepresent one geography and is not a business-representative sample by itself.

| Store | State | Raw price coverage | Sales zero prevalence |
| --- | --- | --- | --- |
| CA_1 | CA | 81.23% | 63.76% |
| TX_1 | TX | 81.39% | 70.71% |
| TX_2 | TX | 81.55% | 66.31% |

Indicative FOODS product screen:

| Screen | Items passing |
| --- | --- |
| Present in all three sales stores | 1,437 |
| Any price in all three stores | 1,437 |
| At least 80% raw price-week coverage in every store | 885 |
| Active by day 365 in every store | 859 |
| Below 95% zero in every store | 1,375 |
| All indicative screens | 857 |

## Performance and Memory Decisions

- Avoided a full sales melt. Converting 30,490 rows × 1,941 days to long format would create 59,181,090 sales rows before calendar or price columns are added.
- Loaded only the six sales identifier columns as one small DataFrame, then scanned daily columns once in chunks of 1,000 rows.
- Used `int32` for the sales scan after checking the observed value range, rather than pandas' larger default `int64` inference.
- Loaded the price table once with categorical identifiers, `int32` weeks, and `float32` prices; aggregate tables were calculated from that copy.
- Did not calculate expensive per-day trends, correlations, promotion effects, or full descriptive statistics because they belong to EDA and add little structural evidence here.

## Risks / Things to Investigate Later

- Decide whether price absence before first sale represents product launch timing, store assortment timing, or another M5 convention before imputing anything.
- Define the final subset's product coverage rule and department allocation with ChatGPT before extraction.
- Preserve chronological ordering and product launch timing during later train/test design.
- Design the Stage 4 wide-to-long process in bounded chunks; a naive melt would create tens of millions of rows.
- Treat calendar days beyond `d_1941` as future-horizon support, not missing sales rows.
- Verify how SNAP flags should be interpreted for each store state before feature engineering.

## Scope Confirmation

No raw data was cleaned or modified. No permanent wide-to-long transformation, final subset, PostgreSQL work, SQL schema, production ETL, feature engineering, modeling, inventory optimization, simulation, dashboard, Docker, or deployment work was performed.
