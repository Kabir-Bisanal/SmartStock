# SmartStock interview guide

## 30-second explanation

SmartStock is an end-to-end retail analytics portfolio system. I transformed Walmart M5 data into 300 product-store demand series, compared forecasting methods with leakage-safe chronological validation, selected a 28-day mean because it beat more complex models on balanced validation, and converted its forecast into safety stock and reorder decisions. PostgreSQL and Streamlit expose the result, while all unavailable inventory and cost inputs are clearly labeled synthetic.

## 2-minute explanation

I scoped M5 to 100 FOOD items across one California, Texas, and Wisconsin store, retaining 582,300 daily rows. I handled product availability explicitly, kept valid zero demand, engineered only past-known demand features, and froze three expanding validation folds plus a locked final test. Five baselines, Ridge, HistGradientBoosting, and XGBoost were evaluated from a single forecast origin for each 30-day path. The 28-day mean had the best balanced validation result, so I froze it before evaluating the test once. I then used validation residuals to estimate uncertainty and combined a 30-day forecast with synthetic on-hand, lead-time, service, and cost values to demonstrate reorder decisions. The application supports PostgreSQL and a visible read-only CSV mode, with tests and deployment configuration around the full flow.

## Architecture walkthrough

Raw immutable CSVs flow through validation, subset selection, long-format ETL, EDA, features, model evaluation, a production forecast, inventory optimization, PostgreSQL, and Streamlit. Synthetic inventory state joins only at the inventory layer. The UI queries PostgreSQL or the same generated artifacts through a CSV adapter.

## Why M5?

M5 offers realistic daily item/store demand, calendar events, and weekly prices with a clear retail hierarchy. It supports both forecasting and data-engineering discussion, while its gaps force honest assumptions for inventory planning.

## Why only 100 products?

V1 deliberately balances realism and iteration speed. The deterministic stratified sample includes low-, medium-, and high-demand products across all FOOD departments while keeping 30-day multi-model evaluation practical.

## Why 3 stores?

`CA_1`, `TX_2`, and `WI_3` represent one store in each M5 state. Every selected item exists in all three, yielding a clean 300-series comparison without claiming national coverage.

## Why mean_28 won?

It achieved the best balanced validation rank across error metrics, horizons, aggregates, bias, and fold stability. It smooths noisy intermittent demand and does not accumulate unstable recursive feature errors.

## Why didn't you simply choose XGBoost?

Complexity is not a selection metric. XGBoost improved some spike-sensitive measures but did not consistently beat the mean across folds and the full 30-day inventory horizon. Choosing it anyway would be validation shopping.

## What is data leakage?

Leakage occurs when information unavailable at forecast time influences training or prediction. Examples are random time splits, target-day sales inside rolling features, future price observations, or using validation Day +1 actual sales to forecast Day +2.

## How did recursive forecasting work?

At origin `T`, only actuals through `T` are available. The system predicts Day +1, appends that prediction—not the actual—to synthetic history, rebuilds lags/rolls, and repeats through Day +30.

## Why are inventory levels synthetic?

M5 does not publish on-hand balances, purchase orders, backorders, lead times, service levels, or costs. I generated deterministic demo inputs and labeled them explicitly rather than pretending they were Walmart facts.

## What is safety stock?

Safety stock is extra inventory above expected lead-time demand to absorb forecast uncertainty. V1 uses a chosen service-level normal quantile times validation residual standard deviation times the square root of lead time.

## What is reorder point?

The reorder point is lead-time forecast demand plus safety stock. When inventory position falls below it, replenishment should be considered.

## How did cost optimization work?

For each series, the engine simulates demand scenarios around the forecast and compares a bounded set of order quantities using demo fixed-order, holding, and shortage costs. It is illustrative, not a solver for actual Walmart economics.

## Why PostgreSQL?

It gives the application durable relational tables, enforced keys, indexed filters, idempotent loading, and a realistic separation between analytics artifacts and serving queries.

## Why Streamlit?

Streamlit made it possible to build a clear interactive decision interface in Python while reusing the tested inventory and data-access layers instead of duplicating business logic in a separate frontend.

## Biggest challenge

The hardest part was preserving forecast-origin truth across a 30-day horizon. Precomputed one-step lags are safe for Day +1 but leak later validation actuals unless the future path is rebuilt recursively.

## Main limitations

The subset is small, sales can be censored by stockouts, the forecaster excludes future price/promotion information, uncertainty uses an approximate normal model, inventory fields are synthetic, and V1 has no authentication, monitoring, or live cloud deployment.

## What would V2 improve?

I would add probabilistic/direct multi-horizon forecasts, known price and promotion scenarios, real ERP inventory and supplier lead times, MOQ/case-pack/shelf-life constraints, scheduled retraining, authentication, and production monitoring.
