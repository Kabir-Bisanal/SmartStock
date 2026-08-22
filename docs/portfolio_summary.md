# SmartStock portfolio summary

**Problem:** Forecast daily retail demand and translate it into defensible inventory/reorder decisions.

**Scale:** 582,300 historical rows covering 100 FOOD products, three stores, and 300 product-store series from Walmart M5.

**Engineering:** Built reproducible validation, profiling, ETL, feature engineering, PostgreSQL loading/query layers, a four-section decision-focused Streamlit application, a read-only CSV fallback, automated checks, and Docker Compose configuration.

**Data science:** Used active-period handling, chronological validation, leakage-safe 30-day recursive simulation, baseline forecasting, and global Ridge/HistGradientBoosting/XGBoost comparisons. A simple 28-day historical mean won the balanced validation ranking—an honest example of selecting evidence over complexity. Its locked-test Days 1–30 results were MAE 1.382, RMSE 2.387, WAPE 61.20%, and RMSSE 0.764.

**Business layer:** Converted daily forecasts and validation-only uncertainty into safety stock, reorder points, stockout risk, service-level recommendations, priorities, and an illustrative cost-aware ordering scenario.

**Disclosure:** M5 contains no actual inventory state, lead times, service levels, or costs. Those inputs and all savings figures are deterministic synthetic demo assumptions, not Walmart results.

**Delivery:** Runs locally with PostgreSQL or the visible CSV demo fallback and is Docker-ready. No live cloud deployment is claimed.
