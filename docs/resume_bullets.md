# Resume bullets

- Built an end-to-end retail demand forecasting and inventory optimization system in Python/Pandas over 582K daily observations across 300 product-store time series, including reproducible ETL, feature engineering, and chronological leakage-safe validation.

- Evaluated statistical baselines and global scikit-learn/XGBoost models over recursive 1-, 7-, and 30-day horizons; selected a 28-day historical mean through balanced validation and reported a one-time locked-test MAE of 1.382 and WAPE of 61.20%.

- Developed a reusable safety-stock, reorder-point, stockout-risk, and scenario-based ordering engine, then delivered it through PostgreSQL, SQLAlchemy, Streamlit, a read-only CSV fallback, automated tests, and Docker Compose; explicitly separated synthetic inventory/cost assumptions from Walmart M5 data.
