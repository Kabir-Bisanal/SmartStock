# SmartStock V1 pipeline

This document is the shortest route for reverse-engineering how the finished system was built. Detailed calculated evidence remains in the stage reports under `reports/`.

| Phase | Purpose and inputs | Main outputs | Important decision |
|---|---|---|---|
| Stages 1–2: foundation | Define the retail problem and acquire official M5 sales, calendar, and price files. | Project layout, dependencies, raw-data validator. | Raw files are immutable and excluded from Git. |
| Stage 3: understanding | Profile the full M5 tables and their join keys without full reshaping. | Data profile and machine summary. | Zero sales are valid intermittent demand, not missing data. |
| Stage 4: V1 subset | Screen FOOD products, stratify demand, and select 100 items across `CA_1`, `TX_2`, and `WI_3`. | Frozen subset manifest and 582,300-row long dataset. | Filter 300 wide rows before melting; fixed seed makes selection repeatable. |
| Stage 5: EDA | Describe active demand, intermittency, seasonality, prices, and events. | EDA report and figures. | Descriptive associations are not causal effects. |
| Stage 6: features | Create target-safe calendar, lifecycle, lag, rolling, zero-rate, and historical price features. | 480,506 active rows, 472,106 feature-ready rows, and frozen fold manifests. | Pre-launch rows are excluded; active zeros remain; every demand window is shifted. |
| Stage 7: baselines | Compare zero, persistence, seasonal-naive, 28-day mean, and Croston-SBA. | Validation predictions, metrics, and leakage audit. | All 30 days are forecast from one origin; later validation actuals never update history. |
| Stage 8: Ridge | Test a global regularized linear model with train-only preprocessing. | Ridge validation report and frozen config. | Short-horizon gains did not persist over the full recursive horizon. |
| Stage 9: final model | Compare HistGradientBoosting and XGBoost with Ridge and the best baseline. | Frozen `mean_28` model, one-time locked-test receipt, production forecast API. | Balanced validation selected the simpler 28-day mean before the test was opened. |
| Stage 10: inventory | Combine the forecast with validation-only uncertainty and synthetic operating assumptions. | 300 service/cost recommendations and policy manifest. | Inventory and cost fields are labeled `synthetic_demo`, never Walmart truth. |
| Stage 11: application | Load validated artifacts into PostgreSQL and expose a focused Streamlit decision workflow. | SQL schema, idempotent loader, queries, CSV fallback, four-section dashboard. | Failed PostgreSQL connections are visible; fallback is never hidden. |
| Stage 12: release | Audit integrity, packaging, documentation, secrets, large files, and deployment configuration. | V1.0 metadata, health tools, Docker files, portfolio documentation, final QA report. | “Deployment-ready” does not mean a live cloud deployment occurred. |

## Reproducing the generated layers

Run stage commands only when rebuilding is intentional. Frozen Stage 9 test evaluation has a one-time receipt and should not be rerun for model selection. The normal reviewer workflow uses the existing generated artifacts and starts the CSV demo or loads those artifacts into PostgreSQL.
