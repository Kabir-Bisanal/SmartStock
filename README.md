# SmartStock — Demand Forecasting & Inventory Optimization System

SmartStock is a student portfolio project about forecasting retail demand and, in later stages, turning forecasts into inventory recommendations. It addresses the cost of both stockouts and excess inventory.

> **Current status:** Stage 10 (inventory optimization and reorder decisions) is complete. The project is under development; database integration, dashboard development, and deployment have not been implemented.

## Dataset

The project uses the official [Walmart M5 Forecasting — Accuracy competition data](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data). The project expects these files:

```text
data/raw/
├── calendar.csv
├── sales_train_evaluation.csv
└── sell_prices.csv
```

Downloaded and generated datasets are ignored by Git because the M5 files are large. Only the empty directory placeholders are versioned.

### Obtain the data

1. Sign in to Kaggle and open the official competition data page linked above.
2. Accept any competition rules or terms Kaggle presents.
3. Choose **Download All** and extract the downloaded archive.
4. Copy `calendar.csv`, `sales_train_evaluation.csv`, and `sell_prices.csv` into `data/raw/` using those exact filenames.
5. Run the validation command below.

Kaggle credentials and the Kaggle command-line client are not required by this repository. If you already maintain an authenticated Kaggle CLI separately, the official competition identifier is `m5-forecasting-accuracy`; do not commit its credentials or downloaded archive.

## Planned architecture

```text
M5 data -> Python ETL -> PostgreSQL -> Feature engineering
        -> Demand forecasting -> Inventory optimization -> Streamlit dashboard
```

The acquisition, profiling, controlled Version 1 transformation, descriptive EDA,
and leakage-safe modeling foundation are complete. Later architecture stages remain unbuilt.

## Project structure

```text
smartstock/
├── app/                         # Future application code
├── data/
│   ├── raw/                     # Original M5 files (not committed)
│   ├── interim/                 # Future intermediate data
│   ├── processed/               # Future model-ready data
│   └── simulated/               # Future, clearly labeled simulated business data
├── notebooks/                   # Future analysis notebooks
├── reports/                     # Generated profiling findings
├── sql/                         # Future SQL files
├── src/smartstock/
│   ├── analysis/                # Reproducible exploratory analysis code
│   ├── data/                    # Dataset acquisition and validation code
│   ├── database/                # Future database code
│   ├── features/                # Future feature engineering code
│   ├── inventory/               # Future inventory logic
│   ├── models/                  # Future forecasting code
│   └── utils/                   # Shared future utilities
└── tests/                       # Automated tests
```

## Local setup

Python 3.11 or 3.12 is recommended. From the repository root on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell reports that `python` is not recognized, install Python from
[python.org](https://www.python.org/downloads/) with the PATH option enabled,
open a new terminal, and confirm the installation with `python --version`.

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

## Validate the raw files

The validator uses only Python's standard library, so it can run before the data-science packages are imported:

```powershell
python src/smartstock/data/validate_raw_data.py
```

By default it checks file presence, size, CSV headers, column counts, and required key columns without scanning every row. Use the optional full row count only when needed because the sales file is large:

```powershell
python src/smartstock/data/validate_raw_data.py --count-rows
```

The command exits with status `0` when all expected files pass validation and `1` when a file is missing or invalid.

## Generate the Stage 3 profile

The Stage 3 profiler reads the real M5 files without modifying them. It scans the
wide sales values in chunks and does not permanently convert them to long format:

```powershell
python src/smartstock/data/profile_m5_data.py
```

It produces:

- `reports/stage3_data_profile.md` — detailed human-readable findings.
- `reports/stage3_profile_summary.json` — the same core results in a reproducible machine-readable form.

The report covers dataset structure, retail hierarchy, join-key verification,
data-quality observations, and three evidence-backed Version 1 subset strategies.
It does not select or extract the final subset.

## Build the Version 1 working dataset

Stage 4 uses the frozen stores `CA_1`, `TX_2`, and `WI_3`, applies documented
eligibility rules, selects 100 FOODS items with fixed department and demand-band
quotas, and melts only the resulting 300 item-store rows:

```powershell
python src/smartstock/data/build_v1_dataset.py
```

Generated and tracked artifacts:

- `config/v1_subset.json` — reproducible stores, item IDs, seed, thresholds, and eligibility policy.
- `reports/stage4_transformation_report.md` — human-readable transformation and validation results.
- `reports/stage4_summary.json` — machine-readable execution statistics.

Generated but Git-ignored data:

- `data/interim/smartstock_v1_long.csv` — the 582,300-row Version 1 daily working dataset.

Missing selling prices remain missing in Stage 4; no imputation or forecasting
feature engineering is performed.

## Run the Stage 5 exploratory analysis

Stage 5 reads the frozen Version 1 interim CSV without modifying it. It separates
pre-launch rows from active-period demand, profiles demand and price behavior, and
assesses modeling readiness without creating feature tables or training models:

```powershell
python src/smartstock/analysis/stage5_eda.py
```

It produces:

- `reports/stage5_eda_report.md` — detailed, beginner-readable findings and decisions.
- `reports/stage5_summary.json` — machine-readable statistics and integrity results.
- `reports/figures/stage5/` — 12 analysis figures used by the report.

The analysis is descriptive and observational. Event, SNAP, and price comparisons
must not be interpreted as causal effects.

## Build the Stage 6 feature dataset

Stage 6 excludes unavailable pre-launch targets, retains genuine active zero demand,
and creates calendar, product-age, past-demand, intermittency, and completed-prior-week
price features. Historical demand windows are shifted so the target day cannot enter
its own predictors:

```powershell
python src/smartstock/features/build_features.py
```

Tracked outputs:

- `config/v1_features.json` — feature timing, category, dtype, and leakage metadata.
- `config/v1_validation.json` — three expanding validation folds and the locked final test.
- `reports/stage6_feature_engineering_report.md` — calculated policies, results, and risks.
- `reports/stage6_summary.json` — machine-readable Stage 6 statistics and checks.

Generated but Git-ignored output:

- `data/processed/smartstock_v1_features.csv` — active rows, warm-up flags, features, and target.

No model is trained by this command. The actual target-week price is scenario-only;
it is not part of the default forecast-safe feature set.

## Run the Stage 7 baseline evaluation

Stage 7 evaluates zero, persistence, 7-day seasonal naive, 28-day mean, and
Croston-SBA rules across the three frozen validation folds:

```powershell
python src/smartstock/models/baseline_evaluation.py
```

Every 30-day forecast is generated once from its fold origin; no validation-period
actual sales update later horizon days. Outputs include:

- `reports/stage7_baseline_metrics.csv` — fold, horizon, aggregate, and segment metrics.
- `reports/stage7_summary.json` — machine-readable conclusions and leakage audits.
- `reports/stage7_baseline_evaluation_report.md` — detailed findings and trade-offs.
- `reports/figures/stage7/` — eight validation-only comparison figures.
- `data/processed/baselines/baseline_validation_predictions.csv` — Git-ignored predictions.

The locked final test (`2016-04-23` through `2016-05-22`) is explicitly blocked
and is not forecast or scored by Stage 7.

## Run the Stage 8 global Ridge evaluation

Stage 8 fits one global regularized linear model per validation fold. Categorical
features are one-hot encoded, numeric features are standardized, and each 30-day
forecast is generated recursively so later lags use earlier predictions rather
than validation-period actual demand:

```powershell
python src/smartstock/models/stage8_evaluation.py
```

Tracked outputs:

- `config/v1_ridge.json` — frozen model, feature, preprocessing, and recursion policy.
- `reports/stage8_ridge_metrics.csv` — fold, horizon, aggregate, segment, and Stage 7 comparison metrics.
- `reports/stage8_summary.json` — machine-readable model results and leakage audits.
- `reports/stage8_ridge_evaluation_report.md` — detailed interpretation and limitations.
- `reports/figures/stage8/` — eight validation-only comparison figures.

Generated but Git-ignored output:

- `data/processed/models/stage8/ridge_validation_predictions.csv` — raw and clipped recursive validation predictions.

Ridge v1 is intentionally price-agnostic. Its fixed `alpha=1.0` is not tuned, and
the locked final test remains untouched.

## Run the Stage 9 final forecasting workflow

Stage 9 compares HistGradientBoosting and XGBoost against the frozen Global Ridge
and 28-Day Historical Mean under the same recursive validation rules. Validation
selected the 28-day mean as the Version 1 production forecaster: boosting improved
some short-horizon and spike-sensitive metrics but did not provide a balanced,
stable improvement across the full inventory horizon.

The controlled workflow has two deliberately separate commands:

```powershell
python -m smartstock.models.stage9_evaluation --phase validation
python -m smartstock.models.stage9_evaluation --phase final-test
```

The validation command freezes `config/v1_final_model.json` before the second
command can access the locked test. The final-test command is guarded by a receipt
and refuses a second evaluation. On a completed checkout, do not rerun selection or
test evaluation; the freeze and receipt are the audit record.

Tracked outputs include:

- `config/v1_stage9_candidates.json` — controlled candidates, features, tuning grids, and policies.
- `config/v1_final_model.json` — final family, parameters, recursive policy, and pre-test freeze evidence.
- `reports/stage9_model_metrics.csv` — validation and one-time locked-test metrics.
- `reports/stage9_summary.json` — machine-readable selection, test, integrity, and production metadata.
- `reports/stage9_final_model_report.md` — full interpretation and limitations.
- `reports/figures/stage9/` — ten model-selection and final-test figures.

Generated, Git-ignored outputs include the validation/test prediction CSVs under
`data/processed/models/stage9/` and the serialized production forecaster at
`models/smartstock_v1_forecaster.joblib`.

Downstream code can load the artifact and call
`smartstock.models.production_forecaster.forecast_demand` for 1-, 7-, or 30-day
daily forecasts. The same model can be trained reproducibly from the frozen config:

```powershell
python -m smartstock.models.train_final_model
```

The production forecaster remains price-agnostic and does not use demand band as a
predictor. The locked test was evaluated exactly once after model selection froze.

## Run the Stage 10 inventory optimization engine

Stage 10 creates a separate deployment refresh of the frozen 28-day mean through
`2016-05-22`, generates a 30-day application forecast, calibrates forecast error
from validation-only residuals, and converts the forecast into inventory decisions:

```powershell
python -m smartstock.inventory.build_recommendations
```

The command produces safety stock, reorder points, target stock levels, service-level
order quantities, stockout-risk estimates, stock statuses, priority scores, and a
bounded scenario-based cost recommendation for all 300 item-store series.

Important limitation: M5 does **not** include real on-hand inventory, purchase orders,
backorders, supplier lead times, service targets, or inventory costs. Stage 10 creates
deterministic synthetic/demo values for those inputs. They are explicitly labeled
`synthetic_demo` and must never be presented as Walmart operational data.

Tracked outputs:

- `config/v1_inventory_policy.json` — demo assumptions, formulas, thresholds, and scenario policy.
- `reports/stage10_inventory_optimization_report.md` — calculated business results and limitations.
- `reports/stage10_summary.json` — machine-readable recommendations and integrity checks.
- `reports/figures/stage10/` — eight inventory-decision figures.

Generated and Git-ignored outputs:

- `models/smartstock_v1_deployment_forecaster.joblib` — deployment refresh trained through `2016-05-22`.
- `data/processed/production/smartstock_v1_30day_forecast.csv` — 9,000 daily forecasts.
- `data/processed/inventory/smartstock_v1_error_calibration.csv` — validation-only uncertainty calibration.
- `data/simulated/smartstock_v1_inventory_snapshot.csv` — synthetic inventory balances and assumptions.
- `data/processed/inventory/smartstock_v1_inventory_recommendations.csv` — 300 application-ready decisions.

The Stage 9 evaluation artifact remains separate and unchanged. The reusable inventory
API is `smartstock.inventory.engine.recommend_inventory`; it has no database or
Streamlit dependency.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

The tests use small synthetic fixtures and do not load the full M5 dataset.
