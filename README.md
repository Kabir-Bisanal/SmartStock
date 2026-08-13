# SmartStock — Demand Forecasting & Inventory Optimization System

SmartStock is a student portfolio project about forecasting retail demand and, in later stages, turning forecasts into inventory recommendations. It addresses the cost of both stockouts and excess inventory.

> **Current status:** Stage 6 (leakage-safe feature engineering and chronological validation design) is complete. The project is under development; no forecasting model, model evaluation, optimization, database, or dashboard has been implemented.

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

## Run tests

```powershell
python -m unittest discover -s tests -v
```

The tests use small synthetic fixtures and do not load the full M5 dataset.
