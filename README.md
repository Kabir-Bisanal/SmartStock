# SmartStock — Demand Forecasting & Inventory Optimization System

SmartStock is a student portfolio project about forecasting retail demand and, in later stages, turning forecasts into inventory recommendations. It addresses the cost of both stockouts and excess inventory.

> **Current status:** Stage 4 (Version 1 subset creation and controlled data transformation) is complete. The project is under development; no feature engineering, forecasting, optimization, database, or dashboard has been implemented.

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

Only the acquisition and validation foundation exists in the current stage.

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

## Run tests

```powershell
python -m unittest discover -s tests -v
```

The tests use only the standard library and generate tiny temporary CSV fixtures; they do not require the real M5 dataset.
