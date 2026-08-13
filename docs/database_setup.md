# SmartStock PostgreSQL setup

PostgreSQL is SmartStock's intended local/production data source. The Streamlit app also has a clearly labeled, read-only CSV demo mode for reviewers who do not have PostgreSQL.

## 1. Install and start PostgreSQL

Install a currently supported PostgreSQL release for Windows and ensure its service is running. SmartStock does not require PostgreSQL command-line tools at application runtime, but `psql` is useful for setup and verification.

## 2. Create an empty database

From PowerShell, using your own PostgreSQL administrator account:

```powershell
psql -U postgres -c "CREATE DATABASE smartstock;"
```

If the database already exists, do not recreate it.

## 3. Configure the connection

Copy the example file and edit only the untracked `.env` copy:

```powershell
Copy-Item .env.example .env
```

Set this format with your own local values:

```text
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/smartstock
SMARTSTOCK_DATA_MODE=auto
```

Never commit `.env`. If a username or password contains URL-special characters, URL-encode it. SmartStock accepts `postgresql://` and normalizes it to the Psycopg 3 driver, but the explicit `postgresql+psycopg://` form is clearer.

Data modes:

- `auto`: use an initialized PostgreSQL database when available; otherwise show the reason and use local CSV demo files.
- `postgres`: require PostgreSQL and fail with an actionable message if unavailable.
- `csv`: explicitly use local generated artifacts in read-only demo mode.

## 4. Initialize the application tables

Activate the project environment, ensure the Stage 10 generated artifacts exist, then run:

```powershell
python -m smartstock.database.initialize_database
```

The command validates schemas and expected cardinalities, creates missing tables and indexes, inserts history in chunks, and reports the real database counts. Re-running it skips a complete existing load. To deliberately replace table contents from the current generated files:

```powershell
python -m smartstock.database.initialize_database --refresh
```

`--refresh` deletes and reloads rows only in the five SmartStock application tables. It does not modify raw, interim, processed, simulated, config, or model files.

## 5. Verify the database

Expected Version 1 counts:

```text
demand_history               582300
forecasts                       9000
inventory_snapshot               300
inventory_recommendations        300
model_metadata                     1
```

An optional `psql` check:

```powershell
psql -U postgres -d smartstock -c "SELECT 'demand_history' AS table_name, COUNT(*) FROM demand_history UNION ALL SELECT 'forecasts', COUNT(*) FROM forecasts UNION ALL SELECT 'inventory_snapshot', COUNT(*) FROM inventory_snapshot UNION ALL SELECT 'inventory_recommendations', COUNT(*) FROM inventory_recommendations UNION ALL SELECT 'model_metadata', COUNT(*) FROM model_metadata;"
```

The initializer also verifies 100 items, 3 stores, and 300 item-store series.

## Common connection issues

- `DATABASE_URL is not configured`: create `.env` and set the URL, or run the app in `csv` mode.
- Connection refused: start the PostgreSQL Windows service and verify port `5432` (or use the configured port).
- Authentication failed: check the user/password and confirm the user can connect to `smartstock`.
- Database is partially loaded: run the initializer with `--refresh` after confirming the URL points to the intended SmartStock database.
- Driver import failure: reinstall project requirements; the application uses Psycopg 3 (`psycopg`), not `psycopg2`.

The application never accepts arbitrary SQL from UI inputs. Filters are converted into SQLAlchemy expressions with bound parameters.
