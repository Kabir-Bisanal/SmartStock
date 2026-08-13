# SmartStock Stage 11 Application Report

## Executive Summary

Stage 11 turns the completed SmartStock forecasting and inventory layers into a locally runnable application. PostgreSQL is the intended data store, SQLAlchemy Core provides bound-parameter access, and Streamlit provides seven business-facing sections. When PostgreSQL is not configured or is unavailable in `auto` mode, the app visibly switches to the same generated CSV artifacts in read-only demo mode.

All real M5/data-derived values and synthetic/demo inventory assumptions are distinguished in the interface. Historical sales, dates, store/product context, prices, and demand forecasts are data-derived. Inventory balances, lead times, service targets, costs, and decisions that depend on them are labeled synthetic because M5 does not contain Walmart stock balances.

## Architecture

```text
Stage 10 generated artifacts
        |
        +--> validated chunked loader --> PostgreSQL
        |                                |
        +--> read-only CSV fallback -----+--> common data-source interface
                                                 |
                                         Streamlit sections
                                                 |
                                  existing Stage 10 scenario engine
```

`DATABASE_URL` is the only database credential setting. `SMARTSTOCK_DATA_MODE` supports `auto`, `postgres`, and `csv`. No credential is stored in source control, and UI values are never converted into raw SQL strings.

## Database Schema

The versioned PostgreSQL schema is `sql/001_create_schema.sql`. Matching SQLAlchemy metadata is in `smartstock.database.schema` so application code and temporary SQLite tests share one structural contract.

Five tables are defined:

| Table | Role | Expected rows | Primary key |
|---|---|---:|---|
| `demand_history` | Real M5/data-derived long sales history | 582,300 | `(date, store_id, item_id)` |
| `forecasts` | Stage 10 production daily forecasts | 9,000 | `(target_date, store_id, item_id)` |
| `inventory_snapshot` | Synthetic demo inventory inputs | 300 | `(store_id, item_id)` |
| `inventory_recommendations` | Persisted Stage 10 decisions | 300 | `(store_id, item_id)` |
| `model_metadata` | Model/policy identity and hashes | 1 | `metadata_id` |

Indexes support date filtering, item/store time-series retrieval, store and department trend queries, forecast retrieval, and status/priority filtering. Indexes are intentionally limited to the dashboard's frequent access paths.

## Database Loading

`python -m smartstock.database.initialize_database`:

1. validates `DATABASE_URL`;
2. validates every Stage 10 artifact and its required columns;
3. creates missing tables/indexes;
4. derives the store-relevant SNAP flag;
5. bulk-loads the 582,300 history rows in bounded chunks rather than row-by-row;
6. loads forecasts, snapshot, recommendations, and metadata;
7. checks table, item, store, and series cardinalities.

An identical complete database is skipped safely on rerun. A partial/outdated database produces an actionable message. `--refresh` removes and reloads rows only from the five application tables.

## Data Source Fallback

In `auto` mode, the resolver uses PostgreSQL only when the connection succeeds and all expected SmartStock table counts validate. Otherwise it reports the reason class and shows `Data source: Local demo files`. `postgres` mode fails closed, while `csv` explicitly selects the fallback.

The CSV adapter exposes the same query methods as PostgreSQL. It caches loaded frames for the Streamlit resource lifetime and remains read-only.

## Streamlit Structure

`app/streamlit_app.py` contains presentation and navigation. `app/app_utils.py` holds pure chart/table/scenario helpers. Database queries and inventory formulas remain outside the UI.

`st.cache_resource` retains the resolved data source and policy. `st.cache_data` caches repeated filter-option and historical aggregation queries. The mutable what-if scenario itself is deliberately not cached.

## Overview Page

The Overview provides product/store/series counts, recent sales, 7- and 30-day forecasts, reorder and critical counts, recommended units, shortage before/after ordering, and demo cost savings. It also shows status, priority, forecast-by-store, and recommended-units-by-store charts, followed by an explicit synthetic inventory disclaimer.

## Sales Analytics

Users can filter store, department, product, and date range; choose daily, weekly, or monthly aggregation; and inspect total demand, store/department contribution, weekday behavior, zero-demand prevalence, mean units, and maximum units. PostgreSQL performs aggregation/filtering rather than returning all history for each interaction.

## Demand Forecasting

The forecasting page combines the last 90 actual observations with 1-, 7-, or 30-day Stage 10 production forecasts. Actual history ends on `2016-05-22`; the future part contains forecasts only. The page names `mean_28` as the selected method and explains that it won the balanced validation comparison over Ridge and boosting candidates.

## Inventory Health

This page provides status, priority, days-of-supply, stockout-risk, inventory-position, reorder-point, safety-stock, and order-quantity views. Tables are sortable and filterable by store, department, status, and priority.

## Reorder Recommendations

Recommendations default to descending priority score and expose service-level and cost-optimized quantities, forecast horizons, inventory position, shortage reduction, and demo savings. The output is explicitly dependent on synthetic inputs.

## Scenario Planner

The in-memory What-if Scenario accepts on-hand, on-order, backorders, lead time, review period, service level, holding cost, stockout cost, and fixed order cost. It calls `smartstock.inventory.engine.recommend_inventory_for_series`; no formula is duplicated in Streamlit. It recalculates inventory position, demand exposure, safety stock, reorder point, target stock, days of supply, risk, status, priority, two order recommendations, and scenario costs without updating the database or CSV.

## Methodology Page

The recruiter-facing page summarizes Walmart M5, the frozen 100-item/3-store scope, wide-to-long engineering, all evaluated forecast families, chronological/leakage-safe evaluation, selected `mean_28` model, synthetic inventory policy, limitations, and reproducibility metadata.

## Reused Forecasting / Inventory Logic

The production forecasts are loaded from the real Stage 10 artifact. Scenario calculations call the authoritative Stage 10 engine. Forecast formulas, safety stock, inventory position, reorder point, priority, and cost optimization were not reimplemented.

## Testing

- 79 project unit tests passed, including 8 focused Stage 11 tests.
- A synthetic temporary SQLite test covers creation, loading, idempotency, filters, duplicate rejection, and SQL-injection-resistant parameter binding.
- A full temporary SQLite integration loaded all real generated artifacts in 24.606 seconds: 582,300 / 9,000 / 300 / 300 / 1 rows; 100 items; 3 stores; 300 series. The database was 150,863,872 bytes and was removed after the test.
- Automated Streamlit testing rendered all seven sections with zero exceptions.
- Live browser QA verified the Overview, Demand Forecasting, and Scenario Planner with zero console errors. Changing on-hand stock from 45 to 0 changed one scenario from `OVERSTOCK / NONE` to `REORDER NOW / HIGH` with a 17-unit recommendation.
- A live PostgreSQL instance was not available: no `DATABASE_URL`, PostgreSQL service, `psql`, or `pg_isready` was present. No PostgreSQL success is claimed.

## Local Run Instructions

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Set DATABASE_URL in .env
python -m smartstock.database.initialize_database
streamlit run app/streamlit_app.py
```

For demo mode without PostgreSQL:

```powershell
$env:SMARTSTOCK_DATA_MODE = "csv"
streamlit run app/streamlit_app.py
```

See `docs/database_setup.md` for database creation, verification, refresh, and troubleshooting.

## Known Limitations

- Inventory and cost values are synthetic educational assumptions, not Walmart operational data.
- Sales may be censored by historical stockouts, and future price schedules are not modeled.
- CSV mode caches the complete 582,300-row history locally; PostgreSQL is the preferred repeated-query mode.
- Codex could not verify a real PostgreSQL connection because no server/configuration was available.
- Authentication, Docker, cloud deployment, CI/CD, scheduled jobs, production monitoring, and final portfolio polish remain out of scope.

## Stage 12 Readiness

The local data/application contract is complete and tested. Stage 12 can concentrate on deployment choice, environment-specific PostgreSQL provisioning, final QA, documentation polish, and portfolio presentation without adding another forecasting or inventory stage.
