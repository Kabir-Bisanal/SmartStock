# Streamlit application guide

SmartStock's dashboard presents one consistent V1 decision workflow through seven sections:

1. **Overview** — recent demand, forecast totals, inventory exposure, reorder count, and demo cost context.
2. **Sales Analytics** — historical demand trends and product/store comparisons.
3. **Demand Forecasting** — historical/forecast boundary, daily predictions, and 1-, 7-, and 30-day totals.
4. **Inventory Health** — inventory position, days of supply, status, risk, and synthetic-data labels.
5. **Reorder Recommendations** — filterable priorities and downloadable decisions.
6. **Scenario Planner** — changes selected synthetic inputs in memory and calls the Stage 10 engine; it never overwrites saved outputs.
7. **Methodology / About** — model, formulas, caveats, and data-source explanation.

Filters cascade by store, department, and product. Empty selections and missing artifacts produce actionable messages instead of fabricated results.

## Data-source modes

- `SMARTSTOCK_DATA_MODE=csv` uses the generated Stage 10 files in read-only demo mode.
- `SMARTSTOCK_DATA_MODE=postgres` requires a valid initialized PostgreSQL database.
- `SMARTSTOCK_DATA_MODE=auto` prefers PostgreSQL and visibly falls back to CSV when unavailable.

The sidebar always states the active source. PostgreSQL access uses SQLAlchemy Core queries with bound parameters. CSV mode reads the same frozen application contract and is intended for local reviews when a database is not practical.

Inventory balances, lead times, service levels, costs, and resulting savings are visibly described as synthetic/demo. There is no authentication, write-back workflow, or production monitoring in V1.
