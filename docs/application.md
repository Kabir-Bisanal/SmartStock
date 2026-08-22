# Streamlit application guide

SmartStock presents one concise decision workflow through four sections:

1. **Overview** — states the business question, shows the project scale and decision snapshot, and keeps methodology in a secondary expander.
2. **Demand forecast** — lets a reviewer select an anonymized SKU and store, compare recent demand with a 1-, 7-, or 30-day forecast, and download the selected forecast.
3. **Inventory recommendations** — the centerpiece. It defaults to products that need an order, explains the decision fields, prioritizes the largest needs, and provides a clean downloadable table.
4. **Scenario planner** — changes selected synthetic assumptions in a form, compares the result with the saved recommendation, and explains why the decision changed. It never overwrites saved outputs.

The interface deliberately groups sales context, inventory health, reorder priorities, and methodology around the decisions they support. This keeps the project easy to demonstrate without removing the underlying analytics.

## Data meaning

- Historical sales, calendar context, and anonymized item/store IDs come from the Walmart M5 dataset.
- Production forecasts come from the frozen 28-day historical-mean model selected through chronological validation.
- Inventory balances, lead times, service levels, costs, shortage reduction, and savings are synthetic demonstration values because M5 does not provide those operational fields.
- `REORDER` is a presentation label derived only when the saved recommended quantity is greater than zero; the engine's original stock status and priority labels remain unchanged.

## Data-source modes

- `SMARTSTOCK_DATA_MODE=csv` uses complete local Stage 10 artifacts when available. In a clean GitHub/cloud checkout it automatically uses the committed `data/public_demo/` bundle.
- `SMARTSTOCK_DATA_MODE=postgres` requires a valid initialized PostgreSQL database.
- `SMARTSTOCK_DATA_MODE=auto` prefers PostgreSQL and visibly falls back to CSV when unavailable.

The sidebar always states the active source. PostgreSQL access uses SQLAlchemy Core queries with bound parameters. The compact CSV bundle contains the latest 120 days (36,000 rows) for charts, all 9,000 saved production forecast rows, 300 synthetic inventory snapshots, 300 authoritative recommendations, and metadata preserving the full 582,300-observation project scope. It contains no model because the public app performs no forecasting inference at runtime; scenario calculations continue to use the authoritative inventory engine in memory.

## Portfolio demonstration path

For a short walkthrough:

1. Open **Overview** and state the question: which products should be reordered, how much, and why?
2. Open **Demand forecast** and show how recent demand becomes a future 30-day demand estimate.
3. Open **Inventory recommendations**, keep the default **Needs reorder** filter, and explain safety stock, reorder point, recommended units, risk, and priority.
4. Open **Scenario planner**, change on-hand stock or lead time, recalculate, and compare the saved and revised decisions.

There is no authentication, transaction workflow, write-back, or production monitoring in Version 1.
