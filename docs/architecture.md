# SmartStock V1 architecture

SmartStock separates the historical M5 evidence from the simulated operational inputs that M5 does not provide. The two streams meet only inside the inventory decision layer.

```mermaid
flowchart TD
    subgraph real["Observed and derived from Walmart M5"]
        A["M5 sales, calendar, and price CSVs"] --> B["Validation and profiling"]
        B --> C["Controlled V1 subset<br/>100 FOODS items × 3 stores"]
        C --> D["Long daily history<br/>582,300 rows"]
        D --> E["EDA and leakage-safe features"]
        E --> F["Chronological model comparison"]
        F --> G["Production 28-day-mean forecast<br/>300 series × 30 days"]
    end

    subgraph simulated["Synthetic demo inputs — not Walmart data"]
        H["On-hand, on-order, backorders"]
        I["Lead times, service levels, costs"]
    end

    G --> J["Inventory optimization engine"]
    H --> J
    I --> J
    J --> K["Reorder recommendations"]
    D --> L["PostgreSQL application tables"]
    G --> L
    K --> L
    L --> M["Streamlit dashboard"]
    D -. "read-only fallback" .-> N["CSV data source"]
    G -.-> N
    K -.-> N
    N --> M
```

## Responsibility boundaries

- `src/smartstock/data`, `analysis`, and `features` turn immutable raw inputs into documented analytical artifacts.
- `src/smartstock/models` evaluates forecasts without allowing validation-period targets into a 30-day forecast path.
- `src/smartstock/inventory` consumes forecasts plus explicitly synthetic business state; it has no UI or database dependency.
- `src/smartstock/database` owns the schema, validated loading, parameterized queries, and source selection.
- `app/` renders a four-section decision experience and delegates all forecasting and inventory calculations to existing project logic.
- `config/` freezes decisions; `reports/` preserves evidence; large generated datasets and models stay outside Git.

PostgreSQL is the intended application store. The CSV adapter is a visible, read-only portfolio fallback that uses the same Stage 10 artifacts.
