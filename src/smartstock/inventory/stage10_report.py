"""Render the calculated SmartStock Stage 10 inventory report."""

from __future__ import annotations

from typing import Any


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def render_stage10_report(summary: dict[str, Any]) -> str:
    """Return a beginner-readable report populated only from generated results."""

    totals = summary["recommendation_results"]
    store_rows = [
        [
            row["store_id"],
            str(row["series"]),
            str(row["requiring_reorder"]),
            f"{row['recommended_units']:,.0f}",
            f"{row['expected_shortage_without_order']:,.1f}",
            f"{row['estimated_cost_savings']:,.1f}",
        ]
        for row in summary["store_results"]
    ]
    department_rows = [
        [
            row["dept_id"],
            str(row["series"]),
            str(row["requiring_reorder"]),
            f"{row['recommended_units']:,.0f}",
            f"{row['expected_shortage_without_order']:,.1f}",
            f"{row['estimated_cost_savings']:,.1f}",
        ]
        for row in summary["department_results"]
    ]
    status_rows = [[name, str(count)] for name, count in totals["stock_status_counts"].items()]
    priority_rows = [[name, str(count)] for name, count in totals["priority_counts"].items()]
    return f"""# SmartStock Stage 10 — Inventory Optimization & Reorder Decision Engine

## Executive Summary

Stage 10 converted the frozen Version 1 demand forecasts into inventory decisions for all **{totals['series']:,} item-store series**. The engine produced lead-time demand, safety stock, reorder points, target stock levels, stockout-risk estimates, service-level reorder quantities, cost-aware order quantities, stock statuses, and transparent priority scores.

The underlying M5 demand is real public competition data. The inventory balances, lead times, service levels, and costs are **synthetic demo assumptions**, not Walmart operational data. Under this demo snapshot, **{totals['requiring_reorder']:,} series require a service-level order**, totaling **{totals['total_recommended_units']:,.0f} units**. The scenario-based cost optimizer reduced estimated planning cost from **{totals['estimated_cost_without_order']:,.1f}** to **{totals['estimated_cost_with_recommendation']:,.1f} demo currency units**, an estimated saving of **{totals['estimated_cost_savings']:,.1f}**.

## Why Inventory Inputs Are Simulated

M5 contains sales, calendar, and selling-price information, but it does not contain actual on-hand inventory, open purchase orders, backorders, supplier lead times, service targets, holding costs, shortage costs, or fixed ordering costs. Creating values for those fields is necessary to demonstrate inventory decisions, but those values must not be represented as Walmart facts.

Every generated balance is marked `inventory_source = synthetic_demo`. The fixed seed and tracked generation rules make the demo reproducible. The predefined coverage profiles were assigned before recommendation statuses were calculated; statuses were not manually adjusted afterward.

## Forecast Inputs

The Stage 9 evaluation artifact remains frozen through **2016-04-22**. A separate deployment artifact was refreshed through **{summary['deployment_forecast']['trained_through']}** using the exact frozen `mean_28` method. It produced **{summary['deployment_forecast']['rows']:,} daily forecasts from {summary['deployment_forecast']['date_min']} through {summary['deployment_forecast']['date_max']}**.

The deployment method is price- and calendar-feature-agnostic. It calculates future target dates algorithmically, predicts the latest 28-value mean, appends that prediction to synthetic history, and repeats. No event, SNAP, price, or invented future calendar data was used.

## Forecast Uncertainty

Residual is defined consistently as:

```text
residual = actual - forecast
```

Calibration used **{summary['uncertainty_calibration']['validation_rows']:,} mean-28 predictions** covering {summary['uncertainty_calibration']['date_min']} through {summary['uncertainty_calibration']['date_max']}. The locked final-test period contributed **zero** calibration rows. Each series had sufficient validation history for item-store calibration, so all {summary['uncertainty_calibration']['series']} series used their own residual mean and daily residual standard deviation.

## Inventory Position

```text
inventory_position = on_hand + on_order - backorders
```

On-order stock matters because it is already expected to arrive. Backorders reduce usable inventory because those units are already owed to unmet demand.

## Safety Stock

```text
safety_stock = z(service_level) × sigma_daily × sqrt(lead_time_days)
```

Safety stock provides extra protection against forecast error. A higher service level produces a larger normal quantile and therefore more inventory protection. Longer lead time also increases safety stock because uncertainty accumulates while replenishment is unavailable. This normal, independent-error approximation is practical for Version 1 but is not a guarantee.

## Reorder Point

```text
reorder_point = forecast lead-time demand + safety stock
```

Replenishment is triggered when inventory position is at or below the units expected to be consumed before the next order arrives, plus uncertainty protection.

## Target Stock Level

```text
target_stock_level = forecast(lead time + review period) + safety stock
```

The default review period is {summary['policy']['review_period_days']} days. The target covers demand until the next review opportunity as well as supplier lead time.

## Reorder Quantity

```text
raw quantity = max(0, target stock level - inventory position)
physical quantity = ceil(raw quantity)
```

Continuous values are preserved for auditability, while the final service-level recommendation is rounded upward because retail units generally cannot be ordered fractionally.

## Stockout Risk

Lead-time demand is approximated by a normal distribution centered on the forecast, with standard deviation `sigma_daily × sqrt(lead_time)`. Risk is the estimated probability that demand exceeds inventory position. It is an estimate under the Version 1 assumptions, not a certainty or causal probability.

## Stock Status

{_table(['Status', 'Series'], status_rows)}

Status precedence is deterministic: `STOCKOUT`, `CRITICAL`, `REORDER NOW`, `LOW`, `OVERSTOCK`, then `HEALTHY`. `CRITICAL` uses an estimated stockout-risk threshold of 75%.

## Priority Score

The 0–100 priority formula is:

```text
100 × (
    0.50 × stockout risk
  + 0.30 × normalized reorder-point gap
  + 0.20 × normalized target-stock gap
)
```

{_table(['Priority', 'Series'], priority_rows)}

This score is an interpretable business rule, not a machine-learning model.

## Cost-Aware Optimization

For each series, {summary['policy']['scenario_count']} deterministic demand scenarios were generated with seed 42 and validation-calibrated residual uncertainty. Daily scenario demand was clipped at zero. A bounded grid included zero, the service-level recommendation, and quantities covering the upper scenario-demand range.

For each quantity:

```text
available inventory = inventory position + order quantity
surplus = max(available inventory - scenario demand, 0)
shortage = max(scenario demand - available inventory, 0)
cost = holding approximation + shortage cost + fixed cost when ordering
```

The optimizer always included “no order,” so its estimated cost cannot be worse than the no-order scenario within the same simulation. These are demo currency assumptions, not Walmart economics.

## Synthetic Inventory Snapshot

The snapshot contains exactly {summary['synthetic_inventory']['rows']} unique item-store series. Planned seeded profiles were:

{_table(['Predefined profile', 'Series'], [[name, str(value)] for name, value in summary['synthetic_inventory']['profile_counts'].items()])}

Lead times vary among 3, 5, 7, 10, and 14 days, with 7 days most common. Service levels vary among 90%, 95%, 97.5%, and 99%, with 95% most common.

## Recommendation Results

- Service-level reorders: **{totals['requiring_reorder']:,} series / {totals['total_recommended_units']:,.0f} units**
- Cost-optimized orders: **{totals['cost_optimized_reorders']:,} series / {totals['total_cost_optimized_units']:,.0f} units**
- Expected shortage without ordering: **{totals['expected_shortage_without_order']:,.1f} units**
- Expected shortage after cost-optimized orders: **{totals['expected_shortage_with_recommendation']:,.1f} units**
- Expected excess without ordering: **{totals['expected_excess_without_order']:,.1f} units**
- Estimated cost saving: **{totals['estimated_cost_savings']:,.1f} demo currency units**

## Store-Level Results

{_table(['Store', 'Series', 'Reorders', 'Service units', 'Shortage before', 'Cost savings'], store_rows)}

## Department-Level Results

{_table(['Department', 'Series', 'Reorders', 'Service units', 'Shortage before', 'Cost savings'], department_rows)}

## Cost / Shortage Results

The cost-aware optimizer reduced expected shortage by **{totals['expected_shortage_reduction']:,.1f} units** in the simulated planning horizons. The service-level and cost-optimal order quantities are deliberately both retained: the first prioritizes the configured service target, while the second explicitly trades shortage cost, holding cost, and fixed ordering cost.

## Limitations

- Inventory and cost values are synthetic educational inputs.
- The normal approximation may be imperfect for zero-heavy and intermittent demand.
- Daily residuals are treated as independent when scaling by the square root of lead time.
- Supplier capacity, minimum order quantities, case packs, shelf life, delivery schedules, and budget constraints are not modeled.
- Estimated costs use an average-horizon holding approximation rather than detailed daily stock flow.
- The optional historical inventory-policy backtest was deferred; the scenario engine and recommendation layer were prioritized for the accelerated roadmap.

## Stage 11 Integration Contract

Stage 11 can load:

- `data/processed/production/smartstock_v1_30day_forecast.csv`
- `data/simulated/smartstock_v1_inventory_snapshot.csv`
- `data/processed/inventory/smartstock_v1_inventory_recommendations.csv`
- `config/v1_inventory_policy.json`
- `models/smartstock_v1_deployment_forecaster.joblib`

The reusable API is `smartstock.inventory.engine.recommend_inventory`. It accepts tidy forecasts, one inventory-state row per series, validation-only uncertainty calibration, and the policy dictionary. It has no Streamlit or database dependency.
"""
