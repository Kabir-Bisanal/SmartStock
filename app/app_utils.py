"""Pure helpers used by the Streamlit presentation layer and its tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from smartstock.inventory.engine import recommend_inventory_for_series


INVENTORY_DISCLAIMER = (
    "Demo Inventory Inputs — M5 does not provide stock balances, supplier lead times, "
    "service targets, or operating costs. These inventory and cost values are synthetic."
)


def apply_inventory_scenario(
    daily_forecast: pd.Series | list[float],
    snapshot: dict[str, Any],
    calibration: dict[str, Any],
    base_policy: dict[str, Any],
    overrides: dict[str, Any],
    *,
    scenario_seed: int = 42,
) -> dict[str, Any]:
    """Run an in-memory what-if scenario through the authoritative Stage 10 engine."""

    inventory_state = deepcopy(snapshot)
    inventory_state.update({key: value for key, value in overrides.items() if key in inventory_state})
    policy = deepcopy(base_policy)
    for key in (
        "review_period_days", "holding_cost_per_unit_per_day", "stockout_cost_per_unit",
        "fixed_order_cost",
    ):
        if key in overrides:
            policy[key] = overrides[key]
    for key in ("holding_cost_per_unit_per_day", "stockout_cost_per_unit", "fixed_order_cost"):
        if key in overrides:
            inventory_state[key] = overrides[key]
    return recommend_inventory_for_series(
        list(daily_forecast), inventory_state, calibration, policy, scenario_seed=scenario_seed
    )


def forecast_chart_data(history: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy chart frame that never invents post-history actual sales."""

    actual = history[["date", "sales"]].rename(columns={"sales": "Actual"}).copy()
    actual["date"] = pd.to_datetime(actual["date"])
    actual["Forecast"] = float("nan")
    future = forecast[["target_date", "forecast"]].rename(
        columns={"target_date": "date", "forecast": "Forecast"}
    ).copy()
    future["date"] = pd.to_datetime(future["date"])
    future["Actual"] = float("nan")
    return pd.concat([actual, future], ignore_index=True).sort_values("date")


def recommendation_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Select and order business-facing recommendation fields."""

    columns = [
        "item_id", "store_id", "dept_id", "stock_status", "priority_label", "priority_score",
        "forecast_7d", "forecast_30d", "inventory_position", "days_of_supply",
        "stockout_risk_pct", "reorder_point", "safety_stock", "recommended_order_qty",
        "cost_optimized_order_qty", "estimated_cost_savings",
        "expected_shortage_without_order", "expected_shortage_with_recommendation",
    ]
    return frame[[column for column in columns if column in frame.columns]].copy()
