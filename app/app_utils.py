"""Pure helpers used by the Streamlit presentation layer and its tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import pandas as pd

from smartstock.inventory.engine import recommend_inventory_for_series


INVENTORY_DISCLAIMER = (
    "Demo inventory inputs — M5 does not provide stock balances, supplier lead times, "
    "service targets, or operating costs. These inventory and cost values are synthetic."
)

RECOMMENDATION_COLUMNS = {
    "order_action": "Action",
    "item_id": "SKU",
    "dept_id": "Department",
    "store_id": "Store",
    "on_hand": "On hand (demo)",
    "on_order": "On order (demo)",
    "inventory_position": "Inventory position (demo)",
    "forecast_30d": "30-day forecast",
    "safety_stock": "Safety stock",
    "reorder_point": "Reorder point",
    "recommended_order_qty": "Recommended units",
    "stock_status": "Stock status",
    "priority_label": "Priority",
    "stockout_risk_pct": "Stockout risk (%)",
}


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
        "review_period_days",
        "holding_cost_per_unit_per_day",
        "stockout_cost_per_unit",
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
    """Return a chart frame that never invents post-history actual sales."""

    actual = history[["date", "sales"]].rename(columns={"sales": "Actual demand"}).copy()
    actual["date"] = pd.to_datetime(actual["date"])
    actual["Forecast demand"] = float("nan")
    future = forecast[["target_date", "forecast"]].rename(
        columns={"target_date": "date", "forecast": "Forecast demand"}
    ).copy()
    future["date"] = pd.to_datetime(future["date"])
    future["Actual demand"] = float("nan")
    return pd.concat([actual, future], ignore_index=True).sort_values("date")


def add_order_action(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a display-only action derived from the saved recommendation quantity."""

    result = frame.copy()
    result["order_action"] = result["recommended_order_qty"].gt(0).map(
        {True: "REORDER", False: "NO ORDER"}
    )
    return result


def filter_recommendations(
    frame: pd.DataFrame,
    *,
    action: str = "Needs reorder",
    store_id: str = "All",
    dept_id: str = "All",
) -> pd.DataFrame:
    """Apply portfolio-facing filters and a deterministic urgency sort."""

    selected = add_order_action(frame)
    if action == "Needs reorder":
        selected = selected[selected["order_action"].eq("REORDER")]
    elif action == "No order":
        selected = selected[selected["order_action"].eq("NO ORDER")]
    elif action != "All decisions":
        raise ValueError(f"Unsupported action filter: {action}")
    if store_id != "All":
        selected = selected[selected["store_id"].eq(store_id)]
    if dept_id != "All":
        selected = selected[selected["dept_id"].eq(dept_id)]
    sort_columns = [
        column
        for column in ("priority_score", "stockout_risk_pct", "recommended_order_qty")
        if column in selected.columns
    ]
    if sort_columns:
        selected = selected.sort_values(sort_columns, ascending=[False] * len(sort_columns))
    return selected.reset_index(drop=True)


def recommendation_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a compact recommendation table with business-readable labels."""

    selected = add_order_action(frame) if "order_action" not in frame.columns else frame.copy()
    available = [column for column in RECOMMENDATION_COLUMNS if column in selected.columns]
    return selected[available].rename(columns=RECOMMENDATION_COLUMNS).reset_index(drop=True)


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a displayed table without adding a synthetic index column."""

    return frame.to_csv(index=False).encode("utf-8")


def shortage_reduction_pct(frame: pd.DataFrame) -> float:
    """Calculate expected shortage reduction, leaving zero-denominator cases safe."""

    before = float(frame["expected_shortage_without_order"].sum())
    after = float(frame["expected_shortage_with_recommendation"].sum())
    return 0.0 if before <= 0 else max(0.0, (before - after) / before)


def scenario_explanations(
    base: Mapping[str, Any],
    revised: Mapping[str, Any],
    original_inputs: Mapping[str, Any],
    new_inputs: Mapping[str, Any],
) -> list[str]:
    """Explain the main reasons a scenario decision changed in plain language."""

    explanations: list[str] = []
    base_position = float(base["inventory_position"])
    revised_position = float(revised["inventory_position"])
    if revised_position != base_position:
        direction = "increased" if revised_position > base_position else "decreased"
        explanations.append(
            f"Inventory position {direction} from {base_position:.1f} to {revised_position:.1f} units."
        )
    old_lead = int(original_inputs["lead_time_days"])
    new_lead = int(new_inputs["lead_time_days"])
    if new_lead != old_lead:
        direction = "more" if new_lead > old_lead else "less"
        explanations.append(
            f"The {new_lead}-day lead time creates {direction} demand exposure than the saved {old_lead}-day assumption."
        )
    old_service = float(original_inputs["service_level"])
    new_service = float(new_inputs["service_level"])
    if new_service != old_service:
        direction = "raises" if new_service > old_service else "reduces"
        explanations.append(
            f"Changing the service target from {old_service:.1%} to {new_service:.1%} {direction} the uncertainty buffer."
        )
    old_review = int(original_inputs["review_period_days"])
    new_review = int(new_inputs["review_period_days"])
    if new_review != old_review:
        explanations.append(
            f"The review period changed from {old_review} to {new_review} days, changing the target stock horizon."
        )
    cost_fields = (
        "holding_cost_per_unit_per_day",
        "stockout_cost_per_unit",
        "fixed_order_cost",
    )
    if any(
        field in original_inputs
        and field in new_inputs
        and float(original_inputs[field]) != float(new_inputs[field])
        for field in cost_fields
    ):
        explanations.append(
            "The cost assumptions changed the cost-optimized alternative; they do not change the service-level order rule."
        )
    if not explanations:
        explanations.append("The operational assumptions match the saved recommendation, so the decision is unchanged.")
    explanations.append(
        f"The recommended order is {int(revised['recommended_order_qty']):,} units, compared with "
        f"{int(base['recommended_order_qty']):,} units in the saved decision."
    )
    return explanations
