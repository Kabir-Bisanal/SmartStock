"""Reusable SmartStock inventory recommendation API for a batch of item-store series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from smartstock.inventory.optimizer import generate_demand_scenarios, optimize_order_quantity
    from smartstock.inventory.policy import (
        days_of_supply,
        inventory_position,
        reorder_point,
        reorder_priority,
        reorder_quantity,
        safety_stock,
        stock_status,
        stockout_probability,
        target_stock_level,
    )
except ModuleNotFoundError:
    from optimizer import generate_demand_scenarios, optimize_order_quantity
    from policy import (
        days_of_supply,
        inventory_position,
        reorder_point,
        reorder_priority,
        reorder_quantity,
        safety_stock,
        stock_status,
        stockout_probability,
        target_stock_level,
    )


SERIES_KEYS = ["store_id", "item_id"]


def recommend_inventory_for_series(
    daily_forecast: np.ndarray | list[float],
    inventory_state: dict[str, Any],
    calibration: dict[str, Any],
    policy: dict[str, Any],
    *,
    scenario_seed: int,
) -> dict[str, Any]:
    """Calculate service-level and cost-aware decisions for one item-store series."""

    daily = np.asarray(daily_forecast, dtype="float64")
    if daily.ndim != 1 or daily.size == 0 or np.any(daily < 0):
        raise ValueError("daily_forecast must be a nonempty nonnegative sequence.")
    lead_time = int(inventory_state.get("lead_time_days", policy["lead_time_days"]))
    review = int(policy["review_period_days"])
    planning_days = lead_time + review
    if lead_time < 1 or lead_time > 30:
        raise ValueError("lead_time_days must be between 1 and 30.")
    if planning_days > len(daily):
        raise ValueError(
            f"Need {planning_days} forecast days for lead time plus review; only {len(daily)} available."
        )
    service_level = float(inventory_state.get("service_level", policy["service_level"]))
    sigma_daily = float(calibration["sigma_daily"])
    residual_mean = float(calibration["residual_mean_used"])
    position = inventory_position(
        inventory_state["on_hand"], inventory_state["on_order"], inventory_state["backorders"]
    )
    lead_demand = float(daily[:lead_time].sum())
    planning_demand = float(daily[:planning_days].sum())
    safety = safety_stock(service_level, sigma_daily, lead_time)
    reorder_level = reorder_point(lead_demand, safety)
    target_level = target_stock_level(planning_demand, safety)
    raw_order, physical_order = reorder_quantity(target_level, position)
    mean_daily = float(daily.mean())
    supply_days, supply_label = days_of_supply(position, mean_daily)
    risk = stockout_probability(position, lead_demand, sigma_daily, lead_time)
    status = stock_status(
        position=position,
        risk=risk,
        reorder_point_units=reorder_level,
        days_supply=supply_days,
        target_level=target_level,
        low_stock_days=float(policy["low_stock_days"]),
        overstock_multiplier=float(policy["overstock_multiplier"]),
        critical_risk_threshold=float(policy["stock_status"]["critical_risk_threshold"]),
    )
    score, priority = reorder_priority(
        risk=risk,
        position=position,
        reorder_point_units=reorder_level,
        target_level=target_level,
        recommended_order_qty_raw=raw_order,
        weights=policy["priority"]["weights"],
        thresholds=policy["priority"]["thresholds"],
    )
    scenarios = generate_demand_scenarios(
        daily[:planning_days],
        residual_mean,
        sigma_daily,
        scenario_count=int(policy["scenario_count"]),
        random_seed=int(scenario_seed),
    )
    cost = optimize_order_quantity(
        inventory_position=position,
        service_order_qty=physical_order,
        scenario_demand=scenarios,
        holding_cost_per_unit_per_day=float(
            inventory_state["holding_cost_per_unit_per_day"]
        ),
        stockout_cost_per_unit=float(inventory_state["stockout_cost_per_unit"]),
        fixed_order_cost=float(inventory_state["fixed_order_cost"]),
        planning_horizon_days=planning_days,
        maximum_candidate_points=int(
            policy["cost_optimization"]["maximum_candidate_points"]
        ),
        upper_quantile=float(policy["cost_optimization"]["candidate_upper_quantile"]),
        upper_buffer_multiplier=float(
            policy["cost_optimization"]["candidate_upper_buffer_multiplier"]
        ),
    )
    return {
        "inventory_position": position,
        "forecast_1d": float(daily[:1].sum()),
        "forecast_7d": float(daily[: min(7, len(daily))].sum()),
        "forecast_30d": float(daily[: min(30, len(daily))].sum()),
        "mean_daily_forecast": mean_daily,
        "lead_time_demand": lead_demand,
        "planning_horizon_days": planning_days,
        "planning_horizon_demand": planning_demand,
        "sigma_daily": sigma_daily,
        "residual_mean_used": residual_mean,
        "safety_stock": safety,
        "reorder_point": reorder_level,
        "target_stock_level": target_level,
        "days_of_supply": supply_days,
        "days_of_supply_label": supply_label,
        "stockout_risk": risk,
        "stockout_risk_pct": 100 * risk,
        "stock_status": status,
        "recommended_order_qty_raw": raw_order,
        "recommended_order_qty": physical_order,
        "priority_score": score,
        "priority_label": priority,
        **cost,
    }


def recommend_inventory(
    forecasts: pd.DataFrame,
    inventory_state: pd.DataFrame,
    calibration: pd.DataFrame,
    policy: dict[str, Any],
) -> pd.DataFrame:
    """Return exactly one application-ready recommendation per item-store series."""

    for name, frame in (("inventory_state", inventory_state), ("calibration", calibration)):
        if frame.duplicated(SERIES_KEYS).any():
            raise ValueError(f"{name} contains duplicate item-store keys.")
    groups = {
        (str(store_id), str(item_id)): group.sort_values("horizon_day")
        for (store_id, item_id), group in forecasts.groupby(SERIES_KEYS, observed=True, sort=True)
    }
    states = inventory_state.set_index(SERIES_KEYS)
    calibrations = calibration.set_index(SERIES_KEYS)
    if set(groups) != set(states.index) or set(groups) != set(calibrations.index):
        raise ValueError("Forecast, inventory, and calibration series keys do not match exactly.")
    rows: list[dict[str, Any]] = []
    for index, key in enumerate(sorted(groups)):
        group = groups[key]
        state = states.loc[key].to_dict()
        calibrated = calibrations.loc[key].to_dict()
        decision = recommend_inventory_for_series(
            group["forecast"].to_numpy(dtype="float64"),
            state,
            calibrated,
            policy,
            scenario_seed=int(policy["random_seed"]) + index,
        )
        rows.append(
            {
                "item_id": key[1],
                "store_id": key[0],
                "dept_id": str(group["dept_id"].iloc[0]),
                "demand_band": str(group["demand_band"].iloc[0]),
                "on_hand": float(state["on_hand"]),
                "on_order": float(state["on_order"]),
                "backorders": float(state["backorders"]),
                "lead_time_days": int(state["lead_time_days"]),
                "service_level": float(state["service_level"]),
                "holding_cost_per_unit_per_day": float(
                    state["holding_cost_per_unit_per_day"]
                ),
                "stockout_cost_per_unit": float(state["stockout_cost_per_unit"]),
                "fixed_order_cost": float(state["fixed_order_cost"]),
                "calibration_level": str(calibrated["calibration_level"]),
                "calibration_count": int(calibrated["calibration_count_used"]),
                "residual_rmse": float(calibrated["residual_rmse_used"]),
                "intermittency_class": str(calibrated["intermittency_class"]),
                "inventory_source": str(state["inventory_source"]),
                "synthetic_profile_plan": str(state["synthetic_profile_plan"]),
                **decision,
            }
        )
    result = pd.DataFrame(rows).sort_values(SERIES_KEYS, kind="stable").reset_index(drop=True)
    if len(result) != 300 or result.duplicated(SERIES_KEYS).any():
        raise RuntimeError("Batch recommendations must contain 300 unique item-store rows.")
    return result
