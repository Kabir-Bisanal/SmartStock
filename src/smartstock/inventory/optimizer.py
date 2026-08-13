"""Deterministic scenario generation and bounded cost-aware order optimization."""

from __future__ import annotations

from typing import Any

import numpy as np


def generate_demand_scenarios(
    daily_forecast: np.ndarray | list[float],
    residual_mean: float,
    sigma_daily: float,
    *,
    scenario_count: int,
    random_seed: int,
) -> np.ndarray:
    """Sample nonnegative planning-horizon demand using validation-calibrated errors."""

    forecast = np.asarray(daily_forecast, dtype="float64")
    if forecast.ndim != 1 or forecast.size == 0 or np.any(forecast < 0):
        raise ValueError("daily_forecast must be a nonempty nonnegative sequence.")
    if sigma_daily < 0 or scenario_count <= 0:
        raise ValueError("Scenario uncertainty and count are invalid.")
    rng = np.random.default_rng(int(random_seed))
    errors = rng.normal(
        loc=float(residual_mean),
        scale=float(sigma_daily),
        size=(int(scenario_count), len(forecast)),
    )
    daily_demand = np.maximum(forecast[None, :] + errors, 0.0)
    return daily_demand.sum(axis=1)


def expected_inventory_cost(
    order_qty: float,
    inventory_position: float,
    scenario_demand: np.ndarray,
    *,
    holding_cost_per_unit_per_day: float,
    stockout_cost_per_unit: float,
    fixed_order_cost: float,
    planning_horizon_days: int,
) -> dict[str, float]:
    """Evaluate expected ending surplus, shortage, and simplified planning cost."""

    if order_qty < 0 or planning_horizon_days <= 0:
        raise ValueError("Order quantity and planning horizon are invalid.")
    available = float(inventory_position) + float(order_qty)
    surplus = np.maximum(available - scenario_demand, 0.0)
    shortage = np.maximum(scenario_demand - available, 0.0)
    holding = surplus * float(holding_cost_per_unit_per_day) * planning_horizon_days / 2
    shortage_cost = shortage * float(stockout_cost_per_unit)
    fixed = float(fixed_order_cost) if order_qty > 0 else 0.0
    total = holding + shortage_cost + fixed
    return {
        "expected_cost": float(total.mean()),
        "expected_shortage": float(shortage.mean()),
        "expected_excess": float(surplus.mean()),
        "expected_holding_cost": float(holding.mean()),
        "expected_shortage_cost": float(shortage_cost.mean()),
        "fixed_order_cost_applied": fixed,
    }


def optimize_order_quantity(
    *,
    inventory_position: float,
    service_order_qty: int,
    scenario_demand: np.ndarray,
    holding_cost_per_unit_per_day: float,
    stockout_cost_per_unit: float,
    fixed_order_cost: float,
    planning_horizon_days: int,
    maximum_candidate_points: int = 301,
    upper_quantile: float = 0.995,
    upper_buffer_multiplier: float = 1.1,
) -> dict[str, Any]:
    """Choose the minimum-cost nonnegative quantity from a transparent finite grid."""

    if maximum_candidate_points < 2:
        raise ValueError("maximum_candidate_points must be at least two.")
    required_for_quantile = max(
        float(np.quantile(scenario_demand, upper_quantile)) - float(inventory_position), 0.0
    )
    upper = int(
        np.ceil(max(float(service_order_qty), required_for_quantile) * upper_buffer_multiplier)
    )
    if upper == 0:
        candidates = np.array([0], dtype="int64")
    elif upper + 1 <= maximum_candidate_points:
        candidates = np.arange(upper + 1, dtype="int64")
    else:
        candidates = np.unique(
            np.ceil(np.linspace(0, upper, maximum_candidate_points)).astype("int64")
        )
    candidates = np.unique(np.append(candidates, [0, int(service_order_qty)]))
    evaluations: list[tuple[int, dict[str, float]]] = []
    for candidate in candidates:
        details = expected_inventory_cost(
            float(candidate),
            inventory_position,
            scenario_demand,
            holding_cost_per_unit_per_day=holding_cost_per_unit_per_day,
            stockout_cost_per_unit=stockout_cost_per_unit,
            fixed_order_cost=fixed_order_cost,
            planning_horizon_days=planning_horizon_days,
        )
        evaluations.append((int(candidate), details))
    chosen_qty, chosen = min(evaluations, key=lambda entry: (entry[1]["expected_cost"], entry[0]))
    without = next(details for quantity, details in evaluations if quantity == 0)
    service = next(
        details for quantity, details in evaluations if quantity == int(service_order_qty)
    )
    return {
        "cost_optimized_order_qty": int(chosen_qty),
        "estimated_cost_without_order": without["expected_cost"],
        "estimated_cost_with_recommendation": chosen["expected_cost"],
        "estimated_cost_with_service_order": service["expected_cost"],
        "estimated_cost_savings": without["expected_cost"] - chosen["expected_cost"],
        "expected_shortage_without_order": without["expected_shortage"],
        "expected_excess_without_order": without["expected_excess"],
        "expected_shortage_with_recommendation": chosen["expected_shortage"],
        "expected_excess_with_recommendation": chosen["expected_excess"],
        "candidate_count": int(len(candidates)),
        "candidate_min": int(candidates.min()),
        "candidate_max": int(candidates.max()),
    }
