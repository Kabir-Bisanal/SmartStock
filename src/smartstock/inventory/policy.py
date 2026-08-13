"""Transparent inventory formulas and deterministic business classifications."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np


def load_policy(path: Path | str) -> dict[str, Any]:
    """Load the tracked policy and expose validated scalar defaults."""

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    values = {name: entry["value"] for name, entry in manifest["defaults"].items()}
    policy = {**manifest, **values}
    validate_policy(policy)
    return policy


def validate_policy(policy: dict[str, Any]) -> None:
    """Reject invalid service, time, cost, and scenario assumptions."""

    if not 0.5 < float(policy["service_level"]) < 1:
        raise ValueError("service_level must be between 0.5 and 1.")
    if not 1 <= int(policy["lead_time_days"]) <= 30:
        raise ValueError("lead_time_days must be between 1 and 30.")
    if int(policy["review_period_days"]) < 0:
        raise ValueError("review_period_days cannot be negative.")
    for name in (
        "holding_cost_per_unit_per_day",
        "stockout_cost_per_unit",
        "fixed_order_cost",
    ):
        if float(policy[name]) < 0:
            raise ValueError(f"{name} cannot be negative.")
    if int(policy["scenario_count"]) <= 0:
        raise ValueError("scenario_count must be positive.")


def inventory_position(on_hand: float, on_order: float, backorders: float) -> float:
    """Return on-hand plus on-order stock less existing backorders."""

    values = {"on_hand": on_hand, "on_order": on_order, "backorders": backorders}
    if any(not np.isfinite(float(value)) or float(value) < 0 for value in values.values()):
        raise ValueError("on_hand, on_order, and backorders must be finite and nonnegative.")
    return float(on_hand) + float(on_order) - float(backorders)


def safety_stock(service_level: float, sigma_daily: float, lead_time_days: int) -> float:
    """Calculate normal-approximation protection stock for independent daily errors."""

    if not 0.5 < float(service_level) < 1:
        raise ValueError("service_level must be between 0.5 and 1.")
    if not np.isfinite(float(sigma_daily)) or float(sigma_daily) < 0:
        raise ValueError("sigma_daily must be finite and nonnegative.")
    if int(lead_time_days) <= 0:
        raise ValueError("lead_time_days must be positive.")
    z_value = NormalDist().inv_cdf(float(service_level))
    return float(z_value * float(sigma_daily) * math.sqrt(int(lead_time_days)))


def reorder_point(lead_time_demand: float, safety_stock_units: float) -> float:
    """Return expected lead-time demand plus safety stock."""

    if lead_time_demand < 0 or safety_stock_units < 0:
        raise ValueError("Demand and safety stock must be nonnegative.")
    return float(lead_time_demand + safety_stock_units)


def target_stock_level(
    planning_horizon_demand: float, safety_stock_units: float
) -> float:
    """Return demand over lead time plus review period, protected by safety stock."""

    if planning_horizon_demand < 0 or safety_stock_units < 0:
        raise ValueError("Demand and safety stock must be nonnegative.")
    return float(planning_horizon_demand + safety_stock_units)


def reorder_quantity(target_level: float, position: float) -> tuple[float, int]:
    """Return continuous and physical-unit service-level reorder recommendations."""

    if target_level < 0:
        raise ValueError("target_level cannot be negative.")
    raw = max(0.0, float(target_level) - float(position))
    return raw, int(math.ceil(raw))


def days_of_supply(
    position: float, mean_daily_demand: float, *, epsilon: float = 1e-9
) -> tuple[float, str]:
    """Estimate coverage days and provide a safe explanatory label."""

    if mean_daily_demand < 0:
        raise ValueError("mean_daily_demand cannot be negative.")
    if mean_daily_demand <= epsilon:
        return float("nan"), "NO EXPECTED DEMAND"
    days = max(float(position), 0.0) / float(mean_daily_demand)
    return float(days), f"{days:.1f} days"


def stockout_probability(
    inventory_position_units: float,
    lead_time_demand: float,
    sigma_daily: float,
    lead_time_days: int,
    *,
    epsilon: float = 1e-9,
) -> float:
    """Estimate P(lead-time demand > inventory position) under a normal approximation."""

    if lead_time_demand < 0 or sigma_daily < 0 or lead_time_days <= 0:
        raise ValueError("Stockout-risk inputs are invalid.")
    sigma_lead_time = float(sigma_daily) * math.sqrt(int(lead_time_days))
    if sigma_lead_time <= epsilon:
        return 1.0 if lead_time_demand > inventory_position_units else 0.0
    distribution = NormalDist(mu=float(lead_time_demand), sigma=sigma_lead_time)
    return float(np.clip(1.0 - distribution.cdf(float(inventory_position_units)), 0.0, 1.0))


def stock_status(
    *,
    position: float,
    risk: float,
    reorder_point_units: float,
    days_supply: float,
    target_level: float,
    low_stock_days: float,
    overstock_multiplier: float,
    critical_risk_threshold: float,
) -> str:
    """Assign the first matching business status in the tracked precedence order."""

    if position <= 0:
        return "STOCKOUT"
    if risk >= critical_risk_threshold:
        return "CRITICAL"
    if position <= reorder_point_units:
        return "REORDER NOW"
    if np.isfinite(days_supply) and days_supply < low_stock_days:
        return "LOW"
    if target_level > 0 and position > overstock_multiplier * target_level:
        return "OVERSTOCK"
    return "HEALTHY"


def reorder_priority(
    *,
    risk: float,
    position: float,
    reorder_point_units: float,
    target_level: float,
    recommended_order_qty_raw: float,
    weights: dict[str, float],
    thresholds: dict[str, float],
) -> tuple[float, str]:
    """Return a transparent normalized priority score and label."""

    if recommended_order_qty_raw <= 0:
        return 0.0, "NONE"
    reorder_gap = max(reorder_point_units - position, 0.0) / max(reorder_point_units, 1.0)
    target_gap = max(target_level - position, 0.0) / max(target_level, 1.0)
    score = 100 * (
        float(weights["stockout_risk"]) * float(np.clip(risk, 0.0, 1.0))
        + float(weights["reorder_point_gap_ratio"]) * float(np.clip(reorder_gap, 0.0, 1.0))
        + float(weights["target_shortage_ratio"]) * float(np.clip(target_gap, 0.0, 1.0))
    )
    score = float(np.clip(score, 0.0, 100.0))
    if score >= float(thresholds["urgent"]):
        label = "URGENT"
    elif score >= float(thresholds["high"]):
        label = "HIGH"
    elif score >= float(thresholds["medium"]):
        label = "MEDIUM"
    else:
        label = "LOW"
    return score, label
