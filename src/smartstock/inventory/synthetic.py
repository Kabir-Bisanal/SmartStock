"""Deterministic synthetic inventory balances for the M5-based educational project."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from smartstock.inventory.policy import reorder_point, safety_stock, target_stock_level
except ModuleNotFoundError:
    from policy import reorder_point, safety_stock, target_stock_level


SERIES_KEYS = ["store_id", "item_id"]


def _profile_assignments(count: int, probabilities: dict[str, float], rng: np.random.Generator) -> np.ndarray:
    """Create fixed proportions first, then seeded-shuffle them without outcome feedback."""

    names = list(probabilities)
    raw = np.asarray([probabilities[name] * count for name in names])
    counts = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - counts))[: count - int(counts.sum())]:
        counts[index] += 1
    labels = np.concatenate(
        [np.repeat(name, profile_count) for name, profile_count in zip(names, counts, strict=True)]
    )
    rng.shuffle(labels)
    return labels


def generate_synthetic_inventory(
    forecasts: pd.DataFrame,
    calibration: pd.DataFrame,
    policy: dict[str, Any],
) -> pd.DataFrame:
    """Generate demo balances from forecast coverage profiles using fixed seed 42."""

    forecast_groups = {
        (str(store_id), str(item_id)): group.sort_values("horizon_day")
        for (store_id, item_id), group in forecasts.groupby(SERIES_KEYS, observed=True, sort=True)
    }
    calibration_index = calibration.set_index(SERIES_KEYS)
    keys = sorted(forecast_groups)
    expected = 300
    if len(keys) != expected:
        raise ValueError(f"Synthetic inventory expects {expected} series; found {len(keys)}.")

    settings = policy["synthetic_inventory"]
    rng = np.random.default_rng(int(settings["generation_seed"]))
    profiles = _profile_assignments(len(keys), settings["profile_probabilities"], rng)
    lead_times = rng.choice(
        settings["lead_time_choices_days"],
        size=len(keys),
        p=settings["lead_time_probabilities"],
    )
    service_levels = rng.choice(
        settings["service_level_choices"],
        size=len(keys),
        p=settings["service_level_probabilities"],
    )
    rows: list[dict[str, Any]] = []
    for index, key in enumerate(keys):
        group = forecast_groups[key]
        daily = group["forecast"].to_numpy(dtype="float64")
        lead_time = int(lead_times[index])
        service_level = float(service_levels[index])
        review = int(policy["review_period_days"])
        if lead_time + review > len(daily):
            raise ValueError("Synthetic profile planning horizon exceeds available forecasts.")
        sigma = float(calibration_index.loc[key, "sigma_daily"])
        lead_demand = float(daily[:lead_time].sum())
        planning_demand = float(daily[: lead_time + review].sum())
        safety = safety_stock(service_level, sigma, lead_time)
        reorder = reorder_point(lead_demand, safety)
        target = target_stock_level(planning_demand, safety)
        mean_daily = float(daily.mean())
        profile = str(profiles[index])

        if profile == "stockout":
            desired_position = -float(rng.integers(0, 4))
        elif profile == "critical":
            uncertainty = sigma * np.sqrt(lead_time)
            desired_position = max(
                0.05 * lead_demand,
                lead_demand - rng.uniform(0.8, 1.5) * max(uncertainty, lead_demand * 0.2),
            )
        elif profile == "reorder":
            desired_position = lead_demand + rng.uniform(0.05, 0.85) * safety
        elif profile == "low":
            lower = reorder + max(0.05 * mean_daily, 0.1)
            upper = min(target, mean_daily * float(policy["low_stock_days"]) * 0.95)
            desired_position = rng.uniform(lower, upper) if upper > lower else lower
        elif profile == "healthy":
            desired_position = rng.uniform(0.9, 1.15) * target
        elif profile == "overstock":
            desired_position = rng.uniform(1.6, 2.2) * target
        else:  # pragma: no cover - guarded by tracked configuration.
            raise ValueError(f"Unknown synthetic inventory profile: {profile}")

        if profile == "stockout":
            on_hand = 0
            on_order = 0
            backorders = int(abs(round(desired_position)))
        else:
            backorders = int(rng.integers(1, 3)) if rng.random() < 0.1 else 0
            on_order = int(round(max(desired_position, 0.0) * rng.uniform(0.0, 0.25)))
            on_hand = int(max(0, round(desired_position - on_order + backorders)))

        metadata = group.iloc[0]
        rows.append(
            {
                "item_id": key[1],
                "store_id": key[0],
                "dept_id": str(metadata["dept_id"]),
                "demand_band": str(metadata["demand_band"]),
                "on_hand": on_hand,
                "on_order": on_order,
                "backorders": backorders,
                "lead_time_days": lead_time,
                "service_level": service_level,
                "holding_cost_per_unit_per_day": float(
                    policy["holding_cost_per_unit_per_day"]
                ),
                "stockout_cost_per_unit": float(policy["stockout_cost_per_unit"]),
                "fixed_order_cost": float(policy["fixed_order_cost"]),
                "inventory_source": "synthetic_demo",
                "synthetic_profile_plan": profile,
            }
        )
    snapshot = pd.DataFrame(rows).sort_values(SERIES_KEYS, kind="stable").reset_index(drop=True)
    if snapshot.duplicated(SERIES_KEYS).any() or len(snapshot) != expected:
        raise RuntimeError("Synthetic inventory does not have exactly one row per series.")
    return snapshot
