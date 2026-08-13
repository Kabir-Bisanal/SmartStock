"""Unit and generated-output integrity tests for SmartStock Stage 10."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smartstock.inventory.engine import recommend_inventory_for_series  # noqa: E402
from smartstock.inventory.optimizer import (  # noqa: E402
    generate_demand_scenarios,
    optimize_order_quantity,
)
from smartstock.inventory.policy import (  # noqa: E402
    days_of_supply,
    inventory_position,
    load_policy,
    reorder_point,
    reorder_quantity,
    safety_stock,
    stockout_probability,
)
from smartstock.inventory.synthetic import generate_synthetic_inventory  # noqa: E402
from smartstock.models.baseline_evaluation import file_sha256  # noqa: E402
from smartstock.models.deployment_forecaster import forecast_deployment_demand  # noqa: E402


POLICY = load_policy(PROJECT_ROOT / "config" / "v1_inventory_policy.json")


def one_series_state(**changes: float) -> dict:
    state = {
        "on_hand": 50,
        "on_order": 5,
        "backorders": 2,
        "lead_time_days": 7,
        "service_level": 0.95,
        "holding_cost_per_unit_per_day": 0.02,
        "stockout_cost_per_unit": 5.0,
        "fixed_order_cost": 25.0,
    }
    state.update(changes)
    return state


CALIBRATION = {
    "sigma_daily": 2.0,
    "residual_mean_used": 0.1,
}


def full_forecast_fixture() -> pd.DataFrame:
    rows = []
    for series_index in range(300):
        store = ("CA_1", "TX_2", "WI_3")[series_index % 3]
        item = f"FOODS_1_{series_index:03d}"
        for day in range(1, 31):
            rows.append(
                {
                    "store_id": store,
                    "item_id": item,
                    "dept_id": "FOODS_1",
                    "demand_band": "medium",
                    "horizon_day": day,
                    "forecast": 1.0 + (series_index % 5) * 0.2,
                }
            )
    return pd.DataFrame(rows)


def full_calibration_fixture(forecasts: pd.DataFrame) -> pd.DataFrame:
    return forecasts[["store_id", "item_id", "dept_id", "demand_band"]].drop_duplicates().assign(
        intermittency_class="regular",
        calibration_level="item_store",
        calibration_count_used=90,
        residual_mean_used=0.0,
        sigma_daily=1.5,
        residual_rmse_used=1.5,
    )


class Stage10InventoryTests(unittest.TestCase):
    def test_inventory_position_formula_and_validation(self) -> None:
        self.assertEqual(inventory_position(10, 4, 3), 11)
        with self.assertRaises(ValueError):
            inventory_position(-1, 0, 0)

    def test_safety_stock_increases_with_service_and_lead_time(self) -> None:
        base = safety_stock(0.90, 2.0, 3)
        self.assertGreater(safety_stock(0.99, 2.0, 3), base)
        self.assertGreater(safety_stock(0.90, 2.0, 10), base)

    def test_reorder_point_and_quantity_formulas(self) -> None:
        self.assertEqual(reorder_point(20, 5), 25)
        raw, rounded = reorder_quantity(30.2, 20)
        self.assertAlmostEqual(raw, 10.2)
        self.assertEqual(rounded, 11)

    def test_no_reorder_when_sufficiently_stocked(self) -> None:
        decision = recommend_inventory_for_series(
            [1.0] * 30,
            one_series_state(on_hand=100, on_order=0, backorders=0),
            CALIBRATION,
            POLICY,
            scenario_seed=42,
        )
        self.assertEqual(decision["recommended_order_qty"], 0)

    def test_positive_reorder_when_below_target(self) -> None:
        decision = recommend_inventory_for_series(
            [2.0] * 30,
            one_series_state(on_hand=1, on_order=0, backorders=0),
            CALIBRATION,
            POLICY,
            scenario_seed=42,
        )
        self.assertGreater(decision["recommended_order_qty"], 0)
        self.assertAlmostEqual(
            decision["reorder_point"],
            decision["lead_time_demand"] + decision["safety_stock"],
        )

    def test_zero_demand_and_days_of_supply_are_safe(self) -> None:
        supply, label = days_of_supply(10, 0)
        self.assertTrue(np.isnan(supply))
        self.assertEqual(label, "NO EXPECTED DEMAND")
        decision = recommend_inventory_for_series(
            [0.0] * 30,
            one_series_state(on_hand=10, on_order=0, backorders=0),
            {"sigma_daily": 0.0, "residual_mean_used": 0.0},
            POLICY,
            scenario_seed=42,
        )
        self.assertEqual(decision["recommended_order_qty"], 0)
        self.assertEqual(decision["stockout_risk"], 0)

    def test_stockout_risk_is_bounded_and_rises_as_inventory_falls(self) -> None:
        low_inventory = stockout_probability(5, 20, 3, 7)
        high_inventory = stockout_probability(30, 20, 3, 7)
        self.assertGreater(low_inventory, high_inventory)
        self.assertGreaterEqual(high_inventory, 0)
        self.assertLessEqual(low_inventory, 1)

    def test_cost_optimizer_is_deterministic_and_nonnegative(self) -> None:
        first_scenarios = generate_demand_scenarios(
            [2.0] * 14, 0.1, 1.5, scenario_count=200, random_seed=42
        )
        second_scenarios = generate_demand_scenarios(
            [2.0] * 14, 0.1, 1.5, scenario_count=200, random_seed=42
        )
        self.assertTrue(np.array_equal(first_scenarios, second_scenarios))
        kwargs = dict(
            inventory_position=10,
            service_order_qty=25,
            scenario_demand=first_scenarios,
            holding_cost_per_unit_per_day=0.02,
            stockout_cost_per_unit=5,
            fixed_order_cost=25,
            planning_horizon_days=14,
        )
        first = optimize_order_quantity(**kwargs)
        second = optimize_order_quantity(**kwargs)
        self.assertEqual(first["cost_optimized_order_qty"], second["cost_optimized_order_qty"])
        self.assertGreaterEqual(first["cost_optimized_order_qty"], 0)
        self.assertGreaterEqual(first["estimated_cost_savings"], -1e-9)

    def test_synthetic_inventory_generation_is_deterministic(self) -> None:
        forecasts = full_forecast_fixture()
        calibration = full_calibration_fixture(forecasts)
        first = generate_synthetic_inventory(forecasts, calibration, POLICY)
        second = generate_synthetic_inventory(forecasts, calibration, POLICY)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 300)
        self.assertFalse(first.duplicated(["store_id", "item_id"]).any())
        self.assertEqual(set(first["inventory_source"]), {"synthetic_demo"})

    def test_deployment_mean28_forecast_uses_algorithmic_dates_and_recursion(self) -> None:
        bundle = {
            "artifact_role": "post_evaluation_deployment_refit",
            "model_family": "mean_28",
            "window_days": 28,
            "trained_through": "2016-05-22",
            "series_count": 1,
            "history": {("CA_1", "FOODS_1_001"): np.arange(1, 29, dtype=float)},
            "metadata": {("CA_1", "FOODS_1_001"): {"dept_id": "FOODS_1", "demand_band": "medium"}},
        }
        forecast = forecast_deployment_demand(bundle, 7)
        self.assertEqual(len(forecast), 7)
        self.assertEqual(forecast["target_date"].min(), pd.Timestamp("2016-05-23"))
        self.assertAlmostEqual(forecast.iloc[0]["forecast"], 14.5)
        self.assertNotEqual(forecast.iloc[1]["forecast"], forecast.iloc[0]["forecast"])

    def test_stage9_evaluation_artifacts_remain_frozen(self) -> None:
        receipt = json.loads(
            (PROJECT_ROOT / "reports" / "stage9_test_evaluation_receipt.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(receipt["evaluation_count"], 1)
        self.assertEqual(
            file_sha256(PROJECT_ROOT / "config" / "v1_final_model.json"),
            receipt["frozen_config_sha256"],
        )
        self.assertEqual(
            file_sha256(PROJECT_ROOT / "models" / "smartstock_v1_forecaster.joblib"),
            receipt["model_artifact"]["sha256"],
        )

    def test_real_recommendation_output_has_300_unique_series_when_generated(self) -> None:
        path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "inventory"
            / "smartstock_v1_inventory_recommendations.csv"
        )
        if not path.exists():
            self.skipTest("Stage 10 generated output is created by the integration pipeline.")
        recommendations = pd.read_csv(path, usecols=["store_id", "item_id"])
        self.assertEqual(len(recommendations), 300)
        self.assertFalse(recommendations.duplicated(["store_id", "item_id"]).any())


if __name__ == "__main__":
    unittest.main()
