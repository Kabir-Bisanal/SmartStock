"""Focused tests for reusable Stage 5 analytical helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from smartstock.analysis.stage5_eda import (  # noqa: E402
    compute_series_metrics,
    detect_weekly_price_changes,
    identify_active_periods,
    store_specific_snap_flag,
)


class Stage5EdaTests(unittest.TestCase):
    def test_active_period_uses_first_known_price_before_first_sale(self) -> None:
        data = pd.DataFrame(
            {
                "store_id": ["CA_1"] * 4,
                "item_id": ["FOODS_1_001"] * 4,
                "date": pd.to_datetime(["2011-01-01", "2011-01-02", "2011-01-03", "2011-01-04"]),
                "sell_price": [None, 2.0, 2.0, 2.0],
                "sales": [0, 0, 0, 1],
            }
        )

        result = identify_active_periods(data)

        self.assertEqual(result["launch_date"].iloc[0], pd.Timestamp("2011-01-02"))
        self.assertEqual(result["first_positive_sale_date"].iloc[0], pd.Timestamp("2011-01-04"))
        self.assertEqual(result["prelaunch_flag"].tolist(), [True, False, False, False])
        self.assertEqual(result["active_flag"].tolist(), [False, True, True, True])

    def test_active_period_falls_back_to_first_positive_sale(self) -> None:
        data = pd.DataFrame(
            {
                "store_id": ["TX_2"] * 3,
                "item_id": ["FOODS_2_001"] * 3,
                "date": pd.to_datetime(["2011-01-01", "2011-01-02", "2011-01-03"]),
                "sell_price": [None, None, None],
                "sales": [0, 2, 0],
            }
        )

        result = identify_active_periods(data)

        self.assertEqual(result["launch_date"].iloc[0], pd.Timestamp("2011-01-02"))
        self.assertEqual(result["prelaunch_flag"].tolist(), [True, False, False])

    def test_store_specific_snap_flag_chooses_state_column(self) -> None:
        data = pd.DataFrame(
            {
                "store_id": ["CA_1", "TX_2", "WI_3"],
                "snap_CA": [1, 1, 1],
                "snap_TX": [0, 0, 0],
                "snap_WI": [0, 0, 1],
            }
        )

        result = store_specific_snap_flag(data)

        self.assertEqual(result.tolist(), [1, 0, 1])

    def test_weekly_price_change_detection_is_within_item_store(self) -> None:
        data = pd.DataFrame(
            {
                "store_id": ["CA_1"] * 4 + ["TX_2"] * 2,
                "item_id": ["FOODS_1_001"] * 6,
                "dept_id": ["FOODS_1"] * 6,
                "wm_yr_wk": [1, 1, 2, 3, 1, 2],
                "sell_price": [2.0, 2.0, 1.5, 2.25, 3.0, 3.0],
                "active_flag": [True] * 6,
            }
        )

        result = detect_weekly_price_changes(data)
        ca = result[result["store_id"].eq("CA_1")]
        tx = result[result["store_id"].eq("TX_2")]

        self.assertEqual(ca["change_direction"].tolist(), ["first observation", "decrease", "increase"])
        self.assertEqual(tx["change_direction"].tolist(), ["first observation", "unchanged"])

    def test_series_metrics_measure_active_zero_prevalence_and_interval(self) -> None:
        data = pd.DataFrame(
            {
                "store_id": ["WI_3"] * 5,
                "item_id": ["FOODS_3_001"] * 5,
                "dept_id": ["FOODS_3"] * 5,
                "demand_band": ["low"] * 5,
                "date": pd.date_range("2011-01-01", periods=5, freq="D"),
                "sales": [0, 2, 0, 0, 3],
                "active_flag": [True] * 5,
            }
        )

        metrics = compute_series_metrics(data).iloc[0]

        self.assertEqual(metrics["active_days"], 5)
        self.assertEqual(metrics["positive_days"], 2)
        self.assertAlmostEqual(metrics["active_zero_pct"], 60.0)
        self.assertAlmostEqual(metrics["average_demand_interval"], 2.5)
        self.assertAlmostEqual(metrics["mean_positive_gap_days"], 3.0)


if __name__ == "__main__":
    unittest.main()
