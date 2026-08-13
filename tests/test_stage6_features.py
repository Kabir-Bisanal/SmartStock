"""Focused tests for Stage 6 leakage-safe feature logic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from smartstock.features.engineering import (  # noqa: E402
    add_demand_history_features,
    add_weekly_price_features,
    filter_active_period,
    store_relevant_snap,
)
from smartstock.features.feature_definitions import (  # noqa: E402
    audit_feature_manifest,
    build_feature_manifest,
    build_validation_manifest,
)
from smartstock.features.validation import validate_manifest_dates  # noqa: E402


class Stage6FeatureTests(unittest.TestCase):
    def test_prelaunch_rows_are_excluded_and_product_age_starts_at_zero(self) -> None:
        data = pd.DataFrame(
            {
                "store_id": ["CA_1"] * 4,
                "item_id": ["FOODS_1_001"] * 4,
                "date": pd.date_range("2011-01-01", periods=4),
                "sell_price": [None, 2.0, 2.0, 2.0],
                "sales": [0, 0, 0, 1],
            }
        )

        active, summary = filter_active_period(data)

        self.assertEqual(active["date"].tolist(), list(pd.date_range("2011-01-02", periods=3)))
        self.assertEqual(active["product_age_days"].tolist(), [0, 1, 2])
        self.assertEqual(summary["prelaunch_rows_excluded"], 1)
        self.assertEqual(summary["active_zero_rows_retained"], 2)

    def test_first_positive_sale_is_only_fallback_when_no_price_exists(self) -> None:
        data = pd.DataFrame(
            {
                "store_id": ["TX_2"] * 4,
                "item_id": ["FOODS_2_001"] * 4,
                "date": pd.date_range("2011-01-01", periods=4),
                "sell_price": [None] * 4,
                "sales": [0, 0, 2, 0],
            }
        )

        active, summary = filter_active_period(data)

        self.assertEqual(active["date"].iloc[0], pd.Timestamp("2011-01-03"))
        self.assertEqual(summary["positive_sale_fallback_pairs"], 1)

    def test_lags_use_prior_values_and_do_not_cross_series(self) -> None:
        dates = pd.date_range("2011-01-01", periods=3)
        data = pd.DataFrame(
            {
                "store_id": ["CA_1"] * 3 + ["CA_1"] * 3,
                "item_id": ["A"] * 3 + ["B"] * 3,
                "date": list(dates) * 2,
                "sales": [10, 20, 30, 100, 200, 300],
            }
        )

        result = add_demand_history_features(data)
        series_a = result[result["item_id"].eq("A")]
        series_b = result[result["item_id"].eq("B")]

        self.assertEqual(series_a["sales_lag_1"].iloc[2], 20)
        self.assertNotEqual(series_a["sales_lag_1"].iloc[2], 30)
        self.assertTrue(pd.isna(series_b["sales_lag_1"].iloc[0]))

    def test_rolling_mean_excludes_target_day(self) -> None:
        data = pd.DataFrame(
            {
                "store_id": ["CA_1"] * 8,
                "item_id": ["A"] * 8,
                "date": pd.date_range("2011-01-01", periods=8),
                "sales": [10, 20, 30, 40, 50, 60, 70, 800],
            }
        )

        result = add_demand_history_features(data)

        self.assertEqual(result["sales_roll_mean_7"].iloc[7], 40)
        self.assertNotEqual(result["sales_roll_mean_7"].iloc[7], (10 + 20 + 30 + 40 + 50 + 60 + 70 + 800) / 8)

    def test_days_since_positive_only_looks_backward(self) -> None:
        data = pd.DataFrame(
            {
                "store_id": ["WI_3"] * 4,
                "item_id": ["A"] * 4,
                "date": pd.date_range("2011-01-01", periods=4),
                "sales": [1, 0, 0, 5],
            }
        )

        result = add_demand_history_features(data)

        self.assertEqual(result["days_since_last_positive_sale"].tolist()[1:], [1, 2, 3])

    def test_snap_mapping_uses_only_store_state(self) -> None:
        data = pd.DataFrame(
            {
                "store_id": ["CA_1", "TX_2", "WI_3"],
                "snap_CA": [1, 1, 1],
                "snap_TX": [0, 0, 0],
                "snap_WI": [0, 0, 1],
            }
        )

        result = store_relevant_snap(data)

        self.assertEqual(result.tolist(), [1, 0, 1])

    def test_price_features_use_completed_prior_weeks(self) -> None:
        data = pd.DataFrame(
            {
                "store_id": ["CA_1"] * 5,
                "item_id": ["A"] * 5,
                "wm_yr_wk": [1, 2, 3, 4, 5],
                "date": pd.date_range("2011-01-01", periods=5, freq="7D"),
                "sell_price": [1.0, 2.0, 3.0, 4.0, 100.0],
            }
        )

        result, _, _ = add_weekly_price_features(data)
        target = result[result["wm_yr_wk"].eq(5)].iloc[0]

        self.assertEqual(target["previous_week_price"], 4.0)
        self.assertNotEqual(target["previous_week_price"], 100.0)
        self.assertEqual(target["price_change_previous_week"], 1.0)
        self.assertAlmostEqual(target["price_pct_change_previous_week"], 100 * (4 / 3 - 1), places=4)
        self.assertAlmostEqual(target["price_vs_4week_median_lagged"], 4 / 2.5 - 1, places=4)

    def test_feature_manifest_excludes_analysis_and_future_fields(self) -> None:
        manifest = build_feature_manifest()
        audit = audit_feature_manifest(manifest)
        default_names = set(manifest["default_model_features"])

        self.assertTrue(audit["passed"])
        self.assertNotIn("demand_band", default_names)
        self.assertNotIn("known_future_sell_price", default_names)
        self.assertNotIn("sales", default_names)

    def test_validation_windows_and_test_lock_are_exact(self) -> None:
        manifest = build_validation_manifest()
        checks = validate_manifest_dates(manifest)

        self.assertTrue(all(checks.values()))
        self.assertEqual(manifest["validation_folds"][0]["start_date"], "2016-01-24")
        self.assertEqual(manifest["validation_folds"][-1]["end_date"], "2016-04-22")
        self.assertEqual(manifest["locked_final_test"]["start_date"], "2016-04-23")
        self.assertEqual(manifest["locked_final_test"]["end_date"], "2016-05-22")


if __name__ == "__main__":
    unittest.main()
