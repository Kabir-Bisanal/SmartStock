"""Focused tests for Stage 7 baselines, metrics, and final-test safeguards."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from smartstock.models.baseline_evaluation import (  # noqa: E402
    assert_window_unlocked,
    generate_fold_predictions,
)
from smartstock.models.baselines import (  # noqa: E402
    croston_sba_forecast,
    last_value_forecast,
    mean_28_forecast,
    seasonal_naive_7_forecast,
    zero_forecast,
)
from smartstock.models.metrics import basic_metrics, rmsse_scale  # noqa: E402


class Stage7BaselineTests(unittest.TestCase):
    def test_zero_forecast_has_requested_length(self) -> None:
        forecast = zero_forecast([1, 0, 2], horizon=5)
        self.assertEqual(forecast.tolist(), [0, 0, 0, 0, 0])

    def test_last_value_repeats_final_history_value(self) -> None:
        forecast = last_value_forecast([1, 3, 7], horizon=4)
        self.assertEqual(forecast.tolist(), [7, 7, 7, 7])

    def test_seasonal_naive_repeats_preorigin_week(self) -> None:
        forecast = seasonal_naive_7_forecast([1, 2, 3, 4, 5, 6, 7], horizon=10)
        self.assertEqual(forecast.tolist(), [1, 2, 3, 4, 5, 6, 7, 1, 2, 3])

    def test_mean_28_uses_only_final_28_values(self) -> None:
        history = np.arange(1, 31, dtype=float)
        forecast = mean_28_forecast(history, horizon=3)
        expected = np.arange(3, 31, dtype=float).mean()
        self.assertTrue(np.allclose(forecast, expected))
        self.assertNotAlmostEqual(float(forecast[0]), history.mean())

    def test_croston_sba_is_deterministic(self) -> None:
        forecast = croston_sba_forecast([0, 2, 0, 0, 4], horizon=3, alpha=0.1)
        expected = (1 - 0.1 / 2) * 2.2 / 2.1
        self.assertTrue(np.allclose(forecast, expected))

    def test_croston_all_zero_history_returns_zero(self) -> None:
        self.assertEqual(croston_sba_forecast([0, 0, 0], horizon=2).tolist(), [0, 0])

    def test_changing_validation_actuals_does_not_change_forecasts(self) -> None:
        dates = pd.date_range("2016-01-01", periods=60)
        data = pd.DataFrame(
            {
                "date": dates,
                "store_id": ["CA_1"] * 60,
                "item_id": ["FOODS_1_001"] * 60,
                "dept_id": ["FOODS_1"] * 60,
                "demand_band": ["medium"] * 60,
                "is_feature_ready": [True] * 60,
                "sales": list(range(1, 31)) + [5] * 30,
            }
        )
        fold = {"name": "validation_fold_1", "start_date": "2016-01-31", "end_date": "2016-02-29"}
        manifest = {"locked_final_test": {"start_date": "2016-03-01", "end_date": "2016-03-30"}}

        original, _ = generate_fold_predictions(data, fold, manifest)
        changed = data.copy()
        changed.loc[changed["date"].ge("2016-01-31"), "sales"] = 999
        modified, _ = generate_fold_predictions(changed, fold, manifest)

        self.assertTrue(np.allclose(original["forecast"], modified["forecast"]))
        self.assertFalse(np.array_equal(original["actual"], modified["actual"]))

    def test_rmsse_scale_uses_training_sequence_only(self) -> None:
        training = np.array([1, 3, 2], dtype=float)
        validation = np.array([999, 999], dtype=float)
        self.assertEqual(rmsse_scale(training), 2.5)
        validation[:] = 0
        self.assertEqual(rmsse_scale(training), 2.5)
        self.assertTrue(np.isnan(rmsse_scale([4, 4, 4])))

    def test_wape_zero_denominator_is_undefined(self) -> None:
        result = basic_metrics(np.array([0, 0]), np.array([1, 2]))
        self.assertTrue(np.isnan(result["wape"]))

    def test_validation_boundaries_match_frozen_manifest(self) -> None:
        manifest = json.loads((PROJECT_ROOT / "config" / "v1_validation.json").read_text(encoding="utf-8"))
        self.assertEqual(
            [(fold["start_date"], fold["end_date"]) for fold in manifest["validation_folds"]],
            [
                ("2016-01-24", "2016-02-22"),
                ("2016-02-23", "2016-03-23"),
                ("2016-03-24", "2016-04-22"),
            ],
        )

    def test_final_test_overlap_is_blocked(self) -> None:
        manifest = {"locked_final_test": {"start_date": "2016-04-23", "end_date": "2016-05-22"}}
        with self.assertRaisesRegex(ValueError, "final-test lock"):
            assert_window_unlocked(pd.Timestamp("2016-04-23"), pd.Timestamp("2016-05-22"), manifest)
        assert_window_unlocked(pd.Timestamp("2016-03-24"), pd.Timestamp("2016-04-22"), manifest)


if __name__ == "__main__":
    unittest.main()
