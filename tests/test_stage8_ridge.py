"""Focused tests for global Ridge preprocessing and recursive origin safety."""

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

from smartstock.models.recursive_forecasting import (  # noqa: E402
    DEMAND_FEATURES,
    clip_nonnegative,
    generate_recursive_predictions,
    recursive_demand_features,
)
from smartstock.models.ridge_model import (  # noqa: E402
    PROHIBITED_FEATURES,
    build_ridge_pipeline,
    prepare_feature_frame,
    select_training_rows,
    validate_ridge_config,
)


class RecordingModel:
    """Small deterministic predictor that records recursively generated inputs."""

    def __init__(self) -> None:
        self.frames: list[pd.DataFrame] = []

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self.frames.append(frame.copy())
        return frame["sales_lag_1"].to_numpy(dtype="float64") + 0.5


def small_config() -> dict:
    config = json.loads((PROJECT_ROOT / "config" / "v1_ridge.json").read_text(encoding="utf-8"))
    config["expected_series"] = 1
    return config


def synthetic_data() -> pd.DataFrame:
    dates = pd.date_range("2016-01-01", periods=60, freq="D")
    sales = np.arange(1, 61, dtype="float32")
    data = pd.DataFrame(
        {
            "date": dates,
            "availability_date": pd.Timestamp("2016-01-01"),
            "is_feature_ready": True,
            "sales": sales,
            "demand_band": "medium",
            "item_id": "FOODS_1_001",
            "store_id": "CA_1",
            "dept_id": "FOODS_1",
            "event_name_1": pd.Series([None] * 60, dtype="string"),
            "event_type_1": pd.Series([None] * 60, dtype="string"),
            "event_name_2": pd.Series([None] * 60, dtype="string"),
            "event_type_2": pd.Series([None] * 60, dtype="string"),
            "year": dates.year,
            "day_of_month": dates.day,
            "is_weekend": dates.dayofweek >= 5,
            "day_of_week_sin": np.sin(2 * np.pi * dates.dayofweek / 7),
            "day_of_week_cos": np.cos(2 * np.pi * dates.dayofweek / 7),
            "month_sin": np.sin(2 * np.pi * (dates.month - 1) / 12),
            "month_cos": np.cos(2 * np.pi * (dates.month - 1) / 12),
            "is_event": False,
            "snap_active": 0,
            "product_age_days": np.arange(60),
        }
    )
    history = pd.Series(sales)
    for lag in (1, 7, 14, 28):
        data[f"sales_lag_{lag}"] = history.shift(lag)
    shifted = history.shift(1)
    for window in (7, 14, 28):
        data[f"sales_roll_mean_{window}"] = shifted.rolling(window).mean()
    for window in (7, 28):
        data[f"sales_roll_std_{window}"] = shifted.rolling(window).std()
    data["known_future_sell_price"] = 3.99
    return data


FOLD = {"name": "validation_fold_1", "start_date": "2016-01-31", "end_date": "2016-02-29"}
MANIFEST = {"locked_final_test": {"start_date": "2016-03-01", "end_date": "2016-03-30"}}


class Stage8RidgeTests(unittest.TestCase):
    def test_training_rows_stop_before_validation(self) -> None:
        data = synthetic_data()
        training = select_training_rows(data, pd.Timestamp(FOLD["start_date"]))
        self.assertTrue(training["date"].lt(FOLD["start_date"]).all())
        self.assertEqual(training["date"].max(), pd.Timestamp("2016-01-30"))

    def test_feature_configuration_excludes_prohibited_fields(self) -> None:
        config = json.loads((PROJECT_ROOT / "config" / "v1_ridge.json").read_text(encoding="utf-8"))
        feature_manifest = json.loads((PROJECT_ROOT / "config" / "v1_features.json").read_text(encoding="utf-8"))
        checks = validate_ridge_config(config, feature_manifest)
        features = set(config["categorical_features"] + config["numeric_features"])
        self.assertTrue(all(checks.values()))
        self.assertFalse(features & PROHIBITED_FEATURES)

    def test_unknown_validation_category_does_not_crash(self) -> None:
        config = small_config()
        data = synthetic_data()
        training = data.iloc[28:35].copy()
        pipeline = build_ridge_pipeline(config)
        x_train = prepare_feature_frame(training, config["categorical_features"], config["numeric_features"])
        pipeline.fit(x_train, training["sales"])
        unknown = training.iloc[[0]].copy()
        unknown["event_name_1"] = "UNSEEN_EVENT"
        x_unknown = prepare_feature_frame(unknown, config["categorical_features"], config["numeric_features"])
        prediction = pipeline.predict(x_unknown)
        self.assertEqual(len(prediction), 1)
        self.assertTrue(np.isfinite(prediction[0]))

    def test_negative_clipping_is_nonnegative_and_not_rounded(self) -> None:
        clipped = clip_nonnegative([-2.5, 0.0, 1.25])
        self.assertEqual(clipped.tolist(), [0.0, 0.0, 1.25])

    def test_recursive_lags_and_rolling_use_predictions(self) -> None:
        data = synthetic_data()
        model = RecordingModel()
        predictions, audit = generate_recursive_predictions(model, data, FOLD, MANIFEST, small_config())
        day_one = float(predictions.loc[predictions["horizon_day"].eq(1), "forecast"].iloc[0])
        self.assertAlmostEqual(float(model.frames[1]["sales_lag_1"].iloc[0]), day_one)
        self.assertAlmostEqual(float(model.frames[7]["sales_lag_7"].iloc[0]), day_one)
        expected_day_two_mean = np.mean([25, 26, 27, 28, 29, 30, day_one])
        self.assertAlmostEqual(float(model.frames[1]["sales_roll_mean_7"].iloc[0]), expected_day_two_mean, places=6)
        self.assertTrue(audit["checks"]["recursive_lag_1_uses_prediction"])
        self.assertTrue(audit["checks"]["recursive_lag_7_uses_prediction"])

    def test_validation_and_future_feature_mutations_do_not_change_forecasts(self) -> None:
        data = synthetic_data()
        original, _ = generate_recursive_predictions(RecordingModel(), data, FOLD, MANIFEST, small_config())
        changed = data.copy()
        mask = changed["date"].ge(FOLD["start_date"])
        changed.loc[mask, "sales"] = 9999
        changed.loc[mask, DEMAND_FEATURES] = -9999
        changed.loc[mask, "known_future_sell_price"] = 9999
        modified, _ = generate_recursive_predictions(RecordingModel(), changed, FOLD, MANIFEST, small_config())
        self.assertTrue(np.allclose(original["raw_forecast"], modified["raw_forecast"]))
        self.assertFalse(np.array_equal(original["actual"], modified["actual"]))

    def test_recursive_feature_definitions_match_stage6_semantics(self) -> None:
        values = np.arange(1, 29, dtype=float)
        features = recursive_demand_features(values)
        self.assertEqual(features["sales_lag_1"], 28)
        self.assertEqual(features["sales_lag_7"], 22)
        self.assertEqual(features["sales_lag_28"], 1)
        self.assertAlmostEqual(features["sales_roll_mean_7"], pd.Series(values[-7:]).mean())
        self.assertAlmostEqual(features["sales_roll_std_7"], pd.Series(values[-7:]).std())
        self.assertAlmostEqual(features["sales_roll_std_28"], pd.Series(values).std())

    def test_preprocessing_fit_is_independent_of_validation_mutation(self) -> None:
        config = small_config()
        data = synthetic_data()
        training = data.iloc[28:35].copy()
        x_train = prepare_feature_frame(training, config["categorical_features"], config["numeric_features"])
        first = build_ridge_pipeline(config).fit(x_train, training["sales"])
        validation = data.iloc[[40]].copy()
        x_validation = prepare_feature_frame(validation, config["categorical_features"], config["numeric_features"])
        before = first.predict(x_validation)
        validation["sales"] = 9999
        validation["event_name_1"] = "NEW_CATEGORY"
        after = first.predict(prepare_feature_frame(validation, config["categorical_features"], config["numeric_features"]))
        self.assertTrue(np.isfinite(after[0]))
        self.assertEqual(first.named_steps["preprocessing"].named_transformers_["numeric"].n_samples_seen_, len(training))
        self.assertNotEqual(float(before[0]), float("inf"))

    def test_final_test_is_blocked(self) -> None:
        final_fold = {"name": "final_test", "start_date": "2016-03-01", "end_date": "2016-03-30"}
        with self.assertRaisesRegex(ValueError, "final-test lock"):
            generate_recursive_predictions(RecordingModel(), synthetic_data(), final_fold, MANIFEST, small_config())


if __name__ == "__main__":
    unittest.main()
