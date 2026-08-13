"""Focused Stage 9 tests for nonlinear preprocessing, recursion, freezing, and I/O."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from smartstock.models.production_forecaster import (  # noqa: E402
    FORECAST_COLUMNS,
    forecast_demand,
    load_forecaster,
    save_forecaster,
)
from smartstock.models.ridge_model import PROHIBITED_FEATURES, prepare_feature_frame  # noqa: E402
from smartstock.models.stage9_evaluation import (  # noqa: E402
    assert_locked_test_can_run,
    freeze_final_model_config,
)
from smartstock.models.stage9_models import (  # noqa: E402
    build_tree_pipeline,
    select_training_rows_through,
    validate_candidate_config,
)


class RecordingModel:
    def __init__(self, offset: float = 0.5) -> None:
        self.offset = offset
        self.frames: list[pd.DataFrame] = []

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self.frames.append(frame.copy())
        return frame["sales_lag_1"].to_numpy(dtype="float64") + self.offset


def candidate_config() -> dict:
    config = json.loads(
        (PROJECT_ROOT / "config" / "v1_stage9_candidates.json").read_text(encoding="utf-8")
    )
    config["expected_series"] = 1
    return config


def synthetic_data() -> pd.DataFrame:
    dates = pd.date_range("2016-01-01", periods=70, freq="D")
    sales = np.asarray([(index % 7) + 1 for index in range(70)], dtype="float32")
    frame = pd.DataFrame(
        {
            "date": dates,
            "availability_date": pd.Timestamp("2016-01-01"),
            "is_feature_ready": True,
            "sales": sales,
            "demand_band": "medium",
            "item_id": "FOODS_1_001",
            "store_id": "CA_1",
            "dept_id": "FOODS_1",
            "event_name_1": pd.Series([None] * 70, dtype="string"),
            "event_type_1": pd.Series([None] * 70, dtype="string"),
            "event_name_2": pd.Series([None] * 70, dtype="string"),
            "event_type_2": pd.Series([None] * 70, dtype="string"),
            "year": dates.year,
            "day_of_month": dates.day,
            "is_weekend": dates.dayofweek >= 5,
            "day_of_week_sin": np.sin(2 * np.pi * dates.dayofweek / 7),
            "day_of_week_cos": np.cos(2 * np.pi * dates.dayofweek / 7),
            "month_sin": np.sin(2 * np.pi * (dates.month - 1) / 12),
            "month_cos": np.cos(2 * np.pi * (dates.month - 1) / 12),
            "is_event": False,
            "snap_active": 0,
            "product_age_days": np.arange(70),
        }
    )
    history = pd.Series(sales)
    for lag in (1, 7, 14, 28):
        frame[f"sales_lag_{lag}"] = history.shift(lag)
    shifted = history.shift(1)
    for window in (7, 14, 28):
        frame[f"sales_roll_mean_{window}"] = shifted.rolling(window).mean()
    for window in (7, 28):
        frame[f"sales_roll_std_{window}"] = shifted.rolling(window).std()
    frame["sell_price"] = 3.99
    frame["known_future_sell_price"] = 4.25
    return frame


class Stage9ModelTests(unittest.TestCase):
    def test_candidate_config_excludes_price_and_analysis_features(self) -> None:
        config = candidate_config()
        config["expected_series"] = 300
        manifest = json.loads(
            (PROJECT_ROOT / "config" / "v1_features.json").read_text(encoding="utf-8")
        )
        checks = validate_candidate_config(config, manifest)
        features = set(config["categorical_features"] + config["numeric_features"])
        self.assertTrue(all(checks.values()))
        self.assertFalse(features & PROHIBITED_FEATURES)
        self.assertNotIn("demand_band", features)
        self.assertFalse(any("price" in name for name in features))

    def test_training_cutoff_includes_origin_but_never_future_rows(self) -> None:
        training = select_training_rows_through(synthetic_data(), pd.Timestamp("2016-02-10"))
        self.assertEqual(training["date"].max(), pd.Timestamp("2016-02-10"))
        self.assertFalse(training["date"].gt("2016-02-10").any())

    def test_hist_gradient_boosting_handles_unseen_category_without_scaling(self) -> None:
        config = candidate_config()
        training = synthetic_data().iloc[28:55].copy()
        parameters = {
            "learning_rate": 0.1,
            "max_iter": 5,
            "max_leaf_nodes": 7,
            "l2_regularization": 1.0,
            "random_state": 42,
        }
        pipeline = build_tree_pipeline(config, "hist_gradient_boosting", parameters)
        x_train = prepare_feature_frame(training, config["categorical_features"], config["numeric_features"])
        pipeline.fit(x_train, training["sales"])
        unseen = training.iloc[[0]].copy()
        unseen["event_name_1"] = "UNSEEN_EVENT"
        prediction = pipeline.predict(
            prepare_feature_frame(unseen, config["categorical_features"], config["numeric_features"])
        )
        self.assertTrue(np.isfinite(prediction[0]))
        self.assertNotIn("StandardScaler", repr(pipeline))

    def test_xgboost_handles_unseen_category(self) -> None:
        config = candidate_config()
        training = synthetic_data().iloc[28:55].copy()
        parameters = {
            "n_estimators": 5,
            "learning_rate": 0.1,
            "max_depth": 2,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": 1,
        }
        pipeline = build_tree_pipeline(config, "xgboost", parameters)
        x_train = prepare_feature_frame(training, config["categorical_features"], config["numeric_features"])
        pipeline.fit(x_train, training["sales"])
        unseen = training.iloc[[0]].copy()
        unseen["event_type_1"] = "UNSEEN_TYPE"
        prediction = pipeline.predict(
            prepare_feature_frame(unseen, config["categorical_features"], config["numeric_features"])
        )
        self.assertTrue(np.isfinite(prediction[0]))

    def test_hist_gradient_boosting_is_deterministic(self) -> None:
        config = candidate_config()
        training = synthetic_data().iloc[28:60].copy()
        x_train = prepare_feature_frame(training, config["categorical_features"], config["numeric_features"])
        parameters = {"max_iter": 5, "max_leaf_nodes": 7, "random_state": 42}
        first = build_tree_pipeline(config, "hist_gradient_boosting", parameters).fit(x_train, training["sales"])
        second = build_tree_pipeline(config, "hist_gradient_boosting", parameters).fit(x_train, training["sales"])
        self.assertTrue(np.allclose(first.predict(x_train), second.predict(x_train)))

    def test_production_forecast_schema_horizons_and_recursive_updates(self) -> None:
        model = RecordingModel()
        bundle = {
            "pipeline": model,
            "config": candidate_config(),
            "model_family": "test",
            "trained_through": "2016-02-09",
        }
        forecasts, audit = forecast_demand(
            bundle, synthetic_data(), "2016-02-09", 7, return_audit=True
        )
        self.assertEqual(len(forecasts), 7)
        self.assertTrue(set(FORECAST_COLUMNS).issubset(forecasts.columns))
        self.assertEqual(forecasts["horizon_day"].tolist(), list(range(1, 8)))
        self.assertAlmostEqual(float(model.frames[1]["sales_lag_1"].iloc[0]), float(forecasts.iloc[0]["forecast"]))
        self.assertTrue(audit["recursive_lag_1_uses_prediction"])
        self.assertTrue(audit["recursive_rolling_uses_synthetic_history"])
        for horizon in (1, 7, 30):
            result = forecast_demand(bundle, synthetic_data(), "2016-02-09", horizon)
            self.assertEqual(len(result), horizon)

    def test_future_actual_and_price_mutation_cannot_change_production_forecast(self) -> None:
        bundle = {
            "pipeline": RecordingModel(),
            "config": candidate_config(),
            "model_family": "test",
            "trained_through": "2016-02-09",
        }
        data = synthetic_data()
        original = forecast_demand(bundle, data, "2016-02-09", 30)
        changed = data.copy()
        mask = changed["date"].gt("2016-02-09")
        changed.loc[mask, "sales"] = 99999
        changed.loc[mask, "sell_price"] = 99999
        changed.loc[mask, "known_future_sell_price"] = 99999
        changed.loc[mask, [name for name in changed.columns if name.startswith("sales_lag_") or name.startswith("sales_roll_")]] = -99999
        modified = forecast_demand(
            {**bundle, "pipeline": RecordingModel()}, changed, "2016-02-09", 30
        )
        self.assertTrue(np.allclose(original["forecast"], modified["forecast"]))

    def test_negative_production_predictions_are_clipped_without_rounding(self) -> None:
        bundle = {
            "pipeline": RecordingModel(offset=-100.25),
            "config": candidate_config(),
            "model_family": "test",
            "trained_through": "2016-02-09",
        }
        result = forecast_demand(bundle, synthetic_data(), "2016-02-09", 1)
        self.assertEqual(float(result["forecast"].iloc[0]), 0.0)
        self.assertLess(float(result["raw_forecast"].iloc[0]), 0.0)

    def test_serialization_and_loading_preserve_forecast_contract(self) -> None:
        bundle = {
            "pipeline": RecordingModel(),
            "config": candidate_config(),
            "model_family": "test",
            "trained_through": "2016-02-09",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            save_forecaster(bundle, path)
            loaded = load_forecaster(path)
            result = forecast_demand(loaded, synthetic_data(), "2016-02-09", 1)
        self.assertEqual(len(result), 1)

    def test_locked_test_requires_freeze_and_blocks_second_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "final.json"
            validation_path = root / "validation.json"
            validation = {
                "locked_final_test": {
                    "start_date": "2016-04-23",
                    "end_date": "2016-05-22",
                    "days": 30,
                }
            }
            validation_path.write_text(json.dumps(validation), encoding="utf-8")
            with self.assertRaises(PermissionError):
                assert_locked_test_can_run(config_path, validation_path, root / "receipt.json", root / "pred.csv")

            config = candidate_config()
            config["expected_series"] = 300
            ranking = pd.DataFrame([{"model_name": "hist_gradient_boosting_base", "selection_rank": 1}])
            freeze_final_model_config(
                config_path,
                config,
                validation,
                selected_model_name="hist_gradient_boosting_base",
                selected_model_family="hist_gradient_boosting",
                selected_parameters=config["candidate_models"]["hist_gradient_boosting"]["base_configuration"],
                ranking=ranking,
                source_hashes={"fixture": "abc"},
            )
            frozen, _, _ = assert_locked_test_can_run(
                config_path, validation_path, root / "receipt.json", root / "pred.csv"
            )
            self.assertEqual(frozen["selection_status"], "frozen_before_test")
            with self.assertRaises(FileExistsError):
                freeze_final_model_config(
                    config_path,
                    config,
                    validation,
                    selected_model_name="hist_gradient_boosting_base",
                    selected_model_family="hist_gradient_boosting",
                    selected_parameters={},
                    ranking=ranking,
                    source_hashes={},
                )
            (root / "receipt.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(PermissionError):
                assert_locked_test_can_run(
                    config_path, validation_path, root / "receipt.json", root / "pred.csv"
                )

    def test_only_supported_production_horizons_are_allowed(self) -> None:
        bundle = {
            "pipeline": RecordingModel(),
            "config": candidate_config(),
            "model_family": "test",
            "trained_through": "2016-02-09",
        }
        with self.assertRaises(ValueError):
            forecast_demand(bundle, synthetic_data(), "2016-02-09", 2)


if __name__ == "__main__":
    unittest.main()
