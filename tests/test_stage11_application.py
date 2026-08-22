"""Focused Stage 11 database, fallback, app-contract, and scenario tests."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from sqlalchemy import create_engine, inspect


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from app.app_utils import (  # noqa: E402
    add_order_action,
    apply_inventory_scenario,
    dataframe_to_csv_bytes,
    filter_recommendations,
    forecast_chart_data,
    recommendation_table,
    scenario_explanations,
    shortage_reduction_pct,
)
from smartstock.database.connection import (  # noqa: E402
    DatabaseSettings,
    normalize_database_url,
)
from smartstock.database.data_source import (  # noqa: E402
    CsvDataSource,
    DataSourceResolution,
    resolve_data_source,
)
from smartstock.database.loaders import (  # noqa: E402
    ArtifactPaths,
    RECOMMENDATION_COLUMNS,
    load_database_artifacts,
    validate_artifact_contracts,
)
from smartstock.database.queries import (  # noqa: E402
    get_forecast,
    get_inventory_recommendations,
    get_overview_metrics,
)
from smartstock.database.verify_database import verify_database  # noqa: E402
from smartstock.inventory.policy import load_policy  # noqa: E402


POLICY = load_policy(PROJECT_ROOT / "config" / "v1_inventory_policy.json")


def small_csv_source() -> CsvDataSource:
    source = CsvDataSource.__new__(CsvDataSource)
    source.paths = None
    dates = pd.date_range("2016-05-20", periods=3)
    source._history_cache = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "store_id": ["CA_1"] * 3 + ["TX_2"] * 3,
            "item_id": ["FOODS_1_001"] * 6,
            "dept_id": ["FOODS_1"] * 6,
            "sales": [0, 2, 1, 3, 0, 4],
            "sell_price": [1.0] * 6,
            "weekday": [date.day_name() for date in list(dates) * 2],
        }
    )
    source._forecast_cache = pd.DataFrame(
        {
            "item_id": ["FOODS_1_001"] * 4,
            "store_id": ["CA_1"] * 2 + ["TX_2"] * 2,
            "dept_id": ["FOODS_1"] * 4,
            "demand_band": ["low"] * 4,
            "forecast_origin": pd.to_datetime(["2016-05-22"] * 4),
            "target_date": pd.to_datetime(["2016-05-23", "2016-05-24"] * 2),
            "horizon_day": [1, 2, 1, 2],
            "forecast": [1.0, 1.5, 2.0, 2.5],
            "model_name": ["mean_28"] * 4,
            "model_version": ["v1"] * 4,
        }
    )
    recommendation = {column: 0 for column in RECOMMENDATION_COLUMNS}
    recommendation.update(
        {
            "store_id": "CA_1", "item_id": "FOODS_1_001", "dept_id": "FOODS_1",
            "demand_band": "low", "inventory_source": "synthetic_demo",
            "stock_status": "REORDER NOW", "priority_label": "HIGH", "priority_score": 70.0,
            "recommended_order_qty": 5, "expected_shortage_without_order": 4.0,
            "expected_shortage_with_recommendation": 1.0, "estimated_cost_savings": 10.0,
        }
    )
    source._recommendation_cache = pd.DataFrame([recommendation])
    source._snapshot_cache = pd.DataFrame(
        [{
            "store_id": "CA_1", "item_id": "FOODS_1_001", "dept_id": "FOODS_1",
            "demand_band": "low", "on_hand": 2.0, "on_order": 0.0, "backorders": 0.0,
            "lead_time_days": 7, "service_level": 0.95,
            "holding_cost_per_unit_per_day": 0.02, "stockout_cost_per_unit": 5.0,
            "fixed_order_cost": 25.0, "inventory_source": "synthetic_demo",
            "synthetic_profile_plan": "reorder",
        }]
    )
    return source


class Stage11ApplicationTests(unittest.TestCase):
    def test_database_url_parsing_and_modes(self) -> None:
        url = normalize_database_url("postgresql://user:secret@localhost:5432/smartstock")
        self.assertEqual(url.drivername, "postgresql+psycopg")
        self.assertEqual(url.database, "smartstock")
        self.assertNotIn("secret", url.render_as_string(hide_password=True))
        with self.assertRaises(ValueError):
            normalize_database_url("sqlite:///not-production.db")
        with patch.dict(os.environ, {"SMARTSTOCK_DATA_MODE": "invalid"}, clear=False):
            with self.assertRaises(ValueError):
                DatabaseSettings.from_environment(load_dotenv=False)

    def test_csv_fallback_is_visible_and_explicit(self) -> None:
        sentinel = small_csv_source()
        with patch("smartstock.database.data_source.CsvDataSource", return_value=sentinel):
            resolution = resolve_data_source(DatabaseSettings(None, "auto"))
        self.assertIs(resolution.source, sentinel)
        self.assertTrue(resolution.fallback_used)
        self.assertIn("DATABASE_URL", resolution.notice)

    def test_csv_filters_and_forecast_horizon(self) -> None:
        source = small_csv_source()
        forecast = source.get_forecast("CA_1", "FOODS_1_001", horizon_days=1)
        self.assertEqual(len(forecast), 1)
        self.assertEqual(forecast.iloc[0]["forecast"], 1.0)
        profile = source.get_demand_profile(filters={"store_id": "CA_1"})
        self.assertEqual(profile["zero_observations"], 1)
        self.assertAlmostEqual(profile["zero_rate"], 1 / 3)
        with self.assertRaises(ValueError):
            source.get_forecast(horizon_days=31)

    def test_scenario_reuses_engine_and_never_mutates_saved_inputs(self) -> None:
        snapshot = {
            "on_hand": 5.0, "on_order": 1.0, "backorders": 0.0, "lead_time_days": 7,
            "service_level": 0.95, "holding_cost_per_unit_per_day": 0.02,
            "stockout_cost_per_unit": 5.0, "fixed_order_cost": 25.0,
        }
        original_snapshot = deepcopy(snapshot)
        original_policy = deepcopy(POLICY)
        decision = apply_inventory_scenario(
            [2.0] * 30,
            snapshot,
            {"sigma_daily": 1.5, "residual_mean_used": 0.0},
            POLICY,
            {"on_hand": 0.0, "lead_time_days": 10, "review_period_days": 5},
        )
        self.assertGreater(decision["recommended_order_qty"], 0)
        self.assertEqual(snapshot, original_snapshot)
        self.assertEqual(POLICY, original_policy)

    def test_forecast_chart_does_not_invent_future_actuals(self) -> None:
        history = pd.DataFrame({"date": pd.to_datetime(["2016-05-21", "2016-05-22"]), "sales": [2, 3]})
        future = pd.DataFrame({"target_date": pd.to_datetime(["2016-05-23"]), "forecast": [2.5]})
        chart = forecast_chart_data(history, future)
        future_row = chart.loc[chart["date"].eq(pd.Timestamp("2016-05-23"))].iloc[0]
        self.assertTrue(pd.isna(future_row["Actual demand"]))
        self.assertEqual(future_row["Forecast demand"], 2.5)

    def test_recommendation_presentation_filters_and_labels_actions(self) -> None:
        rows = pd.concat(
            [
                small_csv_source()._recommendation_cache,
                small_csv_source()._recommendation_cache.assign(
                    item_id="FOODS_1_002",
                    store_id="TX_2",
                    stock_status="HEALTHY",
                    priority_label="NONE",
                    priority_score=0.0,
                    recommended_order_qty=0,
                    expected_shortage_without_order=0.0,
                    expected_shortage_with_recommendation=0.0,
                ),
            ],
            ignore_index=True,
        )
        labeled = add_order_action(rows)
        self.assertEqual(labeled["order_action"].tolist(), ["REORDER", "NO ORDER"])
        reorder = filter_recommendations(rows, action="Needs reorder")
        self.assertEqual(reorder["item_id"].tolist(), ["FOODS_1_001"])
        no_order = filter_recommendations(rows, action="No order", store_id="TX_2")
        self.assertEqual(no_order["item_id"].tolist(), ["FOODS_1_002"])
        table = recommendation_table(reorder)
        self.assertEqual(table.loc[0, "Action"], "REORDER")
        self.assertIn("Recommended units", table.columns)
        self.assertNotIn("Unnamed: 0", dataframe_to_csv_bytes(table).decode("utf-8"))

    def test_shortage_reduction_and_scenario_explanations_are_clear(self) -> None:
        frame = pd.DataFrame(
            {
                "expected_shortage_without_order": [10.0, 0.0],
                "expected_shortage_with_recommendation": [2.0, 0.0],
            }
        )
        self.assertAlmostEqual(shortage_reduction_pct(frame), 0.8)
        explanations = scenario_explanations(
            {"inventory_position": 5.0, "recommended_order_qty": 8},
            {"inventory_position": 2.0, "recommended_order_qty": 12},
            {"lead_time_days": 7, "service_level": 0.95, "review_period_days": 7},
            {"lead_time_days": 10, "service_level": 0.99, "review_period_days": 7},
        )
        text = " ".join(explanations)
        self.assertIn("Inventory position decreased", text)
        self.assertIn("10-day lead time", text)
        self.assertIn("99.0%", text)
        self.assertIn("12 units", text)

    def test_sqlite_schema_queries_and_parameter_safety(self) -> None:
        engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        paths, temp_dir = self._small_artifacts()
        self.addCleanup(temp_dir.cleanup)
        result = load_database_artifacts(engine, paths, strict_expected=False)
        self.assertEqual(result["row_counts"]["demand_history"], 2)
        self.assertEqual(set(inspect(engine).get_table_names()), {
            "demand_history", "forecasts", "inventory_snapshot",
            "inventory_recommendations", "model_metadata",
        })
        overview = get_overview_metrics(engine)
        self.assertEqual(overview["series"], 1)
        self.assertEqual(overview["sales_observations"], 2)
        self.assertEqual(len(get_forecast(engine, "CA_1", "FOODS_1_001", horizon_days=1)), 1)
        malicious = "CA_1'; DROP TABLE forecasts; --"
        self.assertTrue(get_inventory_recommendations(engine, {"store_id": malicious}).empty)
        self.assertIn("forecasts", inspect(engine).get_table_names())
        second = load_database_artifacts(engine, paths, strict_expected=False)
        self.assertEqual(second["load_action"], "skipped_existing_complete_database")
        engine.dispose()

    def test_duplicate_recommendation_contract_is_rejected(self) -> None:
        paths, temp_dir = self._small_artifacts(duplicate_recommendation=True)
        self.addCleanup(temp_dir.cleanup)
        with self.assertRaisesRegex(ValueError, "duplicate item-store"):
            validate_artifact_contracts(paths, strict_expected=False)

    def test_database_verification_helper_accepts_valid_loaded_fixture(self) -> None:
        engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        paths, temp_dir = self._small_artifacts()
        self.addCleanup(temp_dir.cleanup)
        loaded = load_database_artifacts(engine, paths, strict_expected=False)
        result = verify_database(
            engine,
            expected_counts=loaded["row_counts"],
            strict_v1_cardinality=False,
        )
        self.assertEqual(result["connection"], "ok")
        self.assertEqual(result["forecast_days"], 1)
        self.assertEqual(result["synthetic_inventory_rows"], 1)
        engine.dispose()

    def test_streamlit_module_imports_without_rendering(self) -> None:
        import app.streamlit_app as application

        self.assertTrue(callable(application.main))
        self.assertIn("synthetic", application.INVENTORY_DISCLAIMER.lower())

    def _small_artifacts(self, *, duplicate_recommendation: bool = False) -> tuple[ArtifactPaths, tempfile.TemporaryDirectory[str]]:
        temp_dir = tempfile.TemporaryDirectory()
        root = Path(temp_dir.name)
        history = pd.DataFrame(
            {
                "date": ["2016-05-21", "2016-05-22"], "store_id": ["CA_1"] * 2,
                "item_id": ["FOODS_1_001"] * 2, "d": ["d_1940", "d_1941"],
                "dept_id": ["FOODS_1"] * 2, "cat_id": ["FOODS"] * 2, "state_id": ["CA"] * 2,
                "sales": [1, 2], "sell_price": [1.0, 1.0], "weekday": ["Saturday", "Sunday"],
                "wday": [1, 2], "month": [5, 5], "year": [2016, 2016],
                "event_name_1": [None, None], "event_type_1": [None, None],
                "event_name_2": [None, None], "event_type_2": [None, None],
                "snap_CA": [0, 1], "snap_TX": [0, 0], "snap_WI": [0, 0], "demand_band": ["low"] * 2,
            }
        )
        forecast = pd.DataFrame(
            [{
                "item_id": "FOODS_1_001", "store_id": "CA_1", "dept_id": "FOODS_1",
                "demand_band": "low", "forecast_origin": "2016-05-22", "target_date": "2016-05-23",
                "horizon_day": 1, "forecast": 1.5,
            }]
        )
        snapshot = small_csv_source()._snapshot_cache
        recommendation = small_csv_source()._recommendation_cache
        if duplicate_recommendation:
            recommendation = pd.concat([recommendation, recommendation], ignore_index=True)
        history_path, forecast_path = root / "history.csv", root / "forecast.csv"
        snapshot_path, recommendation_path = root / "snapshot.csv", root / "recommendation.csv"
        history.to_csv(history_path, index=False)
        forecast.to_csv(forecast_path, index=False)
        snapshot.to_csv(snapshot_path, index=False)
        recommendation.to_csv(recommendation_path, index=False)
        final_config, policy_path, deployment = root / "model.json", root / "policy.json", root / "model.joblib"
        final_config.write_text(json.dumps({"selected_model_name": "mean_28"}), encoding="utf-8")
        policy_path.write_text(json.dumps({"policy_name": "test_policy"}), encoding="utf-8")
        deployment.write_bytes(b"test artifact")
        return ArtifactPaths(history_path, forecast_path, snapshot_path, recommendation_path, final_config, policy_path, deployment), temp_dir


if __name__ == "__main__":
    unittest.main()
