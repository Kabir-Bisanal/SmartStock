"""Regression tests for the tracked SmartStock public deployment bundle."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smartstock.data.public_demo_bundle import (  # noqa: E402
    EXPECTED_STORES,
    PUBLIC_HISTORY_COLUMNS,
    PublicDemoPaths,
    load_public_demo_metadata,
    validate_public_demo_bundle,
)
from smartstock.database.connection import DatabaseSettings  # noqa: E402
from smartstock.database.data_source import (  # noqa: E402
    CsvDataSource,
    PublicDemoCsvDataSource,
    resolve_data_source,
)
from smartstock.database.loaders import (  # noqa: E402
    ArtifactPaths,
    FORECAST_COLUMNS,
    RECOMMENDATION_COLUMNS,
    SNAPSHOT_COLUMNS,
    file_sha256,
)
from smartstock.inventory.policy import load_policy  # noqa: E402

from app.app_utils import apply_inventory_scenario  # noqa: E402


PUBLIC_PATHS = PublicDemoPaths.from_directory(PROJECT_ROOT / "data/public_demo")


class PublicDemoBundleTests(unittest.TestCase):
    def test_committed_bundle_exists_and_matches_frozen_contract(self) -> None:
        for path in PUBLIC_PATHS.required_files():
            self.assertTrue(path.is_file(), path)
        result = validate_public_demo_bundle(PUBLIC_PATHS)
        self.assertEqual(
            result["row_counts"],
            {
                "history": 36_000,
                "forecasts": 9_000,
                "inventory_snapshot": 300,
                "recommendations": 300,
            },
        )
        self.assertEqual(result["history_start"], "2016-01-24")
        self.assertEqual(result["history_end"], "2016-05-22")
        self.assertEqual(result["history_days"], 120)
        self.assertEqual(result["stores"], EXPECTED_STORES)
        self.assertEqual(result["items"], 100)
        self.assertEqual(result["series"], 300)
        self.assertEqual(result["forecast_start"], "2016-05-23")
        self.assertEqual(result["forecast_end"], "2016-06-21")
        self.assertEqual(result["forecast_days"], 30)

    def test_public_metadata_preserves_immutable_portfolio_facts(self) -> None:
        metadata = load_public_demo_metadata(PUBLIC_PATHS)
        facts = metadata["portfolio_facts"]
        self.assertEqual(facts["source_observations"], 582_300)
        self.assertEqual(facts["item_store_series"], 300)
        self.assertEqual(facts["inventory_decisions"], 300)
        self.assertEqual(facts["reorder_decisions"], 198)
        self.assertEqual(facts["recommended_units"], 4_143)
        self.assertAlmostEqual(facts["expected_shortage_before"], 3312.6121677281303)
        self.assertAlmostEqual(facts["expected_shortage_after"], 167.0241028964644)
        self.assertFalse(metadata["model"]["runtime_model_required"])
        self.assertFalse((PUBLIC_PATHS.metadata.parent / "smartstock_v1_deployment_forecaster.joblib").exists())

    def test_public_history_is_ordered_and_has_complete_series_coverage(self) -> None:
        history = pd.read_csv(PUBLIC_PATHS.history, parse_dates=["date"])
        self.assertEqual(history.columns.tolist(), PUBLIC_HISTORY_COLUMNS)
        self.assertEqual(history["date"].nunique(), 120)
        counts = history.groupby(["store_id", "item_id"], observed=True).size()
        self.assertEqual(len(counts), 300)
        self.assertTrue(counts.eq(120).all())
        self.assertFalse(history.duplicated(["date", "store_id", "item_id"]).any())
        expected = history.sort_values(["date", "store_id", "item_id"], kind="stable")
        pd.testing.assert_frame_equal(history, expected.reset_index(drop=True))

    def test_saved_public_outputs_equal_authoritative_local_outputs_when_available(self) -> None:
        full = ArtifactPaths()
        if not all(
            path.is_file()
            for path in (full.forecasts, full.inventory_snapshot, full.recommendations)
        ):
            self.skipTest("Authoritative ignored local outputs are unavailable in this checkout.")

        comparisons = (
            (
                full.forecasts,
                PUBLIC_PATHS.forecasts,
                FORECAST_COLUMNS,
                ["target_date", "store_id", "item_id"],
            ),
            (
                full.inventory_snapshot,
                PUBLIC_PATHS.inventory_snapshot,
                SNAPSHOT_COLUMNS,
                ["store_id", "item_id"],
            ),
            (
                full.recommendations,
                PUBLIC_PATHS.recommendations,
                RECOMMENDATION_COLUMNS,
                ["store_id", "item_id"],
            ),
        )
        for authoritative_path, public_path, columns, keys in comparisons:
            authoritative = pd.read_csv(authoritative_path, usecols=columns).sort_values(
                keys, kind="stable"
            ).reset_index(drop=True)
            public = pd.read_csv(public_path, usecols=columns).sort_values(
                keys, kind="stable"
            ).reset_index(drop=True)
            pd.testing.assert_frame_equal(authoritative, public, check_exact=True)

    def test_csv_mode_uses_public_bundle_when_full_artifacts_are_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary)
            full_paths = ArtifactPaths(
                history=missing / "history.csv",
                forecasts=missing / "forecasts.csv",
                inventory_snapshot=missing / "snapshot.csv",
                recommendations=missing / "recommendations.csv",
                final_model_config=missing / "model.json",
                inventory_policy=missing / "policy.json",
                deployment_model=missing / "model.joblib",
            )
            resolution = resolve_data_source(
                DatabaseSettings(None, "csv"), full_paths, PUBLIC_PATHS
            )
        self.assertIsInstance(resolution.source, PublicDemoCsvDataSource)
        self.assertIn("compact public bundle", resolution.notice)
        metrics = resolution.source.get_overview_metrics()
        self.assertEqual(metrics["sales_observations"], 582_300)
        self.assertEqual(metrics["bundled_history_observations"], 36_000)

    def test_full_csv_branch_remains_preferred_when_local_artifacts_exist(self) -> None:
        sentinel = object()
        with (
            patch(
                "smartstock.database.data_source._full_csv_artifacts_available",
                return_value=True,
            ),
            patch("smartstock.database.data_source.CsvDataSource", return_value=sentinel),
        ):
            resolution = resolve_data_source(DatabaseSettings(None, "csv"))
        self.assertIs(resolution.source, sentinel)
        self.assertIn("complete local Stage 10 artifacts", resolution.notice)

    def test_postgres_mode_contract_is_unchanged(self) -> None:
        with patch.dict(os.environ, {"DATABASE_URL": ""}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "requires DATABASE_URL"):
                resolve_data_source(DatabaseSettings(None, "postgres"))

    def test_scenario_recalculation_does_not_mutate_public_artifacts(self) -> None:
        source = PublicDemoCsvDataSource(PUBLIC_PATHS)
        saved = source.get_recommendations().iloc[0]
        store_id, item_id = str(saved["store_id"]), str(saved["item_id"])
        forecast = source.get_forecast(store_id, item_id, horizon_days=30)["forecast"]
        snapshot = source.get_snapshot(store_id, item_id).iloc[0].to_dict()
        calibration = {
            "sigma_daily": float(saved["sigma_daily"]),
            "residual_mean_used": float(saved["residual_mean_used"]),
        }
        policy = load_policy(PROJECT_ROOT / "config" / "v1_inventory_policy.json")
        checksums_before = {
            path.name: file_sha256(path) for path in PUBLIC_PATHS.required_files()
        }

        revised = apply_inventory_scenario(
            forecast,
            snapshot,
            calibration,
            policy,
            {"on_hand": float(snapshot["on_hand"]) + 1.0},
        )

        self.assertIn("recommended_order_qty", revised)
        self.assertEqual(
            checksums_before,
            {path.name: file_sha256(path) for path in PUBLIC_PATHS.required_files()},
        )

    def test_public_bundle_contains_no_machine_path_or_secret_markers(self) -> None:
        secret_markers = ("c:\\users\\", "database_url=", "api_key=", "password=")
        for path in PUBLIC_PATHS.required_files():
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            for marker in secret_markers:
                self.assertNotIn(marker, text, f"{marker!r} found in {path.name}")
        json.loads(PUBLIC_PATHS.metadata.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
