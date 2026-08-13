"""Tests for Version 1 eligibility, demand bands, and deterministic selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from smartstock.data.select_v1_subset import (  # noqa: E402
    assign_demand_bands,
    evaluate_eligibility,
    select_stratified_items,
    validate_quota_availability,
)


class SelectV1SubsetTests(unittest.TestCase):
    def test_assigns_demand_bands_from_tertiles(self) -> None:
        items = pd.DataFrame(
            {
                "item_id": [f"item_{number}" for number in range(1, 10)],
                "mean_daily_demand": [float(number) for number in range(1, 10)],
            }
        )

        assigned, thresholds = assign_demand_bands(items)

        self.assertAlmostEqual(thresholds["low_upper_inclusive"], 3.6666666667)
        self.assertAlmostEqual(thresholds["medium_upper_inclusive"], 6.3333333333)
        self.assertEqual(assigned["demand_band"].value_counts().to_dict(), {"low": 3, "medium": 3, "high": 3})

    def test_eligibility_rules_are_combined_without_hiding_overlap(self) -> None:
        metrics = pd.DataFrame(
            {
                "item_id": ["good", "sparse", "late"],
                "structurally_in_required_stores": [True, True, True],
                "store_count": [3, 3, 3],
                "zero_prevalence": [0.5, 0.95, 0.4],
                "remaining_history_days": [900, 900, 729],
            }
        )
        price_coverage = pd.DataFrame(
            {
                "item_id": ["good"] * 3 + ["sparse"] * 3 + ["late"] * 3,
                "store_id": ["CA_1", "TX_2", "WI_3"] * 3,
                "active_period_price_coverage": [0.8, 0.75, 0.9, 0.9, 0.9, 0.9, 0.69, 0.8, 0.8],
            }
        )

        result = evaluate_eligibility(metrics, price_coverage)
        by_item = result.set_index("item_id")

        self.assertTrue(bool(by_item.loc["good", "eligible"]))
        self.assertFalse(bool(by_item.loc["sparse", "passes_sparsity"]))
        self.assertFalse(bool(by_item.loc["late", "passes_history"]))
        self.assertFalse(bool(by_item.loc["late", "passes_price_coverage"]))

    def test_selection_is_deterministic_and_enforces_quotas(self) -> None:
        records = []
        quotas = {
            "FOODS_1": {"low": 2, "medium": 2, "high": 2},
            "FOODS_2": {"low": 1, "medium": 1, "high": 1},
        }
        for department in quotas:
            for band in ("low", "medium", "high"):
                for number in range(5):
                    records.append(
                        {
                            "item_id": f"{department}_{band}_{number}",
                            "department": department,
                            "demand_band": band,
                        }
                    )
        eligible = pd.DataFrame.from_records(records)

        first = select_stratified_items(eligible, quotas, random_seed=42)
        second = select_stratified_items(eligible.sample(frac=1, random_state=7), quotas, random_seed=42)

        self.assertEqual(first["item_id"].tolist(), second["item_id"].tolist())
        selected_counts = first.groupby(["department", "demand_band"]).size().to_dict()
        expected_counts = {
            (department, band): required
            for department, bands in quotas.items()
            for band, required in bands.items()
        }
        self.assertEqual(selected_counts, expected_counts)

    def test_quota_shortage_raises_instead_of_silently_changing_target(self) -> None:
        eligible = pd.DataFrame(
            {"item_id": ["one"], "department": ["FOODS_1"], "demand_band": ["low"]}
        )
        with self.assertRaisesRegex(ValueError, "cannot be satisfied"):
            validate_quota_availability(eligible, {"FOODS_1": {"low": 2}})


if __name__ == "__main__":
    unittest.main()
