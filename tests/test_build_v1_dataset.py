"""Tests for controlled wide-to-long and key-safe join helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from smartstock.data.build_v1_dataset import (  # noqa: E402
    join_calendar,
    join_prices,
    wide_to_long,
)


class BuildV1DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wide = pd.DataFrame(
            {
                "id": ["a", "b"],
                "item_id": ["FOODS_1_001", "FOODS_1_001"],
                "dept_id": ["FOODS_1", "FOODS_1"],
                "cat_id": ["FOODS", "FOODS"],
                "store_id": ["CA_1", "TX_2"],
                "state_id": ["CA", "TX"],
                "d_1": [0, 1],
                "d_2": [2, 0],
            }
        )
        self.calendar = pd.DataFrame(
            {
                "d": ["d_1", "d_2"],
                "date": pd.to_datetime(["2011-01-29", "2011-01-30"]),
                "wm_yr_wk": [11101, 11101],
                "weekday": ["Saturday", "Sunday"],
                "wday": [1, 2],
                "month": [1, 1],
                "year": [2011, 2011],
                "event_name_1": [None, None],
                "event_type_1": [None, None],
                "event_name_2": [None, None],
                "event_type_2": [None, None],
                "snap_CA": [0, 0],
                "snap_TX": [0, 0],
                "snap_WI": [0, 0],
            }
        )

    def test_wide_to_long_and_calendar_join_preserve_rows(self) -> None:
        long_sales = wide_to_long(self.wide, ["d_1", "d_2"])
        joined, stats = join_calendar(long_sales, self.calendar)

        self.assertEqual(len(long_sales), 4)
        self.assertEqual(len(joined), 4)
        self.assertEqual(stats["unmatched_rows"], 0)
        self.assertEqual(joined["sales"].sum(), 3)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(joined["date"]))

    def test_price_join_uses_composite_key_without_multiplying_rows(self) -> None:
        long_sales = wide_to_long(self.wide, ["d_1", "d_2"])
        daily, _ = join_calendar(long_sales, self.calendar)
        prices = pd.DataFrame(
            {
                "store_id": ["CA_1", "TX_2"],
                "item_id": ["FOODS_1_001", "FOODS_1_001"],
                "wm_yr_wk": [11101, 11101],
                "sell_price": [2.0, 3.0],
            }
        )

        joined, stats = join_prices(daily, prices)

        self.assertEqual(len(joined), 4)
        self.assertEqual(stats["row_multiplier"], 1)
        self.assertEqual(stats["missing_price_rows"], 0)
        self.assertEqual(set(joined.loc[joined["store_id"].eq("CA_1"), "sell_price"]), {2.0})

    def test_duplicate_price_composite_key_is_rejected(self) -> None:
        long_sales = wide_to_long(self.wide, ["d_1", "d_2"])
        daily, _ = join_calendar(long_sales, self.calendar)
        prices = pd.DataFrame(
            {
                "store_id": ["CA_1", "CA_1"],
                "item_id": ["FOODS_1_001", "FOODS_1_001"],
                "wm_yr_wk": [11101, 11101],
                "sell_price": [2.0, 2.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "not unique"):
            join_prices(daily, prices)


if __name__ == "__main__":
    unittest.main()
