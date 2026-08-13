"""Focused tests for reusable Stage 3 profiling helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from smartstock.data.profile_m5_data import (  # noqa: E402
    find_daily_sales_columns,
    profile_calendar,
    scan_daily_sales_values,
)


class ProfileM5DataTests(unittest.TestCase):
    def test_finds_and_numerically_sorts_daily_columns(self) -> None:
        columns = ["id", "d_10", "d_invalid", "item_id", "d_2", "d_1"]

        self.assertEqual(find_daily_sales_columns(columns), ["d_1", "d_2", "d_10"])

    def test_scans_sales_values_and_builds_row_metrics(self) -> None:
        contents = (
            "id,item_id,d_1,d_2,d_3\n"
            "one,FOODS_1_001,0,2,0\n"
            "two,FOODS_1_002,-1,0,4\n"
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            sales_path = Path(temporary_dir) / "sales.csv"
            sales_path.write_text(contents, encoding="utf-8")
            profile, metrics = scan_daily_sales_values(
                sales_path,
                ["d_1", "d_2", "d_3"],
                chunk_size=1,
            )

        self.assertEqual(profile["total_values"], 6)
        self.assertEqual(profile["zero_values"], 3)
        self.assertEqual(profile["negative_values"], 1)
        self.assertEqual(profile["minimum"], -1)
        self.assertEqual(profile["maximum"], 4)
        self.assertEqual(metrics["total_units"].tolist(), [2, 3])
        self.assertEqual(metrics["first_positive_day"].tolist(), [2, 3])
        self.assertEqual(metrics["last_positive_day"].tolist(), [2, 3])

    def test_profiles_calendar_events_and_snap_flags(self) -> None:
        calendar = pd.DataFrame(
            {
                "date": pd.to_datetime(["2011-01-29", "2011-01-30"]),
                "wm_yr_wk": pd.Series([11101, 11101], dtype="int32"),
                "weekday": ["Saturday", "Sunday"],
                "wday": [1, 2],
                "month": [1, 1],
                "year": [2011, 2011],
                "d": ["d_1", "d_2"],
                "event_name_1": [None, "Event A"],
                "event_type_1": [None, "Cultural"],
                "event_name_2": [None, "Event B"],
                "event_type_2": [None, "Religious"],
                "snap_CA": [0, 1],
                "snap_TX": [1, 0],
                "snap_WI": [0, 0],
            }
        )

        profile = profile_calendar(calendar)

        self.assertEqual(profile["rows"], 2)
        self.assertTrue(profile["d_is_unique"])
        self.assertEqual(profile["event_days"], 1)
        self.assertEqual(profile["two_event_days"], 1)
        self.assertEqual(profile["event_type_distribution"], {"Cultural": 1, "Religious": 1})
        self.assertEqual(profile["snap_coverage"]["California"]["enabled_days"], 1)


if __name__ == "__main__":
    unittest.main()
