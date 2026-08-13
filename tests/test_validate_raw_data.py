"""Tests for the lightweight raw M5 data validator."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from smartstock.data.validate_raw_data import (  # noqa: E402
    EXPECTED_M5_FILES,
    main,
    validate_raw_data,
)


VALID_FIXTURES = {
    "sales_train_evaluation.csv": (
        "id,item_id,dept_id,cat_id,store_id,state_id,d_1\n"
        "FOODS_1_001_CA_1_evaluation,FOODS_1_001,FOODS_1,FOODS,CA_1,CA,0\n"
    ),
    "sell_prices.csv": (
        "store_id,item_id,wm_yr_wk,sell_price\n"
        "CA_1,FOODS_1_001,11101,2.00\n"
    ),
    "calendar.csv": (
        "date,wm_yr_wk,weekday,wday,month,year,d\n"
        "2011-01-29,11101,Saturday,1,1,2011,d_1\n"
    ),
}


class ValidateRawDataTests(unittest.TestCase):
    def test_reports_all_expected_files_as_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            results = validate_raw_data(Path(temporary_dir))

        self.assertEqual(len(results), len(EXPECTED_M5_FILES))
        self.assertTrue(all(not result.exists for result in results))
        self.assertTrue(all(not result.is_valid for result in results))

    def test_valid_small_csvs_pass_and_count_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            data_dir = Path(temporary_dir)
            for filename, contents in VALID_FIXTURES.items():
                (data_dir / filename).write_text(contents, encoding="utf-8")

            results = validate_raw_data(data_dir, count_rows=True)

        self.assertTrue(all(result.is_valid for result in results))
        self.assertTrue(all(result.row_count == 1 for result in results))
        self.assertTrue(all(result.column_count is not None for result in results))

    def test_missing_key_column_marks_file_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            data_dir = Path(temporary_dir)
            for filename, contents in VALID_FIXTURES.items():
                (data_dir / filename).write_text(contents, encoding="utf-8")

            (data_dir / "sell_prices.csv").write_text(
                "store_id,item_id,wm_yr_wk\nCA_1,FOODS_1_001,11101\n",
                encoding="utf-8",
            )
            results = validate_raw_data(data_dir)

        prices_result = next(
            result for result in results if result.filename == "sell_prices.csv"
        )
        self.assertFalse(prices_result.is_valid)
        self.assertEqual(prices_result.missing_key_columns, ("sell_price",))

    def test_cli_returns_failure_and_names_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = main(["--data-dir", temporary_dir])

        self.assertEqual(exit_code, 1)
        self.assertIn("[MISSING] sales_train_evaluation.csv", output.getvalue())
        self.assertIn("0/3 found; 0/3 valid", output.getvalue())


if __name__ == "__main__":
    unittest.main()

