"""Check that the expected raw Walmart M5 CSV files are available and usable.

This Stage 2 utility deliberately performs lightweight validation only. It does
not clean, transform, analyze, or load the data into a database.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_DATA_DIR = Path("data") / "raw"

# A short list of identifying columns is enough to catch common mistakes, such
# as placing the wrong M5 file in data/raw. This is not a full schema contract.
EXPECTED_M5_FILES: dict[str, tuple[str, ...]] = {
    "sales_train_evaluation.csv": (
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "d_1",
    ),
    "sell_prices.csv": ("store_id", "item_id", "wm_yr_wk", "sell_price"),
    "calendar.csv": (
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "d",
    ),
}


@dataclass(frozen=True)
class FileValidationResult:
    """The lightweight validation result for one expected CSV file."""

    filename: str
    path: Path
    exists: bool
    size_bytes: int | None = None
    column_count: int | None = None
    columns_preview: tuple[str, ...] = ()
    missing_key_columns: tuple[str, ...] = ()
    row_count: int | None = None
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        """Return whether the file exists, is readable, and has its key columns."""

        return (
            self.exists
            and self.error is None
            and not self.missing_key_columns
        )


def inspect_csv_file(
    data_dir: Path,
    filename: str,
    key_columns: tuple[str, ...],
    *,
    count_rows: bool = False,
) -> FileValidationResult:
    """Inspect one CSV without loading it into memory as a dataframe."""

    path = data_dir / filename
    if not path.exists():
        return FileValidationResult(filename=filename, path=path, exists=False)

    if not path.is_file():
        return FileValidationResult(
            filename=filename,
            path=path,
            exists=True,
            error="Expected a file, but this path is not a regular file.",
        )

    size_bytes = path.stat().st_size

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader, None)
            if not header:
                return FileValidationResult(
                    filename=filename,
                    path=path,
                    exists=True,
                    size_bytes=size_bytes,
                    error="The CSV is empty or has no header row.",
                )

            row_count = sum(1 for _ in reader) if count_rows else None
    except (OSError, UnicodeError, csv.Error) as exc:
        return FileValidationResult(
            filename=filename,
            path=path,
            exists=True,
            size_bytes=size_bytes,
            error=f"Could not read the CSV header: {exc}",
        )

    missing_key_columns = tuple(column for column in key_columns if column not in header)

    return FileValidationResult(
        filename=filename,
        path=path,
        exists=True,
        size_bytes=size_bytes,
        column_count=len(header),
        columns_preview=tuple(header[:12]),
        missing_key_columns=missing_key_columns,
        row_count=row_count,
    )


def validate_raw_data(
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    count_rows: bool = False,
) -> list[FileValidationResult]:
    """Validate all required raw M5 files and return their individual results."""

    return [
        inspect_csv_file(
            data_dir,
            filename,
            key_columns,
            count_rows=count_rows,
        )
        for filename, key_columns in EXPECTED_M5_FILES.items()
    ]


def format_file_size(size_bytes: int) -> str:
    """Format a file size in a compact, human-readable form."""

    size = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("Unreachable file-size conversion state")


def print_validation_report(
    results: Sequence[FileValidationResult],
    data_dir: Path,
) -> None:
    """Print a beginner-readable validation summary."""

    print("M5 raw data validation")
    print(f"Data directory: {data_dir.resolve()}")
    print()

    for result in results:
        if not result.exists:
            print(f"[MISSING] {result.filename}")
            print(f"  Expected path: {result.path}")
            continue

        print(f"[FOUND]   {result.filename}")
        if result.size_bytes is not None:
            print(f"  Size: {format_file_size(result.size_bytes)}")
        if result.column_count is not None:
            print(f"  Columns: {result.column_count}")
            print(f"  First columns: {', '.join(result.columns_preview)}")
        if result.row_count is not None:
            print(f"  Data rows: {result.row_count:,}")
        if result.missing_key_columns:
            print(f"  Missing key columns: {', '.join(result.missing_key_columns)}")
        if result.error:
            print(f"  Error: {result.error}")

    found_count = sum(result.exists for result in results)
    valid_count = sum(result.is_valid for result in results)
    print()
    print(f"Summary: {found_count}/{len(results)} found; {valid_count}/{len(results)} valid.")

    if valid_count == len(results):
        print("All required raw M5 files passed the lightweight validation.")
    else:
        print("Raw M5 data is incomplete or invalid.")
        print("See README.md for official Kaggle download and placement instructions.")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description="Check the required raw Walmart M5 CSV files.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Raw-data directory (default: data/raw).",
    )
    parser.add_argument(
        "--count-rows",
        action="store_true",
        help="Scan each CSV and report its data-row count. This can be slow.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line validator and return a process exit code."""

    args = build_parser().parse_args(argv)
    results = validate_raw_data(args.data_dir, count_rows=args.count_rows)
    print_validation_report(results, args.data_dir)
    return 0 if all(result.is_valid for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

