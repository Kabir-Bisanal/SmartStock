"""Chronological validation checks for SmartStock Stage 6."""

from __future__ import annotations

from typing import Any

import pandas as pd


def validate_manifest_dates(manifest: dict[str, Any]) -> dict[str, bool]:
    """Validate frozen boundaries without requiring a feature dataset."""

    parsed_folds = [
        (fold["name"], pd.Timestamp(fold["start_date"]), pd.Timestamp(fold["end_date"]))
        for fold in manifest["validation_folds"]
    ]
    test_start = pd.Timestamp(manifest["locked_final_test"]["start_date"])
    test_end = pd.Timestamp(manifest["locked_final_test"]["end_date"])
    return {
        "three_validation_folds": len(parsed_folds) == 3,
        "each_validation_fold_is_30_days": [(end - start).days + 1 for _, start, end in parsed_folds] == [30, 30, 30],
        "validation_folds_disjoint": all(parsed_folds[index][2] < parsed_folds[index + 1][1] for index in range(len(parsed_folds) - 1)),
        "validation_folds_contiguous": all(parsed_folds[index][2] + pd.Timedelta(days=1) == parsed_folds[index + 1][1] for index in range(len(parsed_folds) - 1)),
        "final_test_disjoint_from_validation": all(end < test_start or start > test_end for _, start, end in parsed_folds),
        "final_test_starts_after_last_validation": parsed_folds[-1][2] + pd.Timedelta(days=1) == test_start,
        "final_test_is_30_days": (test_end - test_start).days + 1 == 30,
        "random_split_disabled": manifest["random_split_allowed"] is False,
        "final_test_locked": manifest["locked_final_test"]["locked"] is True,
    }


def validate_chronological_plan(manifest: dict[str, Any], features: pd.DataFrame) -> dict[str, Any]:
    """Verify exact fold boundaries, expanding training rules, and the test lock."""

    folds = manifest["validation_folds"]
    parsed_folds: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for fold in folds:
        start = pd.Timestamp(fold["start_date"])
        end = pd.Timestamp(fold["end_date"])
        parsed_folds.append((fold["name"], start, end))

    test_start = pd.Timestamp(manifest["locked_final_test"]["start_date"])
    test_end = pd.Timestamp(manifest["locked_final_test"]["end_date"])
    fold_statistics = []
    for name, start, end in parsed_folds:
        training = features[features["is_feature_ready"] & features["date"].lt(start)]
        validation = features[features["is_feature_ready"] & features["date"].between(start, end)]
        fold_statistics.append(
            {
                "name": name,
                "training_end_date": training["date"].max().date().isoformat(),
                "training_rows": len(training),
                "validation_start_date": start.date().isoformat(),
                "validation_end_date": end.date().isoformat(),
                "validation_calendar_days": (end - start).days + 1,
                "validation_rows": len(validation),
                "validation_items": int(validation["item_id"].nunique()),
                "validation_stores": int(validation["store_id"].nunique()),
                "validation_pairs": int(validation.groupby(["store_id", "item_id"], observed=True).ngroups),
            }
        )

    final_test = features[features["is_feature_ready"] & features["date"].between(test_start, test_end)]
    checks = {
        **validate_manifest_dates(manifest),
        "final_test_ends_on_last_observed_date": test_end == features["date"].max(),
        "all_validation_rows_feature_ready": all(stat["validation_rows"] == 9_000 for stat in fold_statistics),
        "final_test_rows_feature_ready": len(final_test) == 9_000,
        "expanding_training_rows_increase": all(fold_statistics[index]["training_rows"] < fold_statistics[index + 1]["training_rows"] for index in range(len(fold_statistics) - 1)),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Chronological validation plan failed: {', '.join(failed)}")
    return {
        "checks": checks,
        "folds": fold_statistics,
        "locked_final_test": {
            "start_date": test_start.date().isoformat(),
            "end_date": test_end.date().isoformat(),
            "calendar_days": (test_end - test_start).days + 1,
            "feature_ready_rows": len(final_test),
            "items": int(final_test["item_id"].nunique()),
            "stores": int(final_test["store_id"].nunique()),
            "item_store_pairs": int(final_test.groupby(["store_id", "item_id"], observed=True).ngroups),
        },
    }
