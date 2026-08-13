"""Build the leakage-safe SmartStock V1 feature dataset and Stage 6 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    from smartstock.features.engineering import (
        add_calendar_features,
        add_demand_history_features,
        add_weekly_price_features,
        filter_active_period,
        finalize_feature_dataset,
        load_v1_dataset,
        store_relevant_snap,
        validate_input,
    )
    from smartstock.features.feature_definitions import (
        audit_feature_manifest,
        build_feature_manifest,
        build_validation_manifest,
    )
    from smartstock.features.stage6_report import render_stage6_report
    from smartstock.features.validation import validate_chronological_plan
except ModuleNotFoundError:  # Supports direct execution from the repository root.
    from engineering import (
        add_calendar_features,
        add_demand_history_features,
        add_weekly_price_features,
        filter_active_period,
        finalize_feature_dataset,
        load_v1_dataset,
        store_relevant_snap,
        validate_input,
    )
    from feature_definitions import audit_feature_manifest, build_feature_manifest, build_validation_manifest
    from stage6_report import render_stage6_report
    from validation import validate_chronological_plan


DEFAULT_INPUT = Path("data") / "interim" / "smartstock_v1_long.csv"
DEFAULT_SUBSET_MANIFEST = Path("config") / "v1_subset.json"
DEFAULT_FEATURE_MANIFEST = Path("config") / "v1_features.json"
DEFAULT_VALIDATION_MANIFEST = Path("config") / "v1_validation.json"
DEFAULT_OUTPUT = Path("data") / "processed" / "smartstock_v1_features.csv"
DEFAULT_REPORT = Path("reports") / "stage6_feature_engineering_report.md"
DEFAULT_SUMMARY = Path("reports") / "stage6_summary.json"


def file_sha256(path: Path) -> str:
    """Calculate a streaming hash without loading the file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Convert pandas and NumPy values into deterministic JSON-compatible data."""

    if isinstance(value, pd.DataFrame):
        return [json_safe(record) for record in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_stage6(
    input_path: Path = DEFAULT_INPUT,
    subset_manifest_path: Path = DEFAULT_SUBSET_MANIFEST,
    feature_manifest_path: Path = DEFAULT_FEATURE_MANIFEST,
    validation_manifest_path: Path = DEFAULT_VALIDATION_MANIFEST,
    output_path: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
    summary_path: Path = DEFAULT_SUMMARY,
) -> dict[str, Any]:
    """Build Stage 6 once, validate it, and persist only approved outputs."""

    started = time.perf_counter()
    input_hash_before = file_sha256(input_path)
    subset_hash_before = file_sha256(subset_manifest_path)
    subset_manifest = json.loads(subset_manifest_path.read_text(encoding="utf-8"))

    print("Loading and validating the frozen Version 1 dataset ...")
    input_data = load_v1_dataset(input_path)
    input_validation = validate_input(input_data, subset_manifest)

    print("Filtering pre-launch rows and creating calendar/history features ...")
    active, active_summary = filter_active_period(input_data)
    active, calendar_checks = add_calendar_features(active)
    active["snap_active"] = store_relevant_snap(active)
    active = add_demand_history_features(active)

    print("Creating weekly-aware past-price features ...")
    featured, extreme_changes, price_review = add_weekly_price_features(active)
    feature_manifest = build_feature_manifest()
    validation_manifest = build_validation_manifest()
    source_reference = {
        "interim_dataset": input_path.as_posix(),
        "interim_dataset_sha256": input_hash_before,
        "subset_manifest": subset_manifest_path.as_posix(),
        "subset_manifest_sha256": subset_hash_before,
    }
    feature_manifest["source"] = source_reference
    validation_manifest["source"] = source_reference
    feature_names = [feature["name"] for feature in feature_manifest["features"]]
    featured, processed_summary = finalize_feature_dataset(featured, feature_names)

    leakage_audit = audit_feature_manifest(feature_manifest)
    if not leakage_audit["passed"]:
        raise ValueError(f"Feature manifest leakage audit failed: {leakage_audit}")
    chronological_validation = validate_chronological_plan(validation_manifest, featured)

    print("Writing the processed feature dataset and tracked manifests ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    processed_summary["file_size_bytes"] = output_path.stat().st_size
    processed_summary["file_size_mib"] = output_path.stat().st_size / 2**20
    processed_summary["sha256"] = file_sha256(output_path)

    write_json(feature_manifest_path, feature_manifest)
    write_json(validation_manifest_path, validation_manifest)

    input_hash_after = file_sha256(input_path)
    subset_hash_after = file_sha256(subset_manifest_path)
    source_integrity = {
        "interim_csv_sha256_before": input_hash_before,
        "interim_csv_sha256_after": input_hash_after,
        "subset_manifest_sha256_before": subset_hash_before,
        "subset_manifest_sha256_after": subset_hash_after,
        "unchanged": input_hash_before == input_hash_after and subset_hash_before == subset_hash_after,
    }
    if not source_integrity["unchanged"]:
        raise RuntimeError("A frozen Stage 4 source changed during the Stage 6 build.")

    constant_columns = [column for column in featured.columns if featured[column].nunique(dropna=False) == 1]
    event_snap = {
        "event_rows": int(featured["is_event"].sum()),
        "snap_active_rows": int(featured["snap_active"].sum()),
        "snap_mapping": {"CA_1": "snap_CA", "TX_2": "snap_TX", "WI_3": "snap_WI"},
    }
    summary: dict[str, Any] = {
        "stage": "Stage 6 - Leakage-Safe Feature Engineering & Chronological Validation Design",
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "input_validation": input_validation,
        "active_period": active_summary,
        "calendar_validation": calendar_checks,
        "event_snap": event_snap,
        "processed_dataset": processed_summary,
        "price_review": price_review,
        "extreme_price_changes": extreme_changes,
        "leakage_audit": leakage_audit,
        "chronological_validation": chronological_validation,
        "constant_columns": constant_columns,
        "feature_counts_by_category": pd.Series([feature["category"] for feature in feature_manifest["features"]]).value_counts().sort_index().to_dict(),
        "performance": {
            "input_csv_load_count": 1,
            "input_memory_mib": input_validation["memory_mib"],
            "feature_dataframe_memory_mib": processed_summary["memory_mib"],
            "weekly_price_granularity": True,
            "full_row_python_loops": False,
            "split_labels_persisted": False,
            "runtime_seconds_before_report_write": round(time.perf_counter() - started, 3),
        },
        "source_integrity": source_integrity,
        "output": {
            "processed_dataset": output_path.as_posix(),
            "feature_manifest": feature_manifest_path.as_posix(),
            "validation_manifest": validation_manifest_path.as_posix(),
            "report": report_path.as_posix(),
            "summary": summary_path.as_posix(),
        },
        "scope": {
            "models_trained": False,
            "forecasts_generated": False,
            "model_metrics_calculated": False,
            "persistent_target_transformation": False,
        },
    }
    summary = json_safe(summary)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_stage6_report(summary, feature_manifest, validation_manifest), encoding="utf-8")
    write_json(summary_path, summary)

    print("\nStage 6 feature engineering complete")
    print(f"Active rows: {processed_summary['rows']:,}")
    print(f"Feature-ready rows: {processed_summary['feature_ready_rows']:,}")
    print(f"Columns: {processed_summary['columns']}")
    print(f"Processed dataset: {output_path.resolve()}")
    print(f"Report: {report_path.resolve()}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--subset-manifest", type=Path, default=DEFAULT_SUBSET_MANIFEST)
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_stage6(
        input_path=args.input,
        subset_manifest_path=args.subset_manifest,
        feature_manifest_path=args.feature_manifest,
        validation_manifest_path=args.validation_manifest,
        output_path=args.output,
        report_path=args.report,
        summary_path=args.summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
