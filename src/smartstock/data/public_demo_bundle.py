"""Build and validate the compact, versioned SmartStock public demo bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from smartstock.database.loaders import (
    ArtifactPaths,
    FORECAST_COLUMNS,
    RECOMMENDATION_COLUMNS,
    SNAPSHOT_COLUMNS,
    file_sha256,
    validate_artifact_contracts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PUBLIC_DEMO_VERSION = "1.0.0"
PUBLIC_HISTORY_DAYS = 120
PUBLIC_HISTORY_COLUMNS = [
    "date",
    "store_id",
    "item_id",
    "dept_id",
    "sales",
    "sell_price",
    "weekday",
]
EXPECTED_STORES = ["CA_1", "TX_2", "WI_3"]
EXPECTED_PUBLIC_COUNTS = {
    "history": 36_000,
    "forecasts": 9_000,
    "inventory_snapshot": 300,
    "recommendations": 300,
}


@dataclass(frozen=True)
class PublicDemoPaths:
    """Paths for the intentionally committed, recruiter-facing demo data."""

    history: Path = PROJECT_ROOT / "data" / "public_demo" / "history_120d.csv"
    forecasts: Path = PROJECT_ROOT / "data" / "public_demo" / "production_forecasts.csv"
    inventory_snapshot: Path = PROJECT_ROOT / "data" / "public_demo" / "inventory_snapshot.csv"
    recommendations: Path = PROJECT_ROOT / "data" / "public_demo" / "inventory_recommendations.csv"
    metadata: Path = PROJECT_ROOT / "data" / "public_demo" / "metadata.json"

    @classmethod
    def from_directory(cls, directory: Path) -> "PublicDemoPaths":
        root = Path(directory)
        return cls(
            history=root / "history_120d.csv",
            forecasts=root / "production_forecasts.csv",
            inventory_snapshot=root / "inventory_snapshot.csv",
            recommendations=root / "inventory_recommendations.csv",
            metadata=root / "metadata.json",
        )

    def required_files(self) -> tuple[Path, ...]:
        return (
            self.history,
            self.forecasts,
            self.inventory_snapshot,
            self.recommendations,
            self.metadata,
        )


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _require_columns(path: Path, required: list[str]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required public demo artifact is missing: {path}")
    available = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(required) - set(available))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")


def load_public_demo_metadata(paths: PublicDemoPaths = PublicDemoPaths()) -> dict[str, Any]:
    if not paths.metadata.is_file():
        raise FileNotFoundError(f"Required public demo metadata is missing: {paths.metadata}")
    return json.loads(paths.metadata.read_text(encoding="utf-8"))


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _copy_csv_with_lf(source: Path, destination: Path) -> None:
    """Copy a frozen CSV without parsing values while normalizing line endings."""

    destination.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))


def build_public_demo_bundle(
    source_paths: ArtifactPaths = ArtifactPaths(),
    output_paths: PublicDemoPaths = PublicDemoPaths(),
    *,
    stage9_summary_path: Path = PROJECT_ROOT / "reports" / "stage9_summary.json",
    stage10_summary_path: Path = PROJECT_ROOT / "reports" / "stage10_summary.json",
    history_days: int = PUBLIC_HISTORY_DAYS,
) -> dict[str, Any]:
    """Extract deterministic public files without recomputing forecasts or decisions."""

    if int(history_days) != PUBLIC_HISTORY_DAYS:
        raise ValueError(f"The Version 1 public bundle is frozen to {PUBLIC_HISTORY_DAYS} history days.")
    validate_artifact_contracts(source_paths)
    output_paths.metadata.parent.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(
        source_paths.history,
        usecols=PUBLIC_HISTORY_COLUMNS,
        parse_dates=["date"],
    )
    history_end = history["date"].max()
    history_start = history_end - pd.Timedelta(days=history_days - 1)
    history = history.loc[history["date"].between(history_start, history_end)].copy()
    history = history.sort_values(["date", "store_id", "item_id"], kind="stable")
    history["date"] = history["date"].dt.strftime("%Y-%m-%d")
    history = history[PUBLIC_HISTORY_COLUMNS].reset_index(drop=True)

    forecasts = pd.read_csv(source_paths.forecasts, usecols=FORECAST_COLUMNS)
    forecasts = forecasts.sort_values(
        ["target_date", "store_id", "item_id"], kind="stable"
    ).reset_index(drop=True)
    snapshot = pd.read_csv(source_paths.inventory_snapshot, usecols=SNAPSHOT_COLUMNS)
    snapshot = snapshot.sort_values(["store_id", "item_id"], kind="stable").reset_index(drop=True)
    recommendations = pd.read_csv(
        source_paths.recommendations, usecols=RECOMMENDATION_COLUMNS
    )
    recommendations = recommendations.sort_values(
        ["store_id", "item_id"], kind="stable"
    ).reset_index(drop=True)

    _write_csv(history, output_paths.history)
    _copy_csv_with_lf(source_paths.forecasts, output_paths.forecasts)
    _copy_csv_with_lf(source_paths.inventory_snapshot, output_paths.inventory_snapshot)
    _copy_csv_with_lf(source_paths.recommendations, output_paths.recommendations)

    stage9 = json.loads(Path(stage9_summary_path).read_text(encoding="utf-8"))
    stage10 = json.loads(Path(stage10_summary_path).read_text(encoding="utf-8"))
    test_metrics = stage9["locked_final_test"]["days_1_30_daily"]
    test_aggregate = stage9["locked_final_test"]["days_1_30_aggregate"]
    recommendation_results = stage10["recommendation_results"]

    bundle_files = {}
    for name, path in (
        ("history", output_paths.history),
        ("forecasts", output_paths.forecasts),
        ("inventory_snapshot", output_paths.inventory_snapshot),
        ("recommendations", output_paths.recommendations),
    ):
        bundle_files[name] = {
            "path": _portable_path(path),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }

    metadata = {
        "bundle_name": "SmartStock compact public demo",
        "bundle_version": PUBLIC_DEMO_VERSION,
        "purpose": "Committed recruiter-facing data; not raw M5 or the full modeling dataset.",
        "history": {
            "source_observations_analyzed": 582_300,
            "bundled_observations": int(len(history)),
            "bundled_days_per_series": history_days,
            "date_start": history_start.date().isoformat(),
            "date_end": history_end.date().isoformat(),
            "columns": PUBLIC_HISTORY_COLUMNS,
        },
        "portfolio_facts": {
            "dataset": "Walmart M5 Forecasting - Accuracy",
            "category": "FOODS",
            "products": 100,
            "stores": EXPECTED_STORES,
            "item_store_series": 300,
            "source_observations": 582_300,
            "production_forecast_rows": int(len(forecasts)),
            "forecast_horizon_days": int(forecasts["horizon_day"].max()),
            "inventory_decisions": int(len(recommendations)),
            "reorder_decisions": int(recommendations["recommended_order_qty"].gt(0).sum()),
            "recommended_units": int(recommendations["recommended_order_qty"].sum()),
            "expected_shortage_before": float(
                recommendations["expected_shortage_without_order"].sum()
            ),
            "expected_shortage_after": float(
                recommendations["expected_shortage_with_recommendation"].sum()
            ),
            "inventory_data_source": "synthetic_demo_not_walmart",
        },
        "model": {
            "selected_model": stage9["selection"]["selected_model_name"],
            "display_name": "28-Day Historical Mean",
            "forecast_origin": str(forecasts["forecast_origin"].max()),
            "forecast_horizon_days": int(forecasts["horizon_day"].max()),
            "runtime_model_required": False,
            "locked_test": {
                "mae": float(test_metrics["mae"]),
                "rmse": float(test_metrics["rmse"]),
                "wape": float(test_metrics["wape"]),
                "rmsse": float(test_metrics["rmsse"]),
                "bias": float(test_metrics["bias"]),
                "aggregate_30d_wape": float(test_aggregate["wape"]),
            },
        },
        "scientific_integrity": {
            "forecasts_recomputed": False,
            "inventory_recommendations_recomputed": False,
            "inventory_formulas_changed": False,
            "source_forecast_sha256": file_sha256(source_paths.forecasts),
            "source_snapshot_sha256": file_sha256(source_paths.inventory_snapshot),
            "source_recommendations_sha256": file_sha256(source_paths.recommendations),
            "expected_reorder_decisions": int(recommendation_results["requiring_reorder"]),
            "expected_recommended_units": int(recommendation_results["total_recommended_units"]),
        },
        "build": {
            "history_selection": "latest 120 dates, inclusive, for every frozen item-store series",
            "row_order": "date/store/item for compact history; frozen source order retained for saved outputs",
            "model_artifact_included": False,
            "source_paths": {
                "history": _portable_path(source_paths.history),
                "forecasts": _portable_path(source_paths.forecasts),
                "inventory_snapshot": _portable_path(source_paths.inventory_snapshot),
                "recommendations": _portable_path(source_paths.recommendations),
            },
        },
        "files": bundle_files,
    }
    output_paths.metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validate_public_demo_bundle(output_paths)


def validate_public_demo_bundle(
    paths: PublicDemoPaths = PublicDemoPaths(),
) -> dict[str, Any]:
    """Validate public cardinalities, coverage, provenance, and immutable portfolio facts."""

    _require_columns(paths.history, PUBLIC_HISTORY_COLUMNS)
    _require_columns(paths.forecasts, FORECAST_COLUMNS)
    _require_columns(paths.inventory_snapshot, SNAPSHOT_COLUMNS)
    _require_columns(paths.recommendations, RECOMMENDATION_COLUMNS)
    metadata = load_public_demo_metadata(paths)
    if metadata.get("bundle_version") != PUBLIC_DEMO_VERSION:
        raise ValueError("Public demo bundle version does not match the supported Version 1 contract.")

    history = pd.read_csv(paths.history, parse_dates=["date"])
    forecasts = pd.read_csv(paths.forecasts, parse_dates=["forecast_origin", "target_date"])
    snapshot = pd.read_csv(paths.inventory_snapshot)
    recommendations = pd.read_csv(paths.recommendations)
    counts = {
        "history": len(history),
        "forecasts": len(forecasts),
        "inventory_snapshot": len(snapshot),
        "recommendations": len(recommendations),
    }
    if counts != EXPECTED_PUBLIC_COUNTS:
        raise ValueError(f"Public demo row counts differ from the Version 1 contract: {counts}")

    history_keys = ["date", "store_id", "item_id"]
    if history.duplicated(history_keys).any():
        raise ValueError("Public history contains duplicate date/item-store keys.")
    if not history.equals(history.sort_values(history_keys, kind="stable").reset_index(drop=True)):
        raise ValueError("Public history is not deterministically ordered.")
    series_days = history.groupby(["store_id", "item_id"], observed=True)["date"].nunique()
    if len(series_days) != 300 or not series_days.eq(PUBLIC_HISTORY_DAYS).all():
        raise ValueError("Public history must contain 120 dates for all 300 item-store series.")
    if sorted(history["store_id"].astype(str).unique()) != EXPECTED_STORES:
        raise ValueError("Public history store coverage differs from the frozen subset.")
    if history["item_id"].nunique() != 100 or history["dept_id"].nunique() != 3:
        raise ValueError("Public history product or department coverage is incomplete.")
    if (history["date"].max() - history["date"].min()).days != PUBLIC_HISTORY_DAYS - 1:
        raise ValueError("Public history does not cover one contiguous 120-day window.")

    if forecasts.duplicated(["target_date", "store_id", "item_id"]).any():
        raise ValueError("Public forecasts contain duplicate target-date/item-store keys.")
    forecast_days = forecasts.groupby(["store_id", "item_id"], observed=True)["horizon_day"]
    if forecast_days.ngroups != 300 or not forecast_days.nunique().eq(30).all():
        raise ValueError("Public forecasts must cover 30 days for all 300 series.")
    if forecasts["forecast"].lt(0).any() or not forecasts["horizon_day"].between(1, 30).all():
        raise ValueError("Public forecasts violate non-negative or horizon constraints.")
    if not forecasts["forecast_origin"].eq(history["date"].max()).all():
        raise ValueError("Public forecast origin must equal the final bundled history date.")

    for name, frame in (("snapshot", snapshot), ("recommendations", recommendations)):
        if frame.duplicated(["store_id", "item_id"]).any():
            raise ValueError(f"Public {name} contains duplicate item-store keys.")
        if not frame["inventory_source"].eq("synthetic_demo").all():
            raise ValueError(f"Public {name} is not explicitly labeled synthetic_demo.")
    if int(recommendations["recommended_order_qty"].gt(0).sum()) != 198:
        raise ValueError("Public recommendations do not preserve the 198 reorder decisions.")
    if int(recommendations["recommended_order_qty"].sum()) != 4_143:
        raise ValueError("Public recommendations do not preserve the 4,143 recommended units.")

    facts = metadata.get("portfolio_facts", {})
    expected_facts = {
        "source_observations": 582_300,
        "products": 100,
        "item_store_series": 300,
        "production_forecast_rows": 9_000,
        "inventory_decisions": 300,
        "reorder_decisions": 198,
        "recommended_units": 4_143,
        "forecast_horizon_days": 30,
    }
    for name, expected in expected_facts.items():
        if facts.get(name) != expected:
            raise ValueError(f"Public metadata fact {name!r} does not equal {expected!r}.")
    if metadata.get("model", {}).get("runtime_model_required") is not False:
        raise ValueError("The saved-output public demo must not require a runtime model artifact.")

    for name, path in (
        ("history", paths.history),
        ("forecasts", paths.forecasts),
        ("inventory_snapshot", paths.inventory_snapshot),
        ("recommendations", paths.recommendations),
    ):
        recorded = metadata.get("files", {}).get(name, {})
        if recorded.get("sha256") != file_sha256(path):
            raise ValueError(f"Public demo checksum mismatch for {name}.")
        if recorded.get("size_bytes") != path.stat().st_size:
            raise ValueError(f"Public demo size metadata mismatch for {name}.")

    return {
        "bundle_version": PUBLIC_DEMO_VERSION,
        "row_counts": counts,
        "history_start": history["date"].min().date().isoformat(),
        "history_end": history["date"].max().date().isoformat(),
        "history_days": int(history["date"].nunique()),
        "stores": sorted(history["store_id"].astype(str).unique().tolist()),
        "items": int(history["item_id"].nunique()),
        "series": int(series_days.size),
        "forecast_start": forecasts["target_date"].min().date().isoformat(),
        "forecast_end": forecasts["target_date"].max().date().isoformat(),
        "forecast_days": int(forecasts["target_date"].nunique()),
        "reorder_decisions": int(recommendations["recommended_order_qty"].gt(0).sum()),
        "recommended_units": int(recommendations["recommended_order_qty"].sum()),
        "total_size_bytes": sum(path.stat().st_size for path in paths.required_files()),
    }
