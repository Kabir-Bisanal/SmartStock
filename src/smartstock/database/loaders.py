"""Validated, idempotent bulk loading of Stage 10 artifacts into SQLAlchemy tables."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sqlalchemy import Engine, delete, func, select

from smartstock.database.schema import TABLES, create_schema


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_ROW_COUNTS = {
    "demand_history": 582_300,
    "forecasts": 9_000,
    "inventory_snapshot": 300,
    "inventory_recommendations": 300,
    "model_metadata": 1,
}

HISTORY_COLUMNS = [
    "date", "store_id", "item_id", "d", "dept_id", "cat_id", "state_id", "sales",
    "sell_price", "weekday", "wday", "month", "year", "event_name_1", "event_type_1",
    "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI", "demand_band",
]
FORECAST_COLUMNS = [
    "item_id", "store_id", "dept_id", "demand_band", "forecast_origin", "target_date",
    "horizon_day", "forecast",
]
SNAPSHOT_COLUMNS = [
    "item_id", "store_id", "dept_id", "demand_band", "on_hand", "on_order", "backorders",
    "lead_time_days", "service_level", "holding_cost_per_unit_per_day", "stockout_cost_per_unit",
    "fixed_order_cost", "inventory_source", "synthetic_profile_plan",
]
RECOMMENDATION_COLUMNS = [column.name for column in TABLES["inventory_recommendations"].columns]


@dataclass(frozen=True)
class ArtifactPaths:
    """Machine-independent paths forming the Stage 10-to-application data contract."""

    history: Path = PROJECT_ROOT / "data" / "interim" / "smartstock_v1_long.csv"
    forecasts: Path = PROJECT_ROOT / "data" / "processed" / "production" / "smartstock_v1_30day_forecast.csv"
    inventory_snapshot: Path = PROJECT_ROOT / "data" / "simulated" / "smartstock_v1_inventory_snapshot.csv"
    recommendations: Path = PROJECT_ROOT / "data" / "processed" / "inventory" / "smartstock_v1_inventory_recommendations.csv"
    final_model_config: Path = PROJECT_ROOT / "config" / "v1_final_model.json"
    inventory_policy: Path = PROJECT_ROOT / "config" / "v1_inventory_policy.json"
    deployment_model: Path = PROJECT_ROOT / "models" / "smartstock_v1_deployment_forecaster.joblib"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def _require_columns(path: Path, required: list[str]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required SmartStock artifact is missing: {path}")
    missing = sorted(set(required) - set(_header(path)))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")


def validate_artifact_contracts(
    paths: ArtifactPaths = ArtifactPaths(), *, strict_expected: bool = True
) -> dict[str, Any]:
    """Validate required files, schemas, key uniqueness, and real/synthetic labeling."""

    _require_columns(paths.history, HISTORY_COLUMNS)
    _require_columns(paths.forecasts, FORECAST_COLUMNS)
    _require_columns(paths.inventory_snapshot, SNAPSHOT_COLUMNS)
    _require_columns(paths.recommendations, RECOMMENDATION_COLUMNS)
    for path in (paths.final_model_config, paths.inventory_policy, paths.deployment_model):
        if not path.exists():
            raise FileNotFoundError(f"Required SmartStock artifact is missing: {path}")

    counts = {
        "demand_history": csv_row_count(paths.history),
        "forecasts": csv_row_count(paths.forecasts),
        "inventory_snapshot": csv_row_count(paths.inventory_snapshot),
        "inventory_recommendations": csv_row_count(paths.recommendations),
        "model_metadata": 1,
    }
    if strict_expected and counts != EXPECTED_ROW_COUNTS:
        raise ValueError(f"Stage 10 artifact row counts differ from the frozen contract: {counts}")

    forecast_keys = pd.read_csv(
        paths.forecasts,
        usecols=["target_date", "store_id", "item_id", "horizon_day", "forecast"],
        parse_dates=["target_date"],
    )
    if forecast_keys.duplicated(["target_date", "store_id", "item_id"]).any():
        raise ValueError("Forecast artifact contains duplicate target-date/item-store keys.")
    if forecast_keys["forecast"].lt(0).any() or not forecast_keys["horizon_day"].between(1, 30).all():
        raise ValueError("Forecast values or horizon days violate the Stage 10 contract.")

    snapshot_keys = pd.read_csv(
        paths.inventory_snapshot, usecols=["store_id", "item_id", "inventory_source"]
    )
    recommendation_keys = pd.read_csv(
        paths.recommendations, usecols=["store_id", "item_id", "inventory_source"]
    )
    for name, frame in (("inventory snapshot", snapshot_keys), ("recommendations", recommendation_keys)):
        if frame.duplicated(["store_id", "item_id"]).any():
            raise ValueError(f"{name.title()} contains duplicate item-store keys.")
        if not frame["inventory_source"].eq("synthetic_demo").all():
            raise ValueError(f"{name.title()} is not explicitly labeled synthetic_demo.")

    return {
        "row_counts": counts,
        "forecast_series": int(forecast_keys.groupby(["store_id", "item_id"]).ngroups),
        "forecast_days": int(forecast_keys["target_date"].nunique()),
        "snapshot_series": len(snapshot_keys),
        "recommendation_series": len(recommendation_keys),
        "synthetic_label_verified": True,
    }


def iter_history_chunks(path: Path, *, chunksize: int = 50_000) -> Iterator[pd.DataFrame]:
    """Read and structurally prepare history in bounded chunks for bulk insertion."""

    dtype = {
        "store_id": "string", "item_id": "string", "d": "string", "dept_id": "string",
        "cat_id": "string", "state_id": "string", "sales": "int32", "sell_price": "float64",
        "weekday": "string", "wday": "int16", "month": "int16", "year": "int16",
        "event_name_1": "string", "event_type_1": "string", "event_name_2": "string",
        "event_type_2": "string", "snap_CA": "int8", "snap_TX": "int8", "snap_WI": "int8",
        "demand_band": "string",
    }
    for chunk in pd.read_csv(
        path, usecols=HISTORY_COLUMNS, parse_dates=["date"], dtype=dtype, chunksize=chunksize
    ):
        chunk["snap_active"] = np.select(
            [chunk["state_id"].eq("CA"), chunk["state_id"].eq("TX"), chunk["state_id"].eq("WI")],
            [chunk["snap_CA"], chunk["snap_TX"], chunk["snap_WI"]],
            default=0,
        ).astype(bool)
        yield chunk.drop(columns=["snap_CA", "snap_TX", "snap_WI"])


def build_model_metadata(paths: ArtifactPaths) -> pd.DataFrame:
    """Build one reproducibility record from tracked configs and the known local artifact."""

    final_config = json.loads(paths.final_model_config.read_text(encoding="utf-8"))
    policy = json.loads(paths.inventory_policy.read_text(encoding="utf-8"))
    forecast = pd.read_csv(paths.forecasts, usecols=["forecast_origin", "horizon_day"], nrows=9_000)
    origin = pd.to_datetime(forecast["forecast_origin"]).max()
    return pd.DataFrame(
        [{
            "metadata_id": "smartstock_v1",
            "selected_model": final_config["selected_model_name"],
            "model_version": "v1",
            "forecast_origin": origin.date(),
            "deployment_train_through": pd.Timestamp("2016-05-22").date(),
            "forecast_horizon_days": int(forecast["horizon_day"].max()),
            "model_artifact_path": "models/smartstock_v1_deployment_forecaster.joblib",
            "model_artifact_sha256": file_sha256(paths.deployment_model),
            "inventory_policy_name": policy["policy_name"],
            "inventory_policy_version": "v1",
            "inventory_policy_sha256": file_sha256(paths.inventory_policy),
            "inventory_data_source": "synthetic_demo_not_walmart",
            "loaded_at_utc": datetime.now(UTC),
        }]
    )


def table_row_counts(engine: Engine) -> dict[str, int]:
    with engine.connect() as connection:
        return {
            name: int(connection.execute(select(func.count()).select_from(table)).scalar_one())
            for name, table in TABLES.items()
        }


def validate_loaded_database(
    engine: Engine, *, expected_counts: dict[str, int] | None = None
) -> dict[str, Any]:
    """Validate loaded cardinalities and key business invariants."""

    counts = table_row_counts(engine)
    expected = expected_counts or EXPECTED_ROW_COUNTS
    if counts != expected:
        raise ValueError(f"Database row counts do not match expected values: {counts}")
    with engine.connect() as connection:
        history_series = len(connection.execute(select(
            TABLES["demand_history"].c.store_id, TABLES["demand_history"].c.item_id
        ).distinct()).all())
        forecast_series = len(connection.execute(select(
            TABLES["forecasts"].c.store_id, TABLES["forecasts"].c.item_id
        ).distinct()).all())
        stores = len(connection.execute(select(TABLES["demand_history"].c.store_id).distinct()).all())
        items = len(connection.execute(select(TABLES["demand_history"].c.item_id).distinct()).all())
    if expected == EXPECTED_ROW_COUNTS and (history_series, forecast_series, stores, items) != (300, 300, 3, 100):
        raise ValueError("Database series/store/item cardinalities do not match SmartStock V1.")
    return {
        "row_counts": counts,
        "history_series": history_series,
        "forecast_series": forecast_series,
        "stores": stores,
        "items": items,
    }


def load_database_artifacts(
    engine: Engine,
    paths: ArtifactPaths = ArtifactPaths(),
    *,
    refresh: bool = False,
    strict_expected: bool = True,
    history_chunksize: int = 50_000,
) -> dict[str, Any]:
    """Create and bulk-load all application tables, safely skipping an identical load."""

    started = perf_counter()
    contract = validate_artifact_contracts(paths, strict_expected=strict_expected)
    expected_counts = contract["row_counts"]
    create_schema(engine)
    current = table_row_counts(engine)
    if any(current.values()) and not refresh:
        if current == expected_counts:
            validated = validate_loaded_database(engine, expected_counts=expected_counts)
            return {**validated, "load_action": "skipped_existing_complete_database", "runtime_seconds": round(perf_counter() - started, 3)}
        raise RuntimeError(
            "Database contains a partial or outdated SmartStock load. Re-run with --refresh."
        )

    with engine.begin() as connection:
        if refresh:
            for table_name in reversed(list(TABLES)):
                connection.execute(delete(TABLES[table_name]))
        for history_chunk in iter_history_chunks(paths.history, chunksize=history_chunksize):
            history_chunk = history_chunk.copy()
            history_chunk["date"] = history_chunk["date"].dt.date
            history_chunk.to_sql("demand_history", connection, if_exists="append", index=False, chunksize=5_000)

        forecast = pd.read_csv(paths.forecasts, parse_dates=["forecast_origin", "target_date"])
        forecast["forecast_origin"] = forecast["forecast_origin"].dt.date
        forecast["target_date"] = forecast["target_date"].dt.date
        forecast["model_name"] = "mean_28"
        forecast["model_version"] = "v1"
        forecast.to_sql("forecasts", connection, if_exists="append", index=False, chunksize=5_000)

        snapshot = pd.read_csv(paths.inventory_snapshot)
        snapshot.to_sql("inventory_snapshot", connection, if_exists="append", index=False, chunksize=1_000)

        recommendations = pd.read_csv(paths.recommendations)
        recommendations.to_sql("inventory_recommendations", connection, if_exists="append", index=False, chunksize=1_000)

        build_model_metadata(paths).to_sql("model_metadata", connection, if_exists="append", index=False)

    validated = validate_loaded_database(engine, expected_counts=expected_counts)
    return {
        **validated,
        "load_action": "refreshed" if refresh else "created_and_loaded",
        "runtime_seconds": round(perf_counter() - started, 3),
        "contract": contract,
    }
