"""Interchangeable PostgreSQL and read-only local CSV data sources for Streamlit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import Engine

from smartstock.database import queries
from smartstock.database.connection import DatabaseSettings, create_database_engine, verify_connection
from smartstock.database.loaders import ArtifactPaths, build_model_metadata, iter_history_chunks, validate_artifact_contracts, validate_loaded_database
from smartstock.data.public_demo_bundle import (
    PublicDemoPaths,
    load_public_demo_metadata,
    validate_public_demo_bundle,
)


def _filter_frame(frame: pd.DataFrame, filters: dict[str, Any] | None = None) -> pd.DataFrame:
    selected = frame
    for name, value in (filters or {}).items():
        if value in (None, "", "All") or name not in selected.columns:
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            selected = selected[selected[name].isin(list(value))]
        else:
            selected = selected[selected[name].eq(value)]
    return selected.copy()


class PostgresDataSource:
    """Read-only application adapter backed by parameterized SQLAlchemy queries."""

    label = "PostgreSQL"

    def __init__(self, engine: Engine):
        self.engine = engine

    def get_filter_options(self) -> dict[str, list[str]]:
        return queries.get_filter_options(self.engine)

    def get_overview_metrics(self, filters: dict[str, Any] | None = None) -> dict[str, float | int]:
        return queries.get_overview_metrics(self.engine, filters)

    def get_sales_trend(self, **kwargs: Any) -> pd.DataFrame:
        return queries.get_sales_trend(self.engine, **kwargs)

    def get_sales_breakdown(self, group_by: str, **kwargs: Any) -> pd.DataFrame:
        return queries.get_sales_breakdown(self.engine, group_by, **kwargs)

    def get_demand_profile(self, **kwargs: Any) -> dict[str, float | int]:
        return queries.get_demand_profile(self.engine, **kwargs)

    def get_recent_history(self, store_id: str, item_id: str, *, days: int = 90) -> pd.DataFrame:
        return queries.get_recent_history(self.engine, store_id, item_id, days=days)

    def get_forecast(self, store_id: str | None = None, item_id: str | None = None, *, horizon_days: int = 30) -> pd.DataFrame:
        return queries.get_forecast(self.engine, store_id, item_id, horizon_days=horizon_days)

    def get_recommendations(self, filters: dict[str, Any] | None = None) -> pd.DataFrame:
        return queries.get_inventory_recommendations(self.engine, filters)

    def get_snapshot(self, store_id: str, item_id: str) -> pd.DataFrame:
        return queries.get_inventory_snapshot(self.engine, store_id, item_id)

    def get_model_metadata(self) -> dict[str, Any]:
        return queries.get_model_metadata(self.engine)


class CsvDataSource:
    """Read-only fallback using the same frozen generated artifacts as PostgreSQL."""

    label = "Local demo files"

    def __init__(self, paths: ArtifactPaths = ArtifactPaths(), *, strict_expected: bool = True):
        validate_artifact_contracts(paths, strict_expected=strict_expected)
        self.paths = paths
        self._history_cache: pd.DataFrame | None = None
        self._forecast_cache: pd.DataFrame | None = None
        self._snapshot_cache: pd.DataFrame | None = None
        self._recommendation_cache: pd.DataFrame | None = None

    def _history(self) -> pd.DataFrame:
        if self._history_cache is None:
            self._history_cache = pd.concat(iter_history_chunks(self.paths.history), ignore_index=True)
        return self._history_cache

    def _forecasts(self) -> pd.DataFrame:
        if self._forecast_cache is None:
            frame = pd.read_csv(self.paths.forecasts, parse_dates=["forecast_origin", "target_date"])
            frame["model_name"] = "mean_28"
            frame["model_version"] = "v1"
            self._forecast_cache = frame
        return self._forecast_cache

    def _snapshot(self) -> pd.DataFrame:
        if self._snapshot_cache is None:
            self._snapshot_cache = pd.read_csv(self.paths.inventory_snapshot)
        return self._snapshot_cache

    def _recommendations(self) -> pd.DataFrame:
        if self._recommendation_cache is None:
            self._recommendation_cache = pd.read_csv(self.paths.recommendations)
        return self._recommendation_cache

    def get_filter_options(self) -> dict[str, list[str]]:
        history = self._history()
        return {
            "stores": sorted(history["store_id"].astype(str).unique().tolist()),
            "departments": sorted(history["dept_id"].astype(str).unique().tolist()),
            "items": sorted(history["item_id"].astype(str).unique().tolist()),
        }

    def get_overview_metrics(self, filters: dict[str, Any] | None = None) -> dict[str, float | int]:
        history = _filter_frame(self._history(), filters)
        forecasts = _filter_frame(self._forecasts(), filters)
        recommendations = _filter_frame(self._recommendations(), filters)
        latest = history["date"].max() if not history.empty else pd.NaT
        recent = history[history["date"].ge(latest - pd.Timedelta(29, unit="D"))] if pd.notna(latest) else history.iloc[0:0]
        return {
            "sales_observations": int(len(history)),
            "products": int(history["item_id"].nunique()),
            "stores": int(history["store_id"].nunique()),
            "series": int(history.groupby(["store_id", "item_id"], observed=True).ngroups),
            "historical_units": float(history["sales"].sum()),
            "recent_30d_units": float(recent["sales"].sum()),
            "forecast_7d": float(forecasts.loc[forecasts["horizon_day"].le(7), "forecast"].sum()),
            "forecast_30d": float(forecasts["forecast"].sum()),
            "reorder_items": int(recommendations["recommended_order_qty"].gt(0).sum()),
            "critical_stockout_items": int(recommendations["stock_status"].isin(["CRITICAL", "STOCKOUT"]).sum()),
            "recommended_units": int(recommendations["recommended_order_qty"].sum()),
            "shortage_before": float(recommendations["expected_shortage_without_order"].sum()),
            "shortage_after": float(recommendations["expected_shortage_with_recommendation"].sum()),
            "estimated_savings": float(recommendations["estimated_cost_savings"].sum()),
        }

    def _filtered_history(self, filters: dict[str, Any] | None, start_date: Any | None, end_date: Any | None) -> pd.DataFrame:
        frame = _filter_frame(self._history(), filters)
        if start_date is not None:
            frame = frame[frame["date"].ge(pd.Timestamp(start_date))]
        if end_date is not None:
            frame = frame[frame["date"].le(pd.Timestamp(end_date))]
        return frame

    def get_sales_trend(
        self, *, filters: dict[str, Any] | None = None, start_date: Any | None = None,
        end_date: Any | None = None, frequency: str = "weekly"
    ) -> pd.DataFrame:
        frame = self._filtered_history(filters, start_date, end_date)
        if frequency == "daily":
            frame = frame.assign(period=frame["date"])
        elif frequency == "weekly":
            frame = frame.assign(period=frame["date"].dt.to_period("W-SUN").dt.start_time)
        elif frequency == "monthly":
            frame = frame.assign(period=frame["date"].dt.to_period("M").dt.start_time)
        else:
            raise ValueError("frequency must be daily, weekly, or monthly.")
        return frame.groupby("period", as_index=False, observed=True)["sales"].sum()

    def get_sales_breakdown(
        self, group_by: str, *, filters: dict[str, Any] | None = None,
        start_date: Any | None = None, end_date: Any | None = None
    ) -> pd.DataFrame:
        if group_by not in {"store_id", "dept_id", "weekday"}:
            raise ValueError("Unsupported sales breakdown.")
        frame = self._filtered_history(filters, start_date, end_date)
        return frame.groupby(group_by, as_index=False, observed=True)["sales"].sum().rename(columns={group_by: "group"})

    def get_demand_profile(
        self, *, filters: dict[str, Any] | None = None,
        start_date: Any | None = None, end_date: Any | None = None
    ) -> dict[str, float | int]:
        frame = self._filtered_history(filters, start_date, end_date)
        observations = len(frame)
        zeros = int(frame["sales"].eq(0).sum())
        return {
            "observations": observations,
            "zero_observations": zeros,
            "zero_rate": zeros / observations if observations else float("nan"),
            "mean_daily_units": float(frame["sales"].mean()) if observations else 0.0,
            "maximum_daily_units": float(frame["sales"].max()) if observations else 0.0,
        }

    def get_recent_history(self, store_id: str, item_id: str, *, days: int = 90) -> pd.DataFrame:
        frame = _filter_frame(self._history(), {"store_id": store_id, "item_id": item_id})
        return frame.sort_values("date").tail(int(days))[["date", "sales", "sell_price"]].reset_index(drop=True)

    def get_forecast(self, store_id: str | None = None, item_id: str | None = None, *, horizon_days: int = 30) -> pd.DataFrame:
        if not 1 <= int(horizon_days) <= 30:
            raise ValueError("horizon_days must be between 1 and 30.")
        frame = _filter_frame(self._forecasts(), {"store_id": store_id, "item_id": item_id})
        return frame[frame["horizon_day"].le(int(horizon_days))].sort_values(["target_date", "store_id", "item_id"]).reset_index(drop=True)

    def get_recommendations(self, filters: dict[str, Any] | None = None) -> pd.DataFrame:
        return _filter_frame(self._recommendations(), filters).sort_values(
            ["priority_score", "store_id", "item_id"], ascending=[False, True, True]
        ).reset_index(drop=True)

    def get_snapshot(self, store_id: str, item_id: str) -> pd.DataFrame:
        return _filter_frame(self._snapshot(), {"store_id": store_id, "item_id": item_id}).reset_index(drop=True)

    def get_model_metadata(self) -> dict[str, Any]:
        return build_model_metadata(self.paths).iloc[0].to_dict()


class PublicDemoCsvDataSource(CsvDataSource):
    """Read-only CSV source backed by the committed compact deployment bundle."""

    label = "Compact public demo"

    def __init__(self, paths: PublicDemoPaths = PublicDemoPaths()):
        validate_public_demo_bundle(paths)
        self.paths = paths
        self._metadata = load_public_demo_metadata(paths)
        self._history_cache = None
        self._forecast_cache = None
        self._snapshot_cache = None
        self._recommendation_cache = None

    def _history(self) -> pd.DataFrame:
        if self._history_cache is None:
            self._history_cache = pd.read_csv(self.paths.history, parse_dates=["date"])
        return self._history_cache

    def get_overview_metrics(
        self, filters: dict[str, Any] | None = None
    ) -> dict[str, float | int]:
        metrics = super().get_overview_metrics(filters)
        metrics["bundled_history_observations"] = int(len(self._history()))
        metrics["bundled_history_days"] = int(self._history()["date"].nunique())
        if not filters:
            facts = self._metadata["portfolio_facts"]
            metrics.update(
                {
                    "sales_observations": int(facts["source_observations"]),
                    "products": int(facts["products"]),
                    "stores": len(facts["stores"]),
                    "series": int(facts["item_store_series"]),
                }
            )
        return metrics

    def get_model_metadata(self) -> dict[str, Any]:
        return dict(self._metadata["model"])


@dataclass(frozen=True)
class DataSourceResolution:
    source: PostgresDataSource | CsvDataSource | PublicDemoCsvDataSource
    notice: str
    fallback_used: bool


def _full_csv_artifacts_available(paths: ArtifactPaths) -> bool:
    return all(
        path.is_file()
        for path in (
            paths.history,
            paths.forecasts,
            paths.inventory_snapshot,
            paths.recommendations,
            paths.final_model_config,
            paths.inventory_policy,
            paths.deployment_model,
        )
    )


def _resolve_csv_source(
    paths: ArtifactPaths,
    public_paths: PublicDemoPaths,
) -> tuple[CsvDataSource | PublicDemoCsvDataSource, str]:
    if _full_csv_artifacts_available(paths):
        return CsvDataSource(paths), "using the complete local Stage 10 artifacts."
    return (
        PublicDemoCsvDataSource(public_paths),
        "using the committed compact public bundle: 120 recent history days plus unchanged "
        "saved forecasts and inventory decisions.",
    )


def resolve_data_source(
    settings: DatabaseSettings,
    paths: ArtifactPaths = ArtifactPaths(),
    public_paths: PublicDemoPaths = PublicDemoPaths(),
) -> DataSourceResolution:
    """Prefer an initialized PostgreSQL database and fall back visibly in auto mode."""

    if settings.data_mode == "csv":
        source, detail = _resolve_csv_source(paths, public_paths)
        return DataSourceResolution(
            source,
            f"CSV demo mode was explicitly selected; {detail}",
            True,
        )
    if settings.database_url:
        engine: Engine | None = None
        try:
            engine = create_database_engine(settings.database_url)
            verify_connection(engine)
            validate_loaded_database(engine)
            return DataSourceResolution(PostgresDataSource(engine), "Connected to the initialized SmartStock PostgreSQL database.", False)
        except Exception as exc:
            if engine is not None:
                engine.dispose()
            if settings.data_mode == "postgres":
                raise RuntimeError(
                    "PostgreSQL mode was requested but the connection or SmartStock schema is unavailable. "
                    "Run the database initializer and check DATABASE_URL."
                ) from exc
            notice = (
                f"PostgreSQL was unavailable ({exc.__class__.__name__}); using the read-only CSV fallback."
            )
            source, detail = _resolve_csv_source(paths, public_paths)
            return DataSourceResolution(source, f"{notice} It is {detail}", True)
    if settings.data_mode == "postgres":
        raise RuntimeError("PostgreSQL mode requires DATABASE_URL.")
    source, detail = _resolve_csv_source(paths, public_paths)
    return DataSourceResolution(
        source,
        f"DATABASE_URL is not configured; using the read-only CSV fallback, {detail}",
        True,
    )
