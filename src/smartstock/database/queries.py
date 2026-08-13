"""Parameterized, reusable SmartStock application queries."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Iterable

import pandas as pd
from sqlalchemy import Engine, Select, and_, case, func, select

from smartstock.database.schema import TABLES


FILTER_COLUMNS = {"store_id", "dept_id", "item_id", "stock_status", "priority_label"}


def _conditions(table_name: str, filters: dict[str, Any] | None = None) -> list[Any]:
    table = TABLES[table_name]
    conditions: list[Any] = []
    for name, value in (filters or {}).items():
        if value in (None, "", "All"):
            continue
        if name not in FILTER_COLUMNS or name not in table.c:
            raise ValueError(f"Unsupported {table_name} filter: {name}")
        column = table.c[name]
        if isinstance(value, (list, tuple, set, frozenset)):
            conditions.append(column.in_(list(value)))
        else:
            conditions.append(column == value)
    return conditions


def _apply(statement: Select[Any], conditions: Iterable[Any]) -> Select[Any]:
    clauses = list(conditions)
    return statement.where(and_(*clauses)) if clauses else statement


def read_frame(engine: Engine, statement: Select[Any]) -> pd.DataFrame:
    with engine.connect() as connection:
        return pd.read_sql(statement, connection)


def get_filter_options(engine: Engine) -> dict[str, list[str]]:
    history = TABLES["demand_history"]
    result: dict[str, list[str]] = {}
    with engine.connect() as connection:
        for output_name, column in (
            ("stores", history.c.store_id),
            ("departments", history.c.dept_id),
            ("items", history.c.item_id),
        ):
            values = connection.execute(select(column).distinct().order_by(column)).scalars().all()
            result[output_name] = [str(value) for value in values]
    return result


def get_overview_metrics(engine: Engine, filters: dict[str, Any] | None = None) -> dict[str, float | int]:
    history = TABLES["demand_history"]
    forecasts = TABLES["forecasts"]
    recommendations = TABLES["inventory_recommendations"]
    history_conditions = _conditions("demand_history", filters)
    forecast_conditions = _conditions("forecasts", filters)
    recommendation_conditions = _conditions("inventory_recommendations", filters)
    with engine.connect() as connection:
        history_row = connection.execute(
            _apply(
                select(
                    func.sum(history.c.sales).label("total_units"),
                    func.max(history.c.date).label("latest_date"),
                    func.count(func.distinct(history.c.item_id)).label("products"),
                    func.count(func.distinct(history.c.store_id)).label("stores"),
                ),
                history_conditions,
            )
        ).mappings().one()
        pairs = len(connection.execute(
            _apply(select(history.c.store_id, history.c.item_id).distinct(), history_conditions)
        ).all())
        forecast_row = connection.execute(
            _apply(
                select(
                    func.sum(forecasts.c.forecast).label("forecast_30d"),
                    func.sum(case((forecasts.c.horizon_day <= 7, forecasts.c.forecast), else_=0)).label("forecast_7d"),
                ),
                forecast_conditions,
            )
        ).mappings().one()
        rec_frame = pd.read_sql(
            _apply(select(recommendations), recommendation_conditions), connection
        )
    latest = pd.Timestamp(history_row["latest_date"]) if history_row["latest_date"] else None
    recent_units = 0.0
    if latest is not None:
        recent_conditions = [*history_conditions, history.c.date >= (latest.date() - timedelta(days=29))]
        with engine.connect() as connection:
            recent_units = float(connection.execute(
                _apply(select(func.coalesce(func.sum(history.c.sales), 0)), recent_conditions)
            ).scalar_one())
    return _overview_dict(history_row, pairs, forecast_row, rec_frame, recent_units)


def _overview_dict(
    history_row: Any,
    pairs: int,
    forecast_row: Any,
    recommendations: pd.DataFrame,
    recent_units: float,
) -> dict[str, float | int]:
    if recommendations.empty:
        recommendation_values = {
            "reorder_items": 0, "critical_stockout_items": 0, "recommended_units": 0,
            "shortage_before": 0.0, "shortage_after": 0.0, "estimated_savings": 0.0,
        }
    else:
        recommendation_values = {
            "reorder_items": int(recommendations["recommended_order_qty"].gt(0).sum()),
            "critical_stockout_items": int(recommendations["stock_status"].isin(["CRITICAL", "STOCKOUT"]).sum()),
            "recommended_units": int(recommendations["recommended_order_qty"].sum()),
            "shortage_before": float(recommendations["expected_shortage_without_order"].sum()),
            "shortage_after": float(recommendations["expected_shortage_with_recommendation"].sum()),
            "estimated_savings": float(recommendations["estimated_cost_savings"].sum()),
        }
    return {
        "products": int(history_row["products"] or 0),
        "stores": int(history_row["stores"] or 0),
        "series": int(pairs),
        "historical_units": float(history_row["total_units"] or 0),
        "recent_30d_units": recent_units,
        "forecast_7d": float(forecast_row["forecast_7d"] or 0),
        "forecast_30d": float(forecast_row["forecast_30d"] or 0),
        **recommendation_values,
    }


def get_sales_trend(
    engine: Engine,
    *,
    filters: dict[str, Any] | None = None,
    start_date: Any | None = None,
    end_date: Any | None = None,
    frequency: str = "weekly",
) -> pd.DataFrame:
    history = TABLES["demand_history"]
    conditions = _conditions("demand_history", filters)
    if start_date is not None:
        conditions.append(history.c.date >= pd.Timestamp(start_date).date())
    if end_date is not None:
        conditions.append(history.c.date <= pd.Timestamp(end_date).date())
    if frequency not in {"daily", "weekly", "monthly"}:
        raise ValueError("frequency must be daily, weekly, or monthly.")
    dialect = engine.dialect.name
    if frequency == "daily":
        period = history.c.date
    elif dialect == "postgresql":
        period = func.date_trunc("week" if frequency == "weekly" else "month", history.c.date)
    elif frequency == "weekly":
        period = func.strftime("%Y-%W", history.c.date)
    else:
        period = func.strftime("%Y-%m", history.c.date)
    statement = _apply(
        select(period.label("period"), func.sum(history.c.sales).label("sales")).group_by(period).order_by(period),
        conditions,
    )
    return read_frame(engine, statement)


def get_sales_breakdown(
    engine: Engine,
    group_by: str,
    *,
    filters: dict[str, Any] | None = None,
    start_date: Any | None = None,
    end_date: Any | None = None,
) -> pd.DataFrame:
    if group_by not in {"store_id", "dept_id", "weekday"}:
        raise ValueError("Unsupported sales breakdown.")
    history = TABLES["demand_history"]
    conditions = _conditions("demand_history", filters)
    if start_date is not None:
        conditions.append(history.c.date >= pd.Timestamp(start_date).date())
    if end_date is not None:
        conditions.append(history.c.date <= pd.Timestamp(end_date).date())
    column = history.c[group_by]
    return read_frame(
        engine,
        _apply(select(column.label("group"), func.sum(history.c.sales).label("sales")).group_by(column).order_by(column), conditions),
    )


def get_demand_profile(
    engine: Engine,
    *,
    filters: dict[str, Any] | None = None,
    start_date: Any | None = None,
    end_date: Any | None = None,
) -> dict[str, float | int]:
    history = TABLES["demand_history"]
    conditions = _conditions("demand_history", filters)
    if start_date is not None:
        conditions.append(history.c.date >= pd.Timestamp(start_date).date())
    if end_date is not None:
        conditions.append(history.c.date <= pd.Timestamp(end_date).date())
    statement = _apply(
        select(
            func.count().label("observations"),
            func.sum(case((history.c.sales == 0, 1), else_=0)).label("zero_observations"),
            func.avg(history.c.sales).label("mean_daily_units"),
            func.max(history.c.sales).label("maximum_daily_units"),
        ),
        conditions,
    )
    with engine.connect() as connection:
        row = connection.execute(statement).mappings().one()
    observations = int(row["observations"] or 0)
    zeros = int(row["zero_observations"] or 0)
    return {
        "observations": observations,
        "zero_observations": zeros,
        "zero_rate": zeros / observations if observations else float("nan"),
        "mean_daily_units": float(row["mean_daily_units"] or 0),
        "maximum_daily_units": float(row["maximum_daily_units"] or 0),
    }


def get_recent_history(
    engine: Engine, store_id: str, item_id: str, *, days: int = 90
) -> pd.DataFrame:
    history = TABLES["demand_history"]
    statement = (
        select(history.c.date, history.c.sales, history.c.sell_price)
        .where(history.c.store_id == store_id, history.c.item_id == item_id)
        .order_by(history.c.date.desc())
        .limit(int(days))
    )
    return read_frame(engine, statement).sort_values("date").reset_index(drop=True)


def get_forecast(
    engine: Engine, store_id: str | None = None, item_id: str | None = None, *, horizon_days: int = 30
) -> pd.DataFrame:
    if not 1 <= int(horizon_days) <= 30:
        raise ValueError("horizon_days must be between 1 and 30.")
    table = TABLES["forecasts"]
    filters = {"store_id": store_id, "item_id": item_id}
    conditions = [*_conditions("forecasts", filters), table.c.horizon_day <= int(horizon_days)]
    return read_frame(engine, _apply(select(table).order_by(table.c.target_date, table.c.store_id, table.c.item_id), conditions))


def get_inventory_recommendations(
    engine: Engine, filters: dict[str, Any] | None = None
) -> pd.DataFrame:
    table = TABLES["inventory_recommendations"]
    statement = _apply(select(table), _conditions("inventory_recommendations", filters)).order_by(
        table.c.priority_score.desc(), table.c.store_id, table.c.item_id
    )
    return read_frame(engine, statement)


def get_inventory_snapshot(engine: Engine, store_id: str, item_id: str) -> pd.DataFrame:
    table = TABLES["inventory_snapshot"]
    return read_frame(engine, select(table).where(table.c.store_id == store_id, table.c.item_id == item_id))


def get_model_metadata(engine: Engine) -> dict[str, Any]:
    frame = read_frame(engine, select(TABLES["model_metadata"]).limit(1))
    return {} if frame.empty else frame.iloc[0].to_dict()
