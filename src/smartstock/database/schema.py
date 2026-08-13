"""SQLAlchemy Core schema shared by the loader, queries, and SQLite tests."""

from __future__ import annotations

from sqlalchemy import Boolean, CheckConstraint, Column, Date, DateTime, Float, Index, Integer, MetaData, String, Table, Text
from sqlalchemy.engine import Engine


metadata = MetaData()

demand_history = Table(
    "demand_history", metadata,
    Column("date", Date, primary_key=True, nullable=False),
    Column("store_id", String(10), primary_key=True, nullable=False),
    Column("item_id", String(40), primary_key=True, nullable=False),
    Column("d", String(10), nullable=False),
    Column("dept_id", String(20), nullable=False),
    Column("cat_id", String(20), nullable=False),
    Column("state_id", String(5), nullable=False),
    Column("sales", Integer, nullable=False),
    Column("sell_price", Float),
    Column("weekday", String(12), nullable=False),
    Column("wday", Integer, nullable=False),
    Column("month", Integer, nullable=False),
    Column("year", Integer, nullable=False),
    Column("event_name_1", String(80)),
    Column("event_type_1", String(40)),
    Column("event_name_2", String(80)),
    Column("event_type_2", String(40)),
    Column("snap_active", Boolean, nullable=False),
    Column("demand_band", String(12), nullable=False),
    CheckConstraint("sales >= 0", name="ck_demand_history_sales_nonnegative"),
)
Index("ix_demand_history_date", demand_history.c.date)
Index("ix_demand_history_item_store_date", demand_history.c.item_id, demand_history.c.store_id, demand_history.c.date)
Index("ix_demand_history_store_date", demand_history.c.store_id, demand_history.c.date)
Index("ix_demand_history_dept_date", demand_history.c.dept_id, demand_history.c.date)

forecasts = Table(
    "forecasts", metadata,
    Column("target_date", Date, primary_key=True, nullable=False),
    Column("store_id", String(10), primary_key=True, nullable=False),
    Column("item_id", String(40), primary_key=True, nullable=False),
    Column("dept_id", String(20), nullable=False),
    Column("demand_band", String(12), nullable=False),
    Column("forecast_origin", Date, nullable=False),
    Column("horizon_day", Integer, nullable=False),
    Column("forecast", Float, nullable=False),
    Column("model_name", String(40), nullable=False),
    Column("model_version", String(30), nullable=False),
    CheckConstraint("forecast >= 0", name="ck_forecasts_nonnegative"),
    CheckConstraint("horizon_day BETWEEN 1 AND 30", name="ck_forecasts_horizon"),
)
Index("ix_forecasts_item_store_target", forecasts.c.item_id, forecasts.c.store_id, forecasts.c.target_date)
Index("ix_forecasts_origin", forecasts.c.forecast_origin)

inventory_snapshot = Table(
    "inventory_snapshot", metadata,
    Column("store_id", String(10), primary_key=True, nullable=False),
    Column("item_id", String(40), primary_key=True, nullable=False),
    Column("dept_id", String(20), nullable=False),
    Column("demand_band", String(12), nullable=False),
    Column("on_hand", Float, nullable=False),
    Column("on_order", Float, nullable=False),
    Column("backorders", Float, nullable=False),
    Column("lead_time_days", Integer, nullable=False),
    Column("service_level", Float, nullable=False),
    Column("holding_cost_per_unit_per_day", Float, nullable=False),
    Column("stockout_cost_per_unit", Float, nullable=False),
    Column("fixed_order_cost", Float, nullable=False),
    Column("inventory_source", String(40), nullable=False),
    Column("synthetic_profile_plan", String(20), nullable=False),
    CheckConstraint("on_hand >= 0 AND on_order >= 0 AND backorders >= 0", name="ck_snapshot_nonnegative"),
)

_rec_floats = [
    "on_hand", "on_order", "backorders", "service_level", "holding_cost_per_unit_per_day",
    "stockout_cost_per_unit", "fixed_order_cost", "residual_rmse", "inventory_position",
    "forecast_1d", "forecast_7d", "forecast_30d", "mean_daily_forecast", "lead_time_demand",
    "planning_horizon_demand", "sigma_daily", "residual_mean_used", "safety_stock",
    "reorder_point", "target_stock_level", "days_of_supply", "stockout_risk", "stockout_risk_pct",
    "recommended_order_qty_raw", "priority_score", "estimated_cost_without_order",
    "estimated_cost_with_recommendation", "estimated_cost_with_service_order", "estimated_cost_savings",
    "expected_shortage_without_order", "expected_excess_without_order",
    "expected_shortage_with_recommendation", "expected_excess_with_recommendation",
]
_rec_integers = [
    "lead_time_days", "calibration_count", "planning_horizon_days", "recommended_order_qty",
    "cost_optimized_order_qty", "candidate_count", "candidate_min", "candidate_max",
]
_rec_text = [
    "dept_id", "demand_band", "calibration_level", "intermittency_class", "inventory_source",
    "synthetic_profile_plan", "days_of_supply_label", "stock_status", "priority_label",
]

inventory_recommendations = Table(
    "inventory_recommendations", metadata,
    Column("store_id", String(10), primary_key=True, nullable=False),
    Column("item_id", String(40), primary_key=True, nullable=False),
    *[Column(name, Float) for name in _rec_floats],
    *[Column(name, Integer) for name in _rec_integers],
    *[Column(name, String(80)) for name in _rec_text],
    CheckConstraint("recommended_order_qty >= 0", name="ck_recommendation_nonnegative"),
)
Index("ix_inventory_recommendations_status", inventory_recommendations.c.stock_status)
Index("ix_inventory_recommendations_priority", inventory_recommendations.c.priority_label)

model_metadata = Table(
    "model_metadata", metadata,
    Column("metadata_id", String(30), primary_key=True),
    Column("selected_model", String(40), nullable=False),
    Column("model_version", String(30), nullable=False),
    Column("forecast_origin", Date, nullable=False),
    Column("deployment_train_through", Date, nullable=False),
    Column("forecast_horizon_days", Integer, nullable=False),
    Column("model_artifact_path", Text, nullable=False),
    Column("model_artifact_sha256", String(64), nullable=False),
    Column("inventory_policy_name", String(80), nullable=False),
    Column("inventory_policy_version", String(30), nullable=False),
    Column("inventory_policy_sha256", String(64), nullable=False),
    Column("inventory_data_source", String(80), nullable=False),
    Column("loaded_at_utc", DateTime(timezone=True), nullable=False),
)

TABLES = {table.name: table for table in (demand_history, forecasts, inventory_snapshot, inventory_recommendations, model_metadata)}


def create_schema(engine: Engine) -> None:
    """Create missing tables and indexes without dropping existing data."""

    metadata.create_all(engine)
