-- SmartStock Stage 11 PostgreSQL schema.
-- Re-running this file is safe: tables and indexes are created only when missing.

CREATE TABLE IF NOT EXISTS demand_history (
    date DATE NOT NULL,
    store_id VARCHAR(10) NOT NULL,
    item_id VARCHAR(40) NOT NULL,
    d VARCHAR(10) NOT NULL,
    dept_id VARCHAR(20) NOT NULL,
    cat_id VARCHAR(20) NOT NULL,
    state_id VARCHAR(5) NOT NULL,
    sales INTEGER NOT NULL CHECK (sales >= 0),
    sell_price DOUBLE PRECISION,
    weekday VARCHAR(12) NOT NULL,
    wday INTEGER NOT NULL,
    month INTEGER NOT NULL,
    year INTEGER NOT NULL,
    event_name_1 VARCHAR(80),
    event_type_1 VARCHAR(40),
    event_name_2 VARCHAR(80),
    event_type_2 VARCHAR(40),
    snap_active BOOLEAN NOT NULL,
    demand_band VARCHAR(12) NOT NULL,
    PRIMARY KEY (date, store_id, item_id)
);

CREATE INDEX IF NOT EXISTS ix_demand_history_date ON demand_history (date);
CREATE INDEX IF NOT EXISTS ix_demand_history_item_store_date ON demand_history (item_id, store_id, date);
CREATE INDEX IF NOT EXISTS ix_demand_history_store_date ON demand_history (store_id, date);
CREATE INDEX IF NOT EXISTS ix_demand_history_dept_date ON demand_history (dept_id, date);

CREATE TABLE IF NOT EXISTS forecasts (
    target_date DATE NOT NULL,
    store_id VARCHAR(10) NOT NULL,
    item_id VARCHAR(40) NOT NULL,
    dept_id VARCHAR(20) NOT NULL,
    demand_band VARCHAR(12) NOT NULL,
    forecast_origin DATE NOT NULL,
    horizon_day INTEGER NOT NULL CHECK (horizon_day BETWEEN 1 AND 30),
    forecast DOUBLE PRECISION NOT NULL CHECK (forecast >= 0),
    model_name VARCHAR(40) NOT NULL,
    model_version VARCHAR(30) NOT NULL,
    PRIMARY KEY (target_date, store_id, item_id)
);

CREATE INDEX IF NOT EXISTS ix_forecasts_item_store_target ON forecasts (item_id, store_id, target_date);
CREATE INDEX IF NOT EXISTS ix_forecasts_origin ON forecasts (forecast_origin);

CREATE TABLE IF NOT EXISTS inventory_snapshot (
    store_id VARCHAR(10) NOT NULL,
    item_id VARCHAR(40) NOT NULL,
    dept_id VARCHAR(20) NOT NULL,
    demand_band VARCHAR(12) NOT NULL,
    on_hand DOUBLE PRECISION NOT NULL CHECK (on_hand >= 0),
    on_order DOUBLE PRECISION NOT NULL CHECK (on_order >= 0),
    backorders DOUBLE PRECISION NOT NULL CHECK (backorders >= 0),
    lead_time_days INTEGER NOT NULL,
    service_level DOUBLE PRECISION NOT NULL,
    holding_cost_per_unit_per_day DOUBLE PRECISION NOT NULL,
    stockout_cost_per_unit DOUBLE PRECISION NOT NULL,
    fixed_order_cost DOUBLE PRECISION NOT NULL,
    inventory_source VARCHAR(40) NOT NULL,
    synthetic_profile_plan VARCHAR(20) NOT NULL,
    PRIMARY KEY (store_id, item_id)
);

CREATE TABLE IF NOT EXISTS inventory_recommendations (
    store_id VARCHAR(10) NOT NULL,
    item_id VARCHAR(40) NOT NULL,
    on_hand DOUBLE PRECISION,
    on_order DOUBLE PRECISION,
    backorders DOUBLE PRECISION,
    service_level DOUBLE PRECISION,
    holding_cost_per_unit_per_day DOUBLE PRECISION,
    stockout_cost_per_unit DOUBLE PRECISION,
    fixed_order_cost DOUBLE PRECISION,
    residual_rmse DOUBLE PRECISION,
    inventory_position DOUBLE PRECISION,
    forecast_1d DOUBLE PRECISION,
    forecast_7d DOUBLE PRECISION,
    forecast_30d DOUBLE PRECISION,
    mean_daily_forecast DOUBLE PRECISION,
    lead_time_demand DOUBLE PRECISION,
    planning_horizon_demand DOUBLE PRECISION,
    sigma_daily DOUBLE PRECISION,
    residual_mean_used DOUBLE PRECISION,
    safety_stock DOUBLE PRECISION,
    reorder_point DOUBLE PRECISION,
    target_stock_level DOUBLE PRECISION,
    days_of_supply DOUBLE PRECISION,
    stockout_risk DOUBLE PRECISION,
    stockout_risk_pct DOUBLE PRECISION,
    recommended_order_qty_raw DOUBLE PRECISION,
    priority_score DOUBLE PRECISION,
    estimated_cost_without_order DOUBLE PRECISION,
    estimated_cost_with_recommendation DOUBLE PRECISION,
    estimated_cost_with_service_order DOUBLE PRECISION,
    estimated_cost_savings DOUBLE PRECISION,
    expected_shortage_without_order DOUBLE PRECISION,
    expected_excess_without_order DOUBLE PRECISION,
    expected_shortage_with_recommendation DOUBLE PRECISION,
    expected_excess_with_recommendation DOUBLE PRECISION,
    lead_time_days INTEGER,
    calibration_count INTEGER,
    planning_horizon_days INTEGER,
    recommended_order_qty INTEGER CHECK (recommended_order_qty >= 0),
    cost_optimized_order_qty INTEGER,
    candidate_count INTEGER,
    candidate_min INTEGER,
    candidate_max INTEGER,
    dept_id VARCHAR(80),
    demand_band VARCHAR(80),
    calibration_level VARCHAR(80),
    intermittency_class VARCHAR(80),
    inventory_source VARCHAR(80),
    synthetic_profile_plan VARCHAR(80),
    days_of_supply_label VARCHAR(80),
    stock_status VARCHAR(80),
    priority_label VARCHAR(80),
    PRIMARY KEY (store_id, item_id)
);

CREATE INDEX IF NOT EXISTS ix_inventory_recommendations_status ON inventory_recommendations (stock_status);
CREATE INDEX IF NOT EXISTS ix_inventory_recommendations_priority ON inventory_recommendations (priority_label);

CREATE TABLE IF NOT EXISTS model_metadata (
    metadata_id VARCHAR(30) PRIMARY KEY,
    selected_model VARCHAR(40) NOT NULL,
    model_version VARCHAR(30) NOT NULL,
    forecast_origin DATE NOT NULL,
    deployment_train_through DATE NOT NULL,
    forecast_horizon_days INTEGER NOT NULL,
    model_artifact_path TEXT NOT NULL,
    model_artifact_sha256 VARCHAR(64) NOT NULL,
    inventory_policy_name VARCHAR(80) NOT NULL,
    inventory_policy_version VARCHAR(30) NOT NULL,
    inventory_policy_sha256 VARCHAR(64) NOT NULL,
    inventory_data_source VARCHAR(80) NOT NULL,
    loaded_at_utc TIMESTAMPTZ NOT NULL
);

COMMENT ON TABLE inventory_snapshot IS 'Synthetic demo inventory because M5 has no stock balances.';
COMMENT ON TABLE inventory_recommendations IS 'Stage 10 recommendations derived partly from synthetic demo inventory and costs.';
