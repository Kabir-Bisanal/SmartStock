"""SmartStock Stage 11 PostgreSQL/CSV Streamlit application."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.app_utils import INVENTORY_DISCLAIMER, apply_inventory_scenario, forecast_chart_data, recommendation_table
from smartstock.database.connection import DatabaseSettings
from smartstock.database.data_source import resolve_data_source
from smartstock.inventory.policy import load_policy


st.set_page_config(page_title="SmartStock", page_icon="📦", layout="wide")


@st.cache_resource(show_spinner="Connecting to SmartStock data …")
def get_resolution():
    return resolve_data_source(DatabaseSettings.from_environment())


@st.cache_resource
def get_policy() -> dict[str, Any]:
    return load_policy(PROJECT_ROOT / "config" / "v1_inventory_policy.json")


@st.cache_data(ttl=300, show_spinner=False)
def cached_filter_options(_source: Any) -> dict[str, list[str]]:
    return _source.get_filter_options()


@st.cache_data(ttl=300, show_spinner=False)
def cached_sales_trend(
    _source: Any,
    filters: tuple[tuple[str, str], ...],
    start_date: str,
    end_date: str,
    frequency: str,
) -> pd.DataFrame:
    return _source.get_sales_trend(
        filters=dict(filters), start_date=start_date, end_date=end_date, frequency=frequency
    )


@st.cache_data(ttl=300, show_spinner=False)
def cached_sales_breakdown(
    _source: Any,
    group_by: str,
    filters: tuple[tuple[str, str], ...],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    return _source.get_sales_breakdown(
        group_by, filters=dict(filters), start_date=start_date, end_date=end_date
    )


def money(value: float) -> str:
    return f"{value:,.2f}"


def global_filters(source: Any, *, key_prefix: str = "global") -> dict[str, str]:
    options = cached_filter_options(source)
    col1, col2, col3 = st.columns(3)
    store = col1.selectbox("Store", ["All", *options["stores"]], key=f"{key_prefix}_store")
    department = col2.selectbox("Department", ["All", *options["departments"]], key=f"{key_prefix}_dept")
    eligible_items = options["items"]
    product = col3.selectbox("Product", ["All", *eligible_items], key=f"{key_prefix}_item")
    return {"store_id": store, "dept_id": department, "item_id": product}


def show_disclaimer() -> None:
    st.info(INVENTORY_DISCLAIMER, icon="ℹ️")


def render_overview(source: Any) -> None:
    st.header("Overview")
    filters = global_filters(source, key_prefix="overview")
    metrics = source.get_overview_metrics(filters)
    row1 = st.columns(4)
    row1[0].metric("Products", f"{metrics['products']:,}")
    row1[1].metric("Stores", f"{metrics['stores']:,}")
    row1[2].metric("Item-store series", f"{metrics['series']:,}")
    row1[3].metric("Recent 30-day units", f"{metrics['recent_30d_units']:,.0f}")
    row2 = st.columns(4)
    row2[0].metric("Forecast: next 7 days", f"{metrics['forecast_7d']:,.1f}")
    row2[1].metric("Forecast: next 30 days", f"{metrics['forecast_30d']:,.1f}")
    row2[2].metric("Series requiring reorder", f"{metrics['reorder_items']:,}")
    row2[3].metric("Critical / stockout", f"{metrics['critical_stockout_items']:,}")
    row3 = st.columns(4)
    row3[0].metric("Recommended units", f"{metrics['recommended_units']:,.0f}")
    row3[1].metric("Shortage before", f"{metrics['shortage_before']:,.1f}")
    row3[2].metric("Shortage after", f"{metrics['shortage_after']:,.1f}")
    row3[3].metric("Estimated savings (demo)", money(float(metrics["estimated_savings"])))
    show_disclaimer()

    recs = source.get_recommendations(filters)
    forecasts = source.get_forecast(
        None if filters["store_id"] == "All" else filters["store_id"],
        None if filters["item_id"] == "All" else filters["item_id"],
        horizon_days=30,
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Stock status distribution")
        if recs.empty:
            st.warning("No inventory recommendations match these filters.")
        else:
            st.bar_chart(recs["stock_status"].value_counts())
        st.subheader("Recommended units by store")
        if not recs.empty:
            st.bar_chart(recs.groupby("store_id")["recommended_order_qty"].sum())
    with right:
        st.subheader("Priority distribution")
        if not recs.empty:
            st.bar_chart(recs["priority_label"].value_counts())
        st.subheader("30-day forecast by store")
        if forecasts.empty:
            st.warning("No forecasts match these filters.")
        else:
            st.bar_chart(forecasts.groupby("store_id")["forecast"].sum())


def render_sales_analytics(source: Any) -> None:
    st.header("Sales Analytics")
    st.caption("Real M5 historical sales; zero-demand days remain valid observations.")
    filters = global_filters(source, key_prefix="sales")
    start_default = pd.Timestamp("2015-05-23").date()
    end_default = pd.Timestamp("2016-05-22").date()
    col1, col2, col3 = st.columns(3)
    start_date = col1.date_input("Start date", value=start_default, min_value=pd.Timestamp("2011-01-29").date(), max_value=end_default)
    end_date = col2.date_input("End date", value=end_default, min_value=pd.Timestamp("2011-01-29").date(), max_value=end_default)
    frequency = col3.selectbox("Trend frequency", ["daily", "weekly", "monthly"], index=1)
    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        return
    kwargs = {"filters": filters, "start_date": start_date, "end_date": end_date}
    frozen_filters = tuple(sorted(filters.items()))
    trend = cached_sales_trend(
        source, frozen_filters, start_date.isoformat(), end_date.isoformat(), frequency
    )
    profile = source.get_demand_profile(**kwargs)
    metrics = st.columns(4)
    metrics[0].metric("Observations", f"{profile['observations']:,}")
    metrics[1].metric("Zero-demand share", f"{profile['zero_rate']:.1%}")
    metrics[2].metric("Mean units / observation", f"{profile['mean_daily_units']:.2f}")
    metrics[3].metric("Maximum units", f"{profile['maximum_daily_units']:,.0f}")
    if trend.empty:
        st.warning("No historical sales match these filters.")
        return
    st.subheader(f"{frequency.title()} total sales")
    st.line_chart(trend.set_index("period")["sales"])
    chart_cols = st.columns(3)
    for column, group_by, title in zip(chart_cols, ("store_id", "dept_id", "weekday"), ("Store comparison", "Department contribution", "Weekday behavior"), strict=True):
        with column:
            st.subheader(title)
            breakdown = cached_sales_breakdown(
                source, group_by, frozen_filters, start_date.isoformat(), end_date.isoformat()
            )
            st.bar_chart(breakdown.set_index("group")["sales"])


def render_forecasting(source: Any) -> None:
    st.header("Demand Forecasting")
    options = cached_filter_options(source)
    col1, col2, col3 = st.columns(3)
    store = col1.selectbox("Store", options["stores"], key="forecast_store")
    item = col2.selectbox("Product", options["items"], key="forecast_item")
    horizon = col3.selectbox("Forecast horizon", [1, 7, 30], index=2, format_func=lambda x: f"{x} day" if x == 1 else f"{x} days")
    history = source.get_recent_history(store, item, days=90)
    forecast = source.get_forecast(store, item, horizon_days=horizon)
    if history.empty or forecast.empty:
        st.warning("History or production forecast is unavailable for this item-store series.")
        return
    stats = st.columns(4)
    stats[0].metric("Day +1", f"{forecast.loc[forecast['horizon_day'].eq(1), 'forecast'].sum():.2f}")
    stats[1].metric("Next 7 days", f"{forecast.loc[forecast['horizon_day'].le(7), 'forecast'].sum():.2f}")
    stats[2].metric("Selected horizon total", f"{forecast['forecast'].sum():.2f}")
    stats[3].metric("Mean daily forecast", f"{forecast['forecast'].mean():.2f}")
    st.subheader("Recent actual demand and future forecast")
    chart = forecast_chart_data(history, forecast).set_index("date")
    st.line_chart(chart[["Actual", "Forecast"]])
    st.caption(
        "Selected production method: **28-Day Historical Mean (`mean_28`)**. More complex Ridge and boosting models were evaluated, but mean_28 gave the strongest balanced validation performance. The future portion contains forecasts only; no actual sales exist after 2016-05-22 in this dataset."
    )


def inventory_filters(source: Any, prefix: str) -> dict[str, Any]:
    options = cached_filter_options(source)
    recs = source.get_recommendations()
    cols = st.columns(4)
    return {
        "store_id": cols[0].selectbox("Store", ["All", *options["stores"]], key=f"{prefix}_store"),
        "dept_id": cols[1].selectbox("Department", ["All", *options["departments"]], key=f"{prefix}_dept"),
        "stock_status": cols[2].selectbox("Stock status", ["All", *sorted(recs["stock_status"].dropna().unique())], key=f"{prefix}_status"),
        "priority_label": cols[3].selectbox("Priority", ["All", *sorted(recs["priority_label"].dropna().unique())], key=f"{prefix}_priority"),
    }


def render_inventory_health(source: Any) -> None:
    st.header("Inventory Health")
    show_disclaimer()
    filters = inventory_filters(source, "health")
    recs = source.get_recommendations(filters)
    if recs.empty:
        st.warning("No inventory rows match these filters.")
        return
    cols = st.columns(4)
    cols[0].metric("Selected series", f"{len(recs):,}")
    cols[1].metric("Median days of supply", f"{recs['days_of_supply'].median():.1f}")
    cols[2].metric("Mean stockout risk", f"{recs['stockout_risk_pct'].mean():.1f}%")
    cols[3].metric("Inventory position", f"{recs['inventory_position'].sum():,.0f}")
    left, right = st.columns(2)
    left.subheader("Stock status counts")
    left.bar_chart(recs["stock_status"].value_counts())
    right.subheader("Stockout risk distribution")
    right.bar_chart(pd.cut(recs["stockout_risk_pct"], bins=[-0.1, 10, 25, 50, 75, 100], labels=["0–10%", "10–25%", "25–50%", "50–75%", "75–100%"]).value_counts(sort=False))
    st.subheader("Inventory detail")
    st.dataframe(recommendation_table(recs), width="stretch", hide_index=True)


def render_reorders(source: Any) -> None:
    st.header("Reorder Recommendations")
    show_disclaimer()
    filters = inventory_filters(source, "reorder")
    recs = source.get_recommendations(filters)
    if recs.empty:
        st.warning("No reorder recommendations match these filters.")
        return
    cols = st.columns(4)
    cols[0].metric("Selected recommendations", f"{len(recs):,}")
    cols[1].metric("Recommended units", f"{recs['recommended_order_qty'].sum():,.0f}")
    shortage_reduction = recs["expected_shortage_without_order"].sum() - recs["expected_shortage_with_recommendation"].sum()
    cols[2].metric("Estimated shortage reduction", f"{shortage_reduction:,.1f}")
    cols[3].metric("Estimated savings (demo)", money(float(recs["estimated_cost_savings"].sum())))
    st.dataframe(recommendation_table(recs), width="stretch", hide_index=True)


def render_scenario_planner(source: Any) -> None:
    st.header("What-if Scenario")
    show_disclaimer()
    options = cached_filter_options(source)
    col1, col2 = st.columns(2)
    store = col1.selectbox("Store", options["stores"], key="scenario_store")
    item = col2.selectbox("Product", options["items"], key="scenario_item")
    snapshot_frame = source.get_snapshot(store, item)
    rec_frame = source.get_recommendations({"store_id": store, "item_id": item})
    forecast = source.get_forecast(store, item, horizon_days=30)
    if snapshot_frame.empty or rec_frame.empty or forecast.empty:
        st.warning("Scenario inputs are incomplete for this product-store selection.")
        return
    snapshot = snapshot_frame.iloc[0].to_dict()
    recommendation = rec_frame.iloc[0].to_dict()
    policy = get_policy()
    c1, c2, c3 = st.columns(3)
    on_hand = c1.number_input("On hand", min_value=0.0, max_value=10_000.0, value=float(snapshot["on_hand"]), step=1.0)
    on_order = c2.number_input("On order", min_value=0.0, max_value=10_000.0, value=float(snapshot["on_order"]), step=1.0)
    backorders = c3.number_input("Backorders", min_value=0.0, max_value=10_000.0, value=float(snapshot["backorders"]), step=1.0)
    c4, c5, c6 = st.columns(3)
    lead_time = c4.number_input("Lead time (days)", min_value=1, max_value=30, value=int(snapshot["lead_time_days"]), step=1)
    max_review = 30 - int(lead_time)
    review_period = c5.number_input(
        "Review period (days)",
        min_value=0,
        max_value=max_review,
        value=min(int(policy["review_period_days"]), max_review),
        step=1,
        key=f"scenario_review_period_{int(lead_time)}",
        help="Lead time plus review period is limited to the available 30-day forecast.",
    )
    service_level = c6.select_slider("Service level", options=[0.90, 0.95, 0.975, 0.99], value=float(snapshot["service_level"]))
    c7, c8, c9 = st.columns(3)
    holding_cost = c7.number_input("Holding cost / unit / day", min_value=0.0, max_value=100.0, value=float(snapshot["holding_cost_per_unit_per_day"]), step=0.01, format="%.3f")
    stockout_cost = c8.number_input("Stockout cost / unit", min_value=0.0, max_value=1_000.0, value=float(snapshot["stockout_cost_per_unit"]), step=0.5)
    fixed_cost = c9.number_input("Fixed order cost", min_value=0.0, max_value=10_000.0, value=float(snapshot["fixed_order_cost"]), step=1.0)
    overrides = {
        "on_hand": on_hand, "on_order": on_order, "backorders": backorders,
        "lead_time_days": lead_time, "service_level": service_level,
        "review_period_days": review_period,
        "holding_cost_per_unit_per_day": holding_cost,
        "stockout_cost_per_unit": stockout_cost, "fixed_order_cost": fixed_cost,
    }
    calibration = {
        "sigma_daily": recommendation["sigma_daily"],
        "residual_mean_used": recommendation["residual_mean_used"],
    }
    result = apply_inventory_scenario(forecast["forecast"], snapshot, calibration, policy, overrides)
    result_cols = st.columns(5)
    result_cols[0].metric("Inventory position", f"{result['inventory_position']:.1f}")
    result_cols[1].metric("Lead-time demand", f"{result['lead_time_demand']:.1f}")
    result_cols[2].metric("Safety stock", f"{result['safety_stock']:.1f}")
    result_cols[3].metric("Reorder point", f"{result['reorder_point']:.1f}")
    result_cols[4].metric("Target stock", f"{result['target_stock_level']:.1f}")
    result_cols2 = st.columns(5)
    result_cols2[0].metric("Days of supply", result["days_of_supply_label"])
    result_cols2[1].metric("Stockout risk", f"{result['stockout_risk_pct']:.1f}%")
    result_cols2[2].metric("Status / priority", f"{result['stock_status']} / {result['priority_label']}")
    result_cols2[3].metric("Service-level order", f"{result['recommended_order_qty']:,}")
    result_cols2[4].metric("Cost-optimized order", f"{result['cost_optimized_order_qty']:,}")
    cost_cols = st.columns(2)
    cost_cols[0].metric("Estimated cost before (demo)", money(result["estimated_cost_without_order"]))
    cost_cols[1].metric("Estimated cost after (demo)", money(result["estimated_cost_with_recommendation"]))
    st.caption("Higher service levels increase safety stock. Longer lead times increase demand exposure and uncertainty. More on-order stock increases inventory position. Higher stockout cost can make a larger cost-optimized order worthwhile. This scenario is in memory only and never overwrites saved recommendations.")


def render_methodology(source: Any, resolution: Any) -> None:
    st.header("Methodology / About")
    metadata = source.get_model_metadata()
    st.markdown(
        """
**Dataset:** Walmart M5 Forecasting — Accuracy data. SmartStock V1 uses 100 FOODS items across CA_1, TX_2, and WI_3, producing 300 item-store series. The original wide sales table was filtered before conversion to a daily long table.

**Forecasting:** Zero, persistence, seasonal naive, 28-day mean, Croston-SBA, Global Ridge, HistGradientBoosting, and XGBoost were compared using chronological rolling validation and leakage-safe recursive forecasting. The selected production method is **28-Day Historical Mean** because it offered the strongest balanced validation performance.

**Inventory:** Validation-calibrated uncertainty feeds safety stock, reorder point, target stock, stockout risk, service-level orders, priority, and cost-aware scenarios. M5 has no inventory balances, so all stock, lead-time, service, and cost inputs are synthetic demo assumptions.

**Limitations:** Sales can be censored by historical stockouts; future prices are not modeled; inventory and cost outputs are educational scenarios rather than Walmart operational facts.
"""
    )
    st.subheader("Reproducibility metadata")
    st.json(
        {
            "forecast_model": metadata.get("selected_model", "mean_28"),
            "forecast_training_through": str(metadata.get("deployment_train_through", "2016-05-22"))[:10],
            "forecast_horizon_days": int(metadata.get("forecast_horizon_days", 30)),
            "inventory_policy_version": metadata.get("inventory_policy_version", "v1"),
            "data_source": resolution.source.label,
            "inventory_data": "synthetic_demo_not_walmart",
        }
    )


def main() -> None:
    st.title("SmartStock")
    st.caption("Demand Forecasting & Inventory Optimization System")
    try:
        resolution = get_resolution()
    except Exception as exc:
        st.error(f"SmartStock could not start: {exc}")
        st.info("Generate Stage 10 artifacts or configure and initialize PostgreSQL, then reload the app.")
        st.stop()
    st.sidebar.success(f"Data source: {resolution.source.label}")
    if resolution.fallback_used:
        st.sidebar.warning(resolution.notice)
    else:
        st.sidebar.caption(resolution.notice)
    section = st.sidebar.radio(
        "Navigate",
        ["Overview", "Sales Analytics", "Demand Forecasting", "Inventory Health", "Reorder Recommendations", "Scenario Planner", "Methodology / About"],
    )
    renderers = {
        "Overview": lambda: render_overview(resolution.source),
        "Sales Analytics": lambda: render_sales_analytics(resolution.source),
        "Demand Forecasting": lambda: render_forecasting(resolution.source),
        "Inventory Health": lambda: render_inventory_health(resolution.source),
        "Reorder Recommendations": lambda: render_reorders(resolution.source),
        "Scenario Planner": lambda: render_scenario_planner(resolution.source),
        "Methodology / About": lambda: render_methodology(resolution.source, resolution),
    }
    renderers[section]()


if __name__ == "__main__":
    main()
