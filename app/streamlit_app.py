"""SmartStock portfolio dashboard for demand and inventory decisions."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.app_utils import (  # noqa: E402
    INVENTORY_DISCLAIMER,
    add_order_action,
    apply_inventory_scenario,
    dataframe_to_csv_bytes,
    filter_recommendations,
    forecast_chart_data,
    recommendation_table,
    scenario_explanations,
    shortage_reduction_pct,
)
from smartstock.database.connection import DatabaseSettings  # noqa: E402
from smartstock.database.data_source import resolve_data_source  # noqa: E402
from smartstock.inventory.policy import load_policy  # noqa: E402


st.set_page_config(
    page_title="SmartStock",
    page_icon=":material/inventory_2:",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOCKED_TEST_METRICS = {
    "daily_mae": 1.382,
    "daily_wape": 0.6120,
    "aggregate_wape": 0.2214,
}


@st.cache_resource(show_spinner="Connecting to SmartStock data …")
def get_resolution():
    return resolve_data_source(DatabaseSettings.from_environment())


@st.cache_resource
def get_policy() -> dict[str, Any]:
    return load_policy(PROJECT_ROOT / "config" / "v1_inventory_policy.json")


@st.cache_data(ttl=300, show_spinner=False)
def cached_filter_options(_source: Any) -> dict[str, list[str]]:
    return _source.get_filter_options()


def money(value: float) -> str:
    return f"{value:,.2f}"


def metric_row(metrics: list[tuple[str, str, str | None]]) -> None:
    """Render a compact row of two to four bordered KPI cards."""

    columns = st.columns(len(metrics))
    for column, (label, value, help_text) in zip(columns, metrics, strict=True):
        column.metric(label, value, help=help_text, border=True)


def recommendations_for_selection(source: Any, department: str, store: str) -> pd.DataFrame:
    filters = {
        "dept_id": None if department == "All" else department,
        "store_id": None if store == "All" else store,
    }
    return source.get_recommendations(filters)


def sku_selector(
    source: Any,
    *,
    prefix: str,
    prioritize_reorders: bool = False,
) -> tuple[str, str, str]:
    """Render linked department, store, and anonymized SKU selectors."""

    options = cached_filter_options(source)
    columns = st.columns(3)
    department = columns[0].selectbox(
        "Department", options["departments"], key=f"{prefix}_department"
    )
    store = columns[1].selectbox("Store", options["stores"], key=f"{prefix}_store")
    eligible = recommendations_for_selection(source, department, store)
    if prioritize_reorders:
        eligible = eligible.sort_values(
            ["priority_score", "recommended_order_qty"], ascending=[False, False]
        )
        item_options = eligible["item_id"].astype(str).drop_duplicates().tolist()
    else:
        item_options = sorted(eligible["item_id"].astype(str).unique().tolist())
    item = columns[2].selectbox(
        "Anonymized SKU", item_options, key=f"{prefix}_item"
    )
    return department, store, item


def render_overview(source: Any) -> None:
    st.header("Overview")
    st.markdown(
        "SmartStock answers a practical retail question: **which products should be reordered, "
        "how many units should be ordered, and why?**"
    )
    st.caption(
        "Portfolio case study using Walmart M5 historical demand. Product IDs are anonymized; "
        "inventory balances, lead times, service targets, and costs are synthetic demo inputs."
    )

    metrics = source.get_overview_metrics()
    recommendations = add_order_action(source.get_recommendations())
    metric_row(
        [
            ("Daily observations", f"{metrics['sales_observations']:,}", "Historical item-store sales observations analyzed."),
            ("Item-store decisions", f"{metrics['series']:,}", "100 products across three stores."),
            ("Need reorder", f"{metrics['reorder_items']:,}", "Saved decisions with a positive order quantity."),
            ("Forecast horizon", "30 days", "Daily demand forecasts used by the inventory engine."),
        ]
    )
    bundled_rows = metrics.get("bundled_history_observations")
    if bundled_rows and bundled_rows < metrics["sales_observations"]:
        st.caption(
            f"The project analyzed {metrics['sales_observations']:,} daily observations. "
            f"This public deployment bundles the most recent {metrics['bundled_history_days']:,} days "
            f"({bundled_rows:,} rows) for lightweight visualization."
        )

    st.subheader("How SmartStock turns history into action")
    steps = [
        (":material/query_stats:", "Historical demand", "Learn demand level, timing, and uncertainty."),
        (":material/monitoring:", "30-day forecast", "Estimate daily units for each product and store."),
        (":material/warning:", "Inventory risk", "Compare expected demand with the demo stock position."),
        (":material/shopping_cart:", "Reorder decision", "Recommend an order quantity and explain urgency."),
    ]
    columns = st.columns(4)
    for column, (icon, title, description) in zip(columns, steps, strict=True):
        with column.container(border=True):
            st.markdown(f"### {icon} {title}")
            st.caption(description)

    shortage_pct = shortage_reduction_pct(recommendations)
    st.subheader("Decision snapshot")
    left, right = st.columns(2)
    action_counts = (
        recommendations.groupby("order_action", as_index=False)
        .size()
        .rename(columns={"size": "Series", "order_action": "Action"})
    )
    action_chart = (
        alt.Chart(action_counts)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X("Series:Q", title="Item-store decisions"),
            y=alt.Y("Action:N", title=None, sort=["REORDER", "NO ORDER"]),
            color=alt.Color(
                "Action:N",
                scale=alt.Scale(domain=["REORDER", "NO ORDER"], range=["#0F766E", "#A8B7B2"]),
                legend=None,
            ),
            tooltip=["Action:N", alt.Tooltip("Series:Q", format=",")],
        )
        .properties(title="Saved inventory actions", height=180)
    )
    left.altair_chart(action_chart, width="stretch")

    by_store = (
        recommendations.groupby("store_id", as_index=False)["recommended_order_qty"]
        .sum()
        .rename(columns={"store_id": "Store", "recommended_order_qty": "Recommended units"})
    )
    store_chart = (
        alt.Chart(by_store)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("Store:N", title=None, sort=["CA_1", "TX_2", "WI_3"]),
            y=alt.Y("Recommended units:Q", title="Units"),
            color=alt.Color("Store:N", legend=None),
            tooltip=["Store:N", alt.Tooltip("Recommended units:Q", format=",")],
        )
        .properties(title="Recommended units by store", height=180)
    )
    right.altair_chart(store_chart, width="stretch")

    with st.container(border=True):
        st.markdown("#### Expected business impact in the synthetic inventory demonstration")
        impact = st.columns(3)
        impact[0].metric("Expected shortage before", f"{metrics['shortage_before']:,.1f} units")
        impact[1].metric("Expected shortage after", f"{metrics['shortage_after']:,.1f} units")
        impact[2].metric("Expected shortage reduction", f"{shortage_pct:.1%}")
        st.caption(
            "This is a scenario result from synthetic inventory assumptions, not measured Walmart savings."
        )

    render_methodology(source)


def render_methodology(source: Any) -> None:
    metadata = source.get_model_metadata()
    with st.expander("Methodology, evidence, and limitations"):
        st.markdown(
            f"""
**Data.** Version 1 uses 100 anonymized FOODS products in CA_1, TX_2, and WI_3: 300
item-store series and 582,300 daily observations from the M5 competition dataset.

**Forecast selection.** Simple baselines, Ridge regression, HistGradientBoosting, and XGBoost
were compared with chronological validation. The selected production method is the
**28-Day Historical Mean** (`{metadata.get('selected_model', 'mean_28')}`) because it gave the
strongest balanced validation result over the recursive 30-day horizon.

**Locked test.** Across Days 1–30, the selected method achieved MAE
**{LOCKED_TEST_METRICS['daily_mae']:.3f} units**, daily WAPE
**{LOCKED_TEST_METRICS['daily_wape']:.2%}**, and 30-day aggregate WAPE
**{LOCKED_TEST_METRICS['aggregate_wape']:.2%}**. Aggregate accuracy matters because replenishment
decisions depend on total demand across a supply horizon.

**Inventory logic.** Forecast demand and validation-calibrated uncertainty feed safety stock,
reorder point, target stock, stockout risk, priority, and order quantity. Saved inventory values
are clearly labeled demo inputs.

**Limitations.** Historical sales can be censored by stockouts; future price schedules are not
modeled; products are anonymized; and inventory outcomes are educational simulations rather
than actual Walmart operations.
"""
        )


def forecast_line_chart(history: pd.DataFrame, forecast: pd.DataFrame) -> alt.LayerChart:
    chart_frame = forecast_chart_data(history, forecast)
    tidy = chart_frame.melt(
        id_vars="date", var_name="Series", value_name="Units"
    ).dropna(subset=["Units"])
    actual = (
        alt.Chart(tidy[tidy["Series"].eq("Actual demand")])
        .mark_line(color="#334E68", strokeWidth=2)
        .encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %d")),
            y=alt.Y("Units:Q", title="Daily units", scale=alt.Scale(zero=True)),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("Units:Q", format=".2f")],
        )
    )
    predicted = (
        alt.Chart(tidy[tidy["Series"].eq("Forecast demand")])
        .mark_line(color="#0F766E", strokeWidth=3, strokeDash=[7, 4], point=True)
        .encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %d")),
            y=alt.Y("Units:Q", title="Daily units", scale=alt.Scale(zero=True)),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("Units:Q", format=".2f")],
        )
    )
    origin = pd.DataFrame({"date": [pd.to_datetime(history["date"]).max()]})
    origin_rule = (
        alt.Chart(origin)
        .mark_rule(color="#E07A5F", strokeDash=[3, 3])
        .encode(x="date:T", tooltip=[alt.Tooltip("date:T", title="Forecast origin")])
    )
    return (actual + predicted + origin_rule).properties(
        title="Recent actual demand and selected future forecast", height=380
    )


def render_forecasting(source: Any) -> None:
    st.header("Demand forecast")
    st.markdown(
        "Inspect the selected 30-day production forecast for one item-store series. "
        "M5 identifies products with anonymized SKU codes rather than consumer-facing names."
    )
    _, store, item = sku_selector(source, prefix="forecast")
    horizon = st.segmented_control(
        "Forecast horizon",
        options=[1, 7, 30],
        default=30,
        format_func=lambda value: f"{value} day" if value == 1 else f"{value} days",
    )
    horizon = int(horizon or 30)
    history = source.get_recent_history(store, item, days=120)
    forecast = source.get_forecast(store, item, horizon_days=horizon)
    if history.empty or forecast.empty:
        st.warning("History or production forecast is unavailable for this item-store series.")
        return

    day_one = float(forecast.loc[forecast["horizon_day"].eq(1), "forecast"].sum())
    week_total = float(forecast.loc[forecast["horizon_day"].le(7), "forecast"].sum())
    metric_row(
        [
            ("Tomorrow", f"{day_one:.2f} units", "Day +1 demand forecast."),
            ("Next 7 days", f"{week_total:.2f} units", "Sum of the first seven daily forecasts."),
            (f"{horizon}-day demand", f"{forecast['forecast'].sum():.2f} units", "Total demand across the selected horizon."),
            ("Average per day", f"{forecast['forecast'].mean():.2f} units", "Mean daily forecast in the selected horizon."),
        ]
    )
    st.altair_chart(forecast_line_chart(history, forecast), width="stretch")
    st.caption(
        "The vertical marker is the forecast origin. The dashed line contains forecasts only; "
        "actual future sales are never fed back into the 30-day forecast."
    )

    forecast_download = forecast[
        ["item_id", "store_id", "forecast_origin", "target_date", "horizon_day", "forecast"]
    ].rename(
        columns={
            "item_id": "SKU",
            "store_id": "Store",
            "forecast_origin": "Forecast origin",
            "target_date": "Target date",
            "horizon_day": "Horizon day",
            "forecast": "Forecast units",
        }
    )
    st.download_button(
        "Download selected forecast",
        data=dataframe_to_csv_bytes(forecast_download),
        file_name=f"smartstock_forecast_{store}_{item}_{horizon}d.csv",
        mime="text/csv",
        icon=":material/download:",
    )
    with st.expander("Why this forecast method was selected"):
        st.markdown(
            "The 28-day historical mean was chosen after a locked, chronological comparison with "
            "naive rules, Ridge regression, HistGradientBoosting, and XGBoost. It did not win every "
            "single metric, but it was the most balanced and stable option for 30-day replenishment. "
            "MAE measures average unit error; WAPE expresses total absolute error relative to total demand."
        )


def render_inventory_recommendations(source: Any) -> None:
    st.header("Inventory recommendations")
    st.markdown(
        "Prioritize replenishment decisions using forecast demand, demo inventory position, "
        "safety stock, and supplier lead-time assumptions."
    )
    st.caption(INVENTORY_DISCLAIMER)

    all_recommendations = source.get_recommendations()
    options = cached_filter_options(source)
    filter_columns = st.columns([1.6, 1, 1])
    action = filter_columns[0].segmented_control(
        "Action", ["Needs reorder", "All decisions", "No order"], default="Needs reorder"
    )
    store = filter_columns[1].selectbox("Store", ["All", *options["stores"]], key="inventory_store")
    department = filter_columns[2].selectbox(
        "Department", ["All", *options["departments"]], key="inventory_department"
    )
    filtered = filter_recommendations(
        all_recommendations,
        action=action or "Needs reorder",
        store_id=store,
        dept_id=department,
    )
    if filtered.empty:
        st.warning("No saved inventory decisions match these filters.")
        return

    high_risk = int(filtered["stock_status"].isin(["CRITICAL", "STOCKOUT"]).sum())
    metric_row(
        [
            ("Decisions shown", f"{len(filtered):,}", "One saved decision per item-store series."),
            ("Recommended units", f"{filtered['recommended_order_qty'].sum():,.0f}", "Service-level order quantity."),
            ("Critical or stockout", f"{high_risk:,}", "Series in the two most severe stock states."),
            ("Shortage reduction", f"{shortage_reduction_pct(filtered):.1%}", "Expected reduction in the synthetic scenario."),
        ]
    )

    st.subheader("How to read the recommendation")
    definition_columns = st.columns(3)
    definitions = [
        ("Safety stock", "Extra units held to absorb forecast uncertainty during lead time."),
        ("Reorder point", "Lead-time demand plus safety stock—the level that triggers an order."),
        ("Recommended quantity", "Units needed to reach target stock after considering inventory position."),
    ]
    for column, (title, description) in zip(definition_columns, definitions, strict=True):
        with column.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(description)

    reorder_rows = filtered[filtered["recommended_order_qty"].gt(0)].nlargest(
        10, ["recommended_order_qty", "priority_score"]
    ).copy()
    if not reorder_rows.empty:
        reorder_rows["SKU and store"] = reorder_rows["item_id"] + " · " + reorder_rows["store_id"]
        top_chart = (
            alt.Chart(reorder_rows)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("recommended_order_qty:Q", title="Recommended units"),
                y=alt.Y("SKU and store:N", title=None, sort="-x"),
                color=alt.Color("store_id:N", title="Store"),
                tooltip=[
                    alt.Tooltip("item_id:N", title="SKU"),
                    alt.Tooltip("store_id:N", title="Store"),
                    alt.Tooltip("recommended_order_qty:Q", title="Units", format=","),
                    alt.Tooltip("stock_status:N", title="Status"),
                    alt.Tooltip("priority_label:N", title="Priority"),
                ],
            )
            .properties(title="Top replenishment needs in the current view", height=320)
        )
        st.altair_chart(top_chart, width="stretch")

    display_table = recommendation_table(filtered)
    st.subheader("Prioritized decision table")
    st.caption("Sorted by urgency, then stockout risk and recommended units.")
    st.dataframe(
        display_table,
        width="stretch",
        height=520,
        hide_index=True,
        column_config={
            "Action": st.column_config.TextColumn("Action", help="REORDER means recommended units are greater than zero."),
            "30-day forecast": st.column_config.NumberColumn(format="%.2f"),
            "Safety stock": st.column_config.NumberColumn(format="%.2f"),
            "Reorder point": st.column_config.NumberColumn(format="%.2f"),
            "Recommended units": st.column_config.NumberColumn(format="%.0f"),
            "Stockout risk (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
        },
    )
    st.download_button(
        "Download filtered recommendations",
        data=dataframe_to_csv_bytes(display_table),
        file_name="smartstock_inventory_recommendations.csv",
        mime="text/csv",
        icon=":material/download:",
    )


def render_scenario_planner(source: Any) -> None:
    st.header("Scenario planner")
    st.markdown(
        "Change operational assumptions for one product and compare the revised decision with "
        "the saved recommendation. Nothing entered here is written back to project data."
    )
    st.caption(INVENTORY_DISCLAIMER)
    _, store, item = sku_selector(source, prefix="scenario", prioritize_reorders=True)
    snapshot_frame = source.get_snapshot(store, item)
    recommendation_frame = source.get_recommendations({"store_id": store, "item_id": item})
    forecast = source.get_forecast(store, item, horizon_days=30)
    if snapshot_frame.empty or recommendation_frame.empty or forecast.empty:
        st.warning("Scenario inputs are incomplete for this item-store selection.")
        return

    snapshot = snapshot_frame.iloc[0].to_dict()
    saved = recommendation_frame.iloc[0].to_dict()
    policy = get_policy()
    default_review = min(int(policy["review_period_days"]), 30 - int(snapshot["lead_time_days"]))
    original_inputs = {
        "lead_time_days": int(snapshot["lead_time_days"]),
        "service_level": float(snapshot["service_level"]),
        "review_period_days": default_review,
        "holding_cost_per_unit_per_day": float(snapshot["holding_cost_per_unit_per_day"]),
        "stockout_cost_per_unit": float(snapshot["stockout_cost_per_unit"]),
        "fixed_order_cost": float(snapshot["fixed_order_cost"]),
    }

    with st.form("inventory_scenario"):
        st.markdown("#### Scenario assumptions")
        stock_columns = st.columns(3)
        on_hand = stock_columns[0].number_input(
            "On hand", min_value=0.0, max_value=10_000.0, value=float(snapshot["on_hand"]), step=1.0
        )
        on_order = stock_columns[1].number_input(
            "On order", min_value=0.0, max_value=10_000.0, value=float(snapshot["on_order"]), step=1.0
        )
        backorders = stock_columns[2].number_input(
            "Backorders", min_value=0.0, max_value=10_000.0, value=float(snapshot["backorders"]), step=1.0
        )
        policy_columns = st.columns(3)
        lead_time = policy_columns[0].number_input(
            "Lead time (days)", min_value=1, max_value=30, value=int(snapshot["lead_time_days"]), step=1
        )
        review_period = policy_columns[1].number_input(
            "Review period (days)", min_value=0, max_value=29, value=default_review, step=1,
            help="Lead time plus review period must fit inside the 30-day forecast."
        )
        service_options = [0.90, 0.95, 0.975, 0.99]
        service_default = min(service_options, key=lambda value: abs(value - float(snapshot["service_level"])))
        service_level = policy_columns[2].select_slider(
            "Service target", options=service_options, value=service_default, format_func=lambda value: f"{value:.1%}"
        )
        with st.expander("Advanced cost assumptions"):
            cost_columns = st.columns(3)
            holding_cost = cost_columns[0].number_input(
                "Holding cost / unit / day", min_value=0.0, max_value=100.0,
                value=float(snapshot["holding_cost_per_unit_per_day"]), step=0.01, format="%.3f"
            )
            stockout_cost = cost_columns[1].number_input(
                "Stockout cost / unit", min_value=0.0, max_value=1_000.0,
                value=float(snapshot["stockout_cost_per_unit"]), step=0.5
            )
            fixed_cost = cost_columns[2].number_input(
                "Fixed order cost", min_value=0.0, max_value=10_000.0,
                value=float(snapshot["fixed_order_cost"]), step=1.0
            )
        st.form_submit_button("Recalculate scenario", type="primary")

    if int(lead_time) + int(review_period) > 30:
        st.error("Lead time plus review period cannot exceed the available 30-day forecast.")
        return
    overrides = {
        "on_hand": on_hand,
        "on_order": on_order,
        "backorders": backorders,
        "lead_time_days": lead_time,
        "service_level": service_level,
        "review_period_days": review_period,
        "holding_cost_per_unit_per_day": holding_cost,
        "stockout_cost_per_unit": stockout_cost,
        "fixed_order_cost": fixed_cost,
    }
    calibration = {
        "sigma_daily": saved["sigma_daily"],
        "residual_mean_used": saved["residual_mean_used"],
    }
    revised = apply_inventory_scenario(
        forecast["forecast"], snapshot, calibration, policy, overrides
    )

    st.subheader("Saved decision compared with this scenario")
    before, after = st.columns(2)
    with before.container(border=True):
        st.markdown("#### Saved recommendation")
        st.metric("Recommended order", f"{int(saved['recommended_order_qty']):,} units")
        st.metric("Inventory position", f"{float(saved['inventory_position']):.1f} units")
        st.metric("Stockout risk", f"{float(saved['stockout_risk_pct']):.1f}%")
        st.metric("Status and priority", f"{saved['stock_status']} · {saved['priority_label']}")
    with after.container(border=True):
        order_delta = int(revised["recommended_order_qty"]) - int(saved["recommended_order_qty"])
        st.markdown("#### Revised scenario")
        st.metric(
            "Recommended order", f"{int(revised['recommended_order_qty']):,} units",
            delta=f"{order_delta:+,} vs saved", delta_color="off"
        )
        st.metric("Inventory position", f"{float(revised['inventory_position']):.1f} units")
        st.metric("Stockout risk", f"{float(revised['stockout_risk_pct']):.1f}%")
        st.metric("Status and priority", f"{revised['stock_status']} · {revised['priority_label']}")

    new_inputs = {
        "lead_time_days": int(lead_time),
        "service_level": float(service_level),
        "review_period_days": int(review_period),
        "holding_cost_per_unit_per_day": float(holding_cost),
        "stockout_cost_per_unit": float(stockout_cost),
        "fixed_order_cost": float(fixed_cost),
    }
    st.markdown("#### Why the recommendation changed")
    for explanation in scenario_explanations(saved, revised, original_inputs, new_inputs):
        st.markdown(f"- {explanation}")

    with st.expander("Decision calculation details"):
        detail_metrics = st.columns(4)
        detail_metrics[0].metric("Lead-time demand", f"{revised['lead_time_demand']:.1f}")
        detail_metrics[1].metric("Safety stock", f"{revised['safety_stock']:.1f}")
        detail_metrics[2].metric("Reorder point", f"{revised['reorder_point']:.1f}")
        detail_metrics[3].metric("Target stock", f"{revised['target_stock_level']:.1f}")
        cost_metrics = st.columns(3)
        cost_metrics[0].metric("Cost-optimized order", f"{int(revised['cost_optimized_order_qty']):,} units")
        cost_metrics[1].metric("Estimated cost before (demo)", money(revised["estimated_cost_without_order"]))
        cost_metrics[2].metric("Estimated cost after (demo)", money(revised["estimated_cost_with_recommendation"]))
    st.caption("Scenario calculation is in memory only and never overwrites the saved recommendation.")


def main() -> None:
    try:
        resolution = get_resolution()
    except Exception as exc:
        st.error(f"SmartStock could not start: {exc}")
        st.info("Generate the inventory artifacts or initialize PostgreSQL, then reload the app.")
        st.stop()

    with st.sidebar:
        st.markdown("## :material/inventory_2: SmartStock")
        st.caption("Retail demand and inventory planning")
        section = st.radio(
            "Explore",
            ["Overview", "Demand forecast", "Inventory recommendations", "Scenario planner"],
        )
        st.markdown("**Data source**")
        if resolution.fallback_used and "explicitly" not in resolution.notice:
            st.warning(resolution.source.label, icon=":material/database:")
        else:
            st.info(resolution.source.label, icon=":material/database:")
        st.caption(resolution.notice)
        st.caption("Historical M5 demand · Synthetic demo inventory")

    st.title("SmartStock")
    st.caption("A decision-support case study for retail demand forecasting and replenishment")
    renderers = {
        "Overview": lambda: render_overview(resolution.source),
        "Demand forecast": lambda: render_forecasting(resolution.source),
        "Inventory recommendations": lambda: render_inventory_recommendations(resolution.source),
        "Scenario planner": lambda: render_scenario_planner(resolution.source),
    }
    renderers[section]()
    st.caption(
        "SmartStock V1 portfolio demonstration · Forecasts use M5 history; inventory and cost values are synthetic assumptions."
    )


if __name__ == "__main__":
    main()
