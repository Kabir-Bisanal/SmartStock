"""Create the focused Stage 5 exploratory-analysis figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_STORES = ("CA_1", "TX_2", "WI_3")
WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
MONTH_ORDER = list(range(1, 13))
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def create_figures(tables: dict[str, pd.DataFrame], figure_dir: Path) -> list[dict[str, str]]:
    """Create a focused set of 12 professional, non-decorative EDA figures."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"CA_1": "#2563EB", "TX_2": "#EA580C", "WI_3": "#16A34A"}
    band_colors = {"low": "#60A5FA", "medium": "#F59E0B", "high": "#DC2626"}
    figures: list[dict[str, str]] = []

    def save(fig: Any, filename: str, title: str) -> None:
        path = figure_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures.append({"filename": filename, "title": title})

    enriched = tables["enriched"]
    active = enriched[enriched["active_flag"]]

    # 1. Demand distribution: clipping is visual only.
    clip_value = float(active["sales"].quantile(0.99))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].hist(active["sales"].clip(upper=clip_value), bins=np.arange(-0.5, clip_value + 1.5, 1), color="#2563EB", edgecolor="white")
    axes[0].set_title(f"Active daily sales (visualized through p99 = {clip_value:g})")
    axes[0].set_xlabel("Units sold per item-store-day")
    axes[0].set_ylabel("Observations")
    positive = active.loc[active["sales"].gt(0), "sales"]
    axes[1].hist(np.log1p(positive), bins=40, color="#0F766E", edgecolor="white")
    axes[1].set_title("Positive daily sales on log(1 + sales) scale")
    axes[1].set_xlabel("log(1 + units sold)")
    axes[1].set_ylabel("Observations")
    fig.suptitle("Overall active-period demand distribution\nUnderlying data is unchanged; clipping/log scaling is display-only", fontsize=13)
    save(fig, "01_overall_sales_distribution.png", "Overall active-period sales distribution")

    # 2. Demand bands.
    band = tables["demand_band"].set_index("demand_band").reindex(["low", "medium", "high"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].bar(band.index, band["active_mean_sales"], color=[band_colors[index] for index in band.index])
    axes[0].set_title("Mean active daily sales")
    axes[0].set_ylabel("Units per active observation")
    axes[1].bar(band.index, band["active_zero_pct"], color=[band_colors[index] for index in band.index])
    axes[1].set_title("Active-period zero prevalence")
    axes[1].set_ylabel("Zero-sales observations (%)")
    fig.suptitle("Demand bands remain behaviorally distinct after launch", fontsize=13)
    save(fig, "02_demand_band_comparison.png", "Demand-band comparison")

    # 3. Departments.
    dept = tables["department"].set_index("dept_id").sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].bar(dept.index, dept["sales_share_pct"], color="#7C3AED")
    axes[0].set_title("Share of V1 unit sales")
    axes[0].set_ylabel("Sales share (%)")
    axes[1].bar(dept.index, dept["active_zero_pct"], color="#A78BFA")
    axes[1].set_title("Active-period zero prevalence")
    axes[1].set_ylabel("Zero-sales observations (%)")
    fig.suptitle("Department demand composition", fontsize=13)
    save(fig, "03_department_comparison.png", "Department demand comparison")

    # 4. Stores and department mix.
    store = tables["store"].set_index("store_id").reindex(EXPECTED_STORES)
    mix = tables["store_department"].pivot(index="store_id", columns="dept_id", values="store_sales_share_pct").reindex(EXPECTED_STORES)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].bar(store.index, store["active_mean_sales"], color=[colors[index] for index in store.index])
    axes[0].set_title("Mean active daily sales")
    axes[0].set_ylabel("Units per active observation")
    bottom = np.zeros(len(mix))
    for department, color in zip(mix.columns, ["#93C5FD", "#FBBF24", "#34D399"], strict=False):
        axes[1].bar(mix.index, mix[department], bottom=bottom, label=str(department), color=color)
        bottom += mix[department].to_numpy()
    axes[1].set_title("Department share within each store")
    axes[1].set_ylabel("Store unit-sales share (%)")
    axes[1].legend(title="Department", fontsize=8)
    fig.suptitle("Store demand scale and department mix", fontsize=13)
    save(fig, "04_store_comparison.png", "Store demand and department mix")

    # 5. Normalized weekly store trends.
    fig, ax = plt.subplots(figsize=(11, 5))
    for store_id, group in tables["weekly_store"].groupby("store_id", observed=True):
        ax.plot(group["date"], group["normalized_13_week_mean"], label=str(store_id), color=colors[str(store_id)], linewidth=1.8)
    ax.axhline(1, color="#64748B", linewidth=1, linestyle="--")
    ax.set_title("Normalized weekly sales by store (13-week centered mean)")
    ax.set_xlabel("Week")
    ax.set_ylabel("Weekly sales / store median")
    ax.legend(title="Store")
    save(fig, "05_normalized_weekly_store_trends.png", "Normalized weekly store trends")

    # 6. Overall monthly trend.
    monthly = tables["monthly_trend"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(monthly["date"], monthly["monthly_sales"], color="#94A3B8", linewidth=1, label="Monthly units")
    ax.plot(monthly["date"], monthly["12_month_mean"], color="#1D4ED8", linewidth=2.2, label="12-month centered mean")
    ax.plot(monthly["date"], monthly["linear_trend"], color="#DC2626", linestyle="--", linewidth=1.5, label="Descriptive linear trend")
    ax.set_title("SmartStock V1 monthly unit-sales trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Units sold")
    ax.legend()
    save(fig, "06_monthly_sales_trend.png", "Monthly sales trend")

    # 7. Weekday pattern by store.
    weekday = tables["weekday_store"].copy()
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(WEEKDAY_ORDER))
    for store_id, group in weekday.groupby("store_id", observed=True):
        ordered = group.set_index("weekday").reindex(WEEKDAY_ORDER)
        ax.plot(x, ordered["sales"], marker="o", label=str(store_id), color=colors[str(store_id)])
    ax.set_xticks(x, WEEKDAY_ORDER, rotation=25)
    ax.set_title("Mean active daily sales by weekday and store")
    ax.set_ylabel("Units per active observation")
    ax.legend(title="Store")
    save(fig, "07_weekday_pattern.png", "Weekday demand pattern")

    # 8. Monthly seasonality.
    season = tables["monthly_seasonality"]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for store_id, group in season.groupby("store_id", observed=True):
        ordered = group.set_index("month").reindex(MONTH_ORDER)
        ax.plot(MONTH_ORDER, ordered["average_daily_sales"], marker="o", label=str(store_id), color=colors[str(store_id)])
    ax.set_xticks(MONTH_ORDER, MONTH_LABELS)
    ax.set_title("Recurring month-of-year pattern in daily store totals")
    ax.set_ylabel("Average daily units")
    ax.legend(title="Store")
    save(fig, "08_monthly_seasonality.png", "Monthly seasonality by store")

    # 9. Event and SNAP comparisons.
    event = tables["event_comparison"].copy()
    snap = tables["snap_comparison"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    event_labels = ["Non-event", "Event"]
    event_values = [float(event.loc[event["event_day"].eq(flag), "mean_sales"].iloc[0]) for flag in (False, True)]
    axes[0].bar(event_labels, event_values, color=["#94A3B8", "#8B5CF6"])
    axes[0].set_title("Event vs non-event")
    axes[0].set_ylabel("Mean active daily sales")
    width = 0.35
    x = np.arange(len(EXPECTED_STORES))
    snap_pivot = snap.pivot(index="store_id", columns="snap_active", values="mean_sales").reindex(EXPECTED_STORES)
    axes[1].bar(x - width / 2, snap_pivot[0], width, label="Non-SNAP", color="#94A3B8")
    axes[1].bar(x + width / 2, snap_pivot[1], width, label="SNAP", color="#059669")
    axes[1].set_xticks(x, EXPECTED_STORES)
    axes[1].set_title("State-specific SNAP association")
    axes[1].set_ylabel("Mean active daily sales")
    axes[1].legend()
    fig.suptitle("Calendar-associated demand differences (observational, not causal)", fontsize=13)
    save(fig, "09_event_and_snap_comparison.png", "Event and SNAP comparisons")

    # 10. Price distribution by department.
    weekly_prices = tables["weekly_prices"]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    departments = sorted(str(value) for value in weekly_prices["dept_id"].unique())
    price_groups = [weekly_prices.loc[weekly_prices["dept_id"].astype("string").eq(dept), "sell_price"] for dept in departments]
    ax.boxplot(price_groups, tick_labels=departments, showfliers=False, whis=(5, 95), patch_artist=True, boxprops={"facecolor": "#FBBF24"})
    ax.set_title("Weekly selling-price distribution by department\nWhiskers show 5th-95th percentiles; underlying values are unchanged")
    ax.set_ylabel("Selling price")
    save(fig, "10_price_distribution.png", "Price distribution by department")

    # 11. Price changes and weekly demand.
    direction = tables["price_direction"].set_index("change_direction").reindex(["decrease", "unchanged", "increase"])
    fig, ax = plt.subplots(figsize=(9.5, 4.7))
    ax.bar(direction.index, direction["mean_weekly_sales"], color=["#16A34A", "#94A3B8", "#DC2626"])
    ax.set_title("Weekly sales by sequential price-change direction\nObservational comparison; not a causal estimate")
    ax.set_ylabel("Mean weekly units")
    save(fig, "11_price_change_vs_demand.png", "Price-change direction and demand")

    # 12. Intermittency and launch timing.
    series = tables["series_metrics"]
    launches = tables["product_launches"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].hist(series["active_zero_pct"], bins=np.arange(0, 102, 5), color="#0EA5E9", edgecolor="white")
    axes[0].set_title("Item-store active zero prevalence")
    axes[0].set_xlabel("Active-period zero-sales days (%)")
    axes[0].set_ylabel("Series")
    launch_counts = launches["launch_year"].value_counts().sort_index()
    axes[1].bar(launch_counts.index.astype(str), launch_counts.values, color="#7C3AED")
    axes[1].set_title("Selected products by earliest known-price year")
    axes[1].set_xlabel("Launch year")
    axes[1].set_ylabel("Products")
    fig.suptitle("Intermittent demand and product launch timing", fontsize=13)
    save(fig, "12_intermittency_and_launches.png", "Intermittency and launch timing")

    return figures


