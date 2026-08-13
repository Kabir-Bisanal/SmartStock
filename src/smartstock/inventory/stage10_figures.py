"""Business-focused figures for SmartStock Stage 10 inventory recommendations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STATUS_ORDER = ["STOCKOUT", "CRITICAL", "REORDER NOW", "LOW", "HEALTHY", "OVERSTOCK"]
PRIORITY_ORDER = ["URGENT", "HIGH", "MEDIUM", "LOW", "NONE"]


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path.as_posix()


def _label_bars(ax: plt.Axes, bars: Any, *, decimals: int = 0) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def create_stage10_figures(recommendations: pd.DataFrame, figure_dir: Path) -> list[str]:
    """Create eight readable inventory decision figures."""

    paths: list[str] = []
    status_counts = recommendations["stock_status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(status_counts.index, status_counts.values, color="#4c78a8")
    ax.set_title("Synthetic Demo Stock Status Distribution")
    ax.set_ylabel("Item-store series")
    ax.tick_params(axis="x", rotation=20)
    ax.spines[["top", "right"]].set_visible(False)
    _label_bars(ax, bars)
    paths.append(_save(fig, figure_dir / "01_stock_status_distribution.png"))

    positive_orders = recommendations.loc[
        recommendations["recommended_order_qty"].gt(0), "recommended_order_qty"
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(positive_orders, bins=20, color="#f58518", edgecolor="white")
    ax.set_title("Service-Level Reorder Quantity Distribution")
    ax.set_xlabel("Recommended units (positive orders only)")
    ax.set_ylabel("Item-store series")
    ax.spines[["top", "right"]].set_visible(False)
    paths.append(_save(fig, figure_dir / "02_reorder_quantity_distribution.png"))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(recommendations["stockout_risk_pct"], bins=np.arange(0, 105, 5), color="#e45756", edgecolor="white")
    ax.axvline(75, color="#222222", linestyle="--", label="Critical threshold")
    ax.set_title("Estimated Lead-Time Stockout Risk")
    ax.set_xlabel("Estimated risk (%)")
    ax.set_ylabel("Item-store series")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    paths.append(_save(fig, figure_dir / "03_stockout_risk_distribution.png"))

    priority_counts = recommendations["priority_label"].value_counts().reindex(PRIORITY_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(priority_counts.index, priority_counts.values, color="#b279a2")
    ax.set_title("Reorder Priority Distribution")
    ax.set_ylabel("Item-store series")
    ax.spines[["top", "right"]].set_visible(False)
    _label_bars(ax, bars)
    paths.append(_save(fig, figure_dir / "04_priority_distribution.png"))

    by_store = recommendations.groupby("store_id", observed=True)["recommended_order_qty"].sum()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(by_store.index, by_store.values, color="#54a24b")
    ax.set_title("Recommended Service-Level Units by Store")
    ax.set_ylabel("Units")
    ax.spines[["top", "right"]].set_visible(False)
    _label_bars(ax, bars)
    paths.append(_save(fig, figure_dir / "05_recommendations_by_store.png"))

    costs = [
        recommendations["estimated_cost_without_order"].sum(),
        recommendations["estimated_cost_with_recommendation"].sum(),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(["No order", "Cost-optimized order"], costs, color=["#e45756", "#72b7b2"])
    ax.set_title("Estimated Scenario Cost Before vs After")
    ax.set_ylabel("Demo currency units")
    ax.spines[["top", "right"]].set_visible(False)
    _label_bars(ax, bars, decimals=1)
    paths.append(_save(fig, figure_dir / "06_cost_before_after.png"))

    shortages = [
        recommendations["expected_shortage_without_order"].sum(),
        recommendations["expected_shortage_with_recommendation"].sum(),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(["No order", "Cost-optimized order"], shortages, color=["#e45756", "#72b7b2"])
    ax.set_title("Expected Scenario Shortage Before vs After")
    ax.set_ylabel("Units")
    ax.spines[["top", "right"]].set_visible(False)
    _label_bars(ax, bars, decimals=1)
    paths.append(_save(fig, figure_dir / "07_shortage_before_after.png"))

    finite_supply = recommendations.loc[np.isfinite(recommendations["days_of_supply"]), "days_of_supply"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(finite_supply.clip(upper=60), bins=20, color="#4c78a8", edgecolor="white")
    ax.axvline(10, color="#e45756", linestyle="--", label="Demo low-stock threshold")
    ax.set_title("Days of Supply Distribution (Capped at 60 for Display)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Item-store series")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    paths.append(_save(fig, figure_dir / "08_days_of_supply_distribution.png"))
    return paths
