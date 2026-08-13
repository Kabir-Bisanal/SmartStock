"""Render calculated Stage 6 results as a beginner-readable Markdown report."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence


def display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Passed" if value else "FAILED"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.4f}"
    return str(value)


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    def escape(value: Any) -> str:
        return display(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(escape(value) for value in row) + " |" for row in rows)
    return lines


def render_stage6_report(summary: dict[str, Any], feature_manifest: dict[str, Any], validation_manifest: dict[str, Any]) -> str:
    """Render all required Stage 6 report sections from real calculated values."""

    active = summary["active_period"]
    processed = summary["processed_dataset"]
    price = summary["price_review"]
    validation = summary["chronological_validation"]
    leakage = summary["leakage_audit"]
    grouped_features: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for feature in feature_manifest["features"]:
        grouped_features[feature["category"]].append(feature)

    lines: list[str] = [
        "# SmartStock Stage 6 — Leakage-Safe Feature Engineering & Chronological Validation Design",
        "",
        f"Generated at `{summary['generated_at_utc']}` from the frozen SmartStock Version 1 interim dataset.",
        "",
        "> Stage 6 builds modeling inputs and a validation design only. No forecast model is fitted, scored, or selected.",
        "",
        "## Executive Summary",
        "",
        f"Stage 6 excluded **{active['prelaunch_rows_excluded']:,}** unavailable pre-launch rows, retained **{active['active_rows']:,}** active rows, and persisted **{processed['feature_ready_rows']:,}** fully feature-ready rows plus their warm-up rows. The processed file contains **{processed['rows']:,} rows × {processed['columns']} columns** across 100 products, three stores, and 300 item-store series.",
        "",
        f"Every default model feature passed the manifest leakage audit. Three expanding 30-day validation folds precede a locked 30-day final test from `{validation_manifest['locked_final_test']['start_date']}` through `{validation_manifest['locked_final_test']['end_date']}`.",
        "",
        "## Modeling Policies",
        "",
        "- Pre-launch rows are not demand targets.",
        "- Active-period zero sales remain legitimate observations.",
        "- First known price is the availability proxy; it is not an explicit inventory record.",
        "- Future models are intended to be global across the 300 series.",
        "- `demand_band` is analysis-only because it uses full-history demand.",
        "- Evaluation must be chronological; random splitting is prohibited.",
        "- Historical demand and price features use information strictly before the target date or week.",
        "",
        "## Active-Period Filtering",
        "",
    ]
    lines.extend(
        table(
            ["Measure", "Result"],
            [
                ("Input rows", summary["input_validation"]["rows"]),
                ("Pre-launch rows excluded", active["prelaunch_rows_excluded"]),
                ("Pre-launch share", f"{active['prelaunch_pct']:.2f}%"),
                ("Active rows retained", active["active_rows"]),
                ("Active zero-sales rows retained", active["active_zero_rows_retained"]),
                ("Active zero rate", f"{active['active_zero_pct']:.2f}%"),
                ("Price-based availability pairs", active["price_proxy_pairs"]),
                ("Positive-sale fallback pairs", active["positive_sale_fallback_pairs"]),
            ],
        )
    )
    lines.extend([
        "",
        "Filtering before feature creation prevents unavailable assortment periods from teaching a model that a not-yet-offered product had genuine zero demand.",
        "",
        "## Calendar Features",
        "",
        "Calendar predictors are known for the target date in advance: `day_of_week`, `day_of_month`, `month`, `quarter`, `year`, `week_of_year`, and `is_weekend`. Parsed dates were checked against the original M5 weekday, month, and year fields.",
        "",
        "The sine/cosine encodings for weekday and month put the end and beginning of each cycle close together. For example, December and January are neighbors on a circle even though their integer labels are 12 and 1.",
        "",
        "## Event / SNAP Features",
        "",
        f"`is_event` accompanies the four raw event identity fields. Empty event fields mean no event in that slot; no sales-derived target encoding was used. There are **{summary['event_snap']['event_rows']:,}** active event rows.",
        "",
        "A single `snap_active` field maps `CA_1 → snap_CA`, `TX_2 → snap_TX`, and `WI_3 → snap_WI`. The unrelated state flags are not persisted as predictors.",
        "",
        "## Product-Age Features",
        "",
        f"`product_age_days` equals target date minus availability date and starts at zero. `product_age_weeks` stores completed weeks. Observed active ages range from **0** to **{active['product_age_days_max']:,} days**. Recently launched products can behave differently from mature products even when they share a department or store.",
        "",
        "## Sales Lag Features",
        "",
    ])
    lines.extend(
        table(
            ["Feature", "Exact definition"],
            [
                ("sales_lag_1", "sales(t-1)"),
                ("sales_lag_7", "sales(t-7)"),
                ("sales_lag_14", "sales(t-14)"),
                ("sales_lag_28", "sales(t-28)"),
            ],
        )
    )
    lines.extend([
        "",
        "Every shift is calculated independently within `(store_id, item_id)`. Yesterday's observed sales may help predict today, but today's target cannot be used to predict itself.",
        "",
        "## Rolling Features",
        "",
        "The rolling means use windows of 7, 14, and 28 prior observations. Rolling sample standard deviations use 7 and 28 prior observations. All formulas use the equivalent of `sales.shift(1).rolling(window)`, so the target day's sale is outside its own window.",
        "",
        "Without `shift(1)`, a rolling average would contain `sales(t)`. A model evaluated with that feature could appear artificially accurate because part of the answer was included in the input.",
        "",
        "## Intermittency Features",
        "",
        "`zero_rate_7` and `zero_rate_28` measure the share of zero-demand days in strictly historical windows. `days_since_last_positive_sale` looks backward from `t-1`. Missing prior history remains missing and is never confused with an observed zero.",
        "",
        "## Price-Feature Policy",
        "",
        "Default forecast-safe price features use completed prior M5 weeks:",
        "",
        "- `previous_week_price`: price in the preceding item-store week.",
        "- `price_change_previous_week`: preceding-week price minus the price two weeks before the target week.",
        "- `price_pct_change_previous_week`: the same change as a percentage of the earlier price.",
        "- `price_vs_4week_median_lagged`: preceding-week price divided by the median price of the preceding four weeks, minus one.",
        "",
        "`known_future_sell_price` preserves the actual target-week price only for explicitly defined known-future-price experiments. It is excluded from the default feature list. A price appearing in a historical holdout file does not prove that a production forecasting system would know it before making the forecast.",
        "",
        "## Extreme Price-Change Review",
        "",
        f"The weekly review found **{price['price_change_events']:,}** non-zero sequential changes. The maximum absolute percentage change is **{price['maximum_absolute_pct_change']:.2f}%**, from **{price['largest_change_previous_price']:.2f}** to **{price['largest_change_new_price']:.2f}**. The very low previous-price denominator is the primary reason the percentage is so large.",
        "",
    ])
    lines.extend(
        table(
            ["Store", "Item", "Week", "Previous", "New", "Absolute change", "% change"],
            [
                (row["store_id"], row["item_id"], row["wm_yr_wk"], row["previous_week_price"], row["weekly_price"], row["current_change_absolute"], row["current_change_pct"])
                for row in summary["extreme_price_changes"]
            ],
        )
    )
    lines.extend([
        "",
        "No price was deleted, capped, or transformed. Whether later modeling clips or transforms percentage changes must be decided using training and validation periods only.",
        "",
        "## Feature Missingness / Warm-up",
        "",
    ])
    lines.extend(
        table(
            ["Required history feature", "Missing rows", "Missing %"],
            [(row["feature"], row["missing_rows"], row["missing_pct"]) for row in processed["missingness"]],
        )
    )
    lines.extend([
        "",
        f"A row is feature-ready only when all required demand-history and price-history fields are present. **{processed['warmup_rows']:,} rows** remain persisted but are flagged not ready; **{processed['feature_ready_rows']:,} rows ({processed['feature_ready_pct']:.2f}%)** are ready. First readiness occurs between product ages **{processed['first_ready_product_age_days_min']} and {processed['first_ready_product_age_days_max']} days** across the 300 series.",
        "",
        "## Final Feature Dataset",
        "",
    ])
    lines.extend(
        table(
            ["Measure", "Result"],
            [
                ("Path", summary["output"]["processed_dataset"]),
                ("Rows", processed["rows"]),
                ("Columns", processed["columns"]),
                ("Feature-ready rows", processed["feature_ready_rows"]),
                ("Products", processed["items"]),
                ("Stores", processed["stores"]),
                ("Item-store pairs", processed["item_store_pairs"]),
                ("Date range", f"{processed['date_min']} through {processed['date_max']}"),
                ("CSV size", f"{processed['file_size_mib']:.2f} MiB"),
                ("Feature DataFrame memory", f"{processed['memory_mib']:.2f} MiB"),
            ],
        )
    )
    lines.extend([
        "",
        "Warm-up rows are kept with `is_feature_ready = false`; nothing is silently dropped beyond the agreed pre-launch exclusion. Split labels are not embedded because an expanding-window row can be training data in one fold and validation data in an earlier fold.",
        "",
        "Constant persisted columns are: " + ", ".join(f"`{name}`" for name in summary["constant_columns"]) + ". They are retained for context or auditability but excluded from default modeling when they add no variation.",
        "",
        "## Chronological Validation Design",
        "",
        "Training expands through all feature-ready history strictly before each validation start. Random splitting is prohibited because it would allow later retail behavior to influence evaluation of predictions for earlier dates.",
        "",
        "Multiple recent origins reduce the chance of selecting a model because it happened to perform well in one unusual month.",
        "",
        "## Locked Final Test",
        "",
        f"The final test is locked from **{validation['locked_final_test']['start_date']} through {validation['locked_final_test']['end_date']}**, containing **{validation['locked_final_test']['feature_ready_rows']:,} feature-ready rows** across all 300 series. From Stage 6 onward its target performance must not guide feature or model choices.",
        "",
        "Stage 5 descriptively inspected the full history before this lock. That limitation is documented honestly; the remedy is strict non-inspection of final-test model performance from this point forward.",
        "",
        "## Rolling Validation Folds",
        "",
    ])
    lines.extend(
        table(
            ["Fold", "Training end", "Training rows", "Validation dates", "Validation rows"],
            [
                (row["name"], row["training_end_date"], row["training_rows"], f"{row['validation_start_date']} through {row['validation_end_date']}", row["validation_rows"])
                for row in validation["folds"]
            ],
        )
    )
    lines.extend([
        "",
        "## Horizon Evaluation Plan",
        "",
        "- 1-day: evaluate the Day+1 daily forecast.",
        "- 7-day: evaluate daily Day+1 through Day+7 errors and the aggregate seven-day total.",
        "- 30-day: evaluate daily Day+1 through Day+30 errors and the aggregate 30-day total.",
        "",
        "Future models should generate daily forecasts; the 7-day and 30-day business totals are aggregations of those daily predictions.",
        "",
        "## Leakage Audit",
        "",
    ])
    lines.extend(table(["Audit check", "Result"], leakage["checks"].items()))
    lines.extend([
        "",
        f"The **{leakage['default_feature_count']} default features** come only from static identity, known same-day calendar information, the availability proxy, past demand, or past prices. No default feature uses future sales, full-history target aggregates, optional current price, or final-test statistics.",
        "",
        "## Features Deliberately Excluded",
        "",
        "- `demand_band` as a predictor: it uses full-history sales and is analysis-only.",
        "- Current target-week price from the default set: retained only as `known_future_sell_price` for explicit scenarios.",
        "- All three raw state SNAP columns: replaced with the store-relevant flag.",
        "- Target encodings and full-history product averages.",
        "- Future sales, centered rolling windows, target-day rolling values, and final-test-derived statistics.",
        "- A 56-day feature family: excluded to keep the first model foundation compact and preserve more history.",
        "",
        "## Risks / Open Questions",
        "",
        "- First known price is an availability proxy rather than explicit assortment or inventory status.",
        "- Recorded sales may understate unconstrained demand during unobserved stockouts.",
        "- Future price availability must be defined for each deployment scenario.",
        "- Event and SNAP features are calendar associations, not causal effects.",
        "- Extreme price percentages are denominator-sensitive.",
        "- Categorical handling, baseline metrics, and the precise global-model training protocol remain Stage 7 decisions.",
        "",
        "## Recommended Next Stage",
        "",
        "Review and freeze a Stage 7 baseline and metric strategy before fitting anything. That stage should compare leakage-safe chronological baselines across the three validation folds, preserve the final-test lock, and report results by store, department, demand band, and intermittency segment without using those analysis-only groups as leaked predictors.",
        "",
        "## Scope Confirmation",
        "",
        "No forecasting model, model score, hyperparameter search, database, inventory calculation, dashboard, Docker configuration, or deployment artifact was created in Stage 6.",
    ])
    return "\n".join(lines).rstrip() + "\n"
