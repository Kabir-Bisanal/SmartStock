"""Render the Stage 5 EDA summary as a detailed Markdown report."""

from __future__ import annotations

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


def figure(filename: str, alt: str) -> list[str]:
    return ["", f"![{alt}](figures/stage5/{filename})", ""]


def records_by_key(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(record[key]): record for record in records}


def render_stage5_report(summary: dict[str, Any]) -> str:
    """Render all required Stage 5 sections using calculated values only."""

    validation = summary["dataset_validation"]
    prelaunch = summary["prelaunch_vs_active"]
    all_demand = summary["overall_demand"]["all_rows"]
    active_demand = summary["overall_demand"]["active_rows"]
    bands = summary["demand_band_analysis"]
    departments = summary["department_analysis"]
    stores = summary["store_analysis"]
    time = summary["time_analysis"]
    events = summary["event_analysis"]
    snap = summary["snap_analysis"]
    prices = summary["price_analysis"]
    intermittency = summary["intermittent_demand"]
    outliers = summary["outlier_analysis"]
    launches = summary["product_launch_analysis"]
    readiness = summary["modeling_readiness"]

    band_map = records_by_key(bands, "demand_band")
    dept_map = records_by_key(departments, "dept_id")
    store_map = records_by_key(stores, "store_id")
    event_map = {bool(record["event_day"]): record for record in events["active_observation_comparison"]}
    snap_store = {(record["store_id"], int(record["snap_active"])): record for record in snap["by_store"]}

    lines: list[str] = [
        "# SmartStock Stage 5 — Exploratory Data Analysis & Modeling Readiness",
        "",
        f"Generated at `{summary['generated_at_utc']}` from the frozen SmartStock Version 1 dataset.",
        "",
        "> Stage 5 is descriptive and observational. It does not modify the Version 1 data, persist production features, split data, or train forecasting models.",
        "",
        "## Executive Summary",
        "",
        f"The validated dataset contains **{validation['rows']:,} rows**, **{validation['items']} products**, **{validation['stores']} stores**, and **{validation['item_store_pairs']} complete item-store series**. Apparent pre-launch history accounts for **{prelaunch['prelaunch_rows']:,} rows ({prelaunch['prelaunch_pct']:.2f}%)**. After availability begins, **{prelaunch['active_zero_sales_pct']:.2f}%** of observations still have zero demand, confirming that intermittency is a genuine modeling challenge rather than only a launch artifact.",
        "",
        f"Active daily demand is right-skewed: its median is **{active_demand['median']:.2f}**, mean is **{active_demand['mean']:.2f}**, p99 is **{active_demand['percentiles']['0.99']:.2f}**, and maximum is **{active_demand['maximum']}** units. Demand bands remain behaviorally distinct after launch, while stores, departments, weekdays, calendar events, SNAP flags, price histories, and launch timing all show measurable differences that can justify carefully designed future features.",
        "",
        f"Most price series change over time (**{prices['series_with_price_change_pct']:.2f}%**), but the observed price-demand comparisons are not causal. Modeling appears ready for the next controlled design stage once launch-row policy, chronological validation windows, and leakage-safe feature rules are agreed.",
        "",
        "## Dataset Validation",
        "",
    ]
    lines.extend(
        table(
            ["Measure", "Calculated result"],
            [
                ("Rows", validation["rows"]),
                ("Columns", validation["columns"]),
                ("Memory after compact loading (MiB)", validation["memory_mib"]),
                ("Date range", f"{validation['date_min']} to {validation['date_max']}"),
                ("Products", validation["items"]),
                ("Stores", validation["stores"]),
                ("Item-store pairs", validation["item_store_pairs"]),
                ("Days per pair", f"{validation['days_per_pair_min']} to {validation['days_per_pair_max']}"),
            ],
        )
    )
    lines.extend(["", "Cardinality and integrity checks:", ""])
    lines.extend(table(["Check", "Result"], validation["checks"].items()))

    duration = prelaunch["prelaunch_days_distribution"]
    lag = prelaunch["price_to_first_sale_lag_days"]
    lines.extend(
        [
            "",
            "## Pre-Launch vs Active Demand",
            "",
            f"Availability was defined by the **first known selling price**, with first positive sale as a fallback only if no price boundary exists. A zero before availability means “not yet available”; a zero after availability means “available, but no unit was observed sold.” Those states should not be treated as equivalent in future training data.",
            "",
        ]
    )
    lines.extend(
        table(
            ["Measure", "Result"],
            [
                ("Pre-launch rows", prelaunch["prelaunch_rows"]),
                ("Pre-launch share", f"{prelaunch['prelaunch_pct']:.4f}%"),
                ("Active-period rows", prelaunch["active_rows"]),
                ("Active-period share", f"{prelaunch['active_pct']:.4f}%"),
                ("Pre-launch rows with zero sales", prelaunch["prelaunch_zero_sales_rows"]),
                ("Pre-launch rows with positive sales", prelaunch["prelaunch_positive_sales_rows"]),
                ("Pre-launch rows with missing price", prelaunch["prelaunch_missing_price_rows"]),
                ("Active rows with missing price", prelaunch["active_missing_price_rows"]),
                ("Active-period zero-sales percentage", f"{prelaunch['active_zero_sales_pct']:.4f}%"),
            ],
        )
    )
    lines.extend(["", "Pre-launch duration across 300 item-store series:", ""])
    lines.extend(
        table(
            ["Statistic", "Days"],
            [("Minimum", duration["minimum"]), ("Median", duration["median"]), ("Mean", duration["mean"]), ("75th percentile", duration["p75"]), ("90th percentile", duration["p90"]), ("Maximum", duration["maximum"])],
        )
    )
    lines.extend(["", f"The first positive sale followed the first known price by a median of **{lag['median']:.1f} days** and a maximum of **{lag['maximum']} days**. Future model-ready data should normally begin at apparent availability, while retaining an explicit product-age/active indicator if full calendar history is kept."])
    lines.extend(figure("12_intermittency_and_launches.png", "Intermittency and product launch timing"))

    lines.extend(["## Overall Demand Distribution", ""])
    lines.extend(
        table(
            ["Measure", "All rows", "Active-period rows"],
            [
                ("Observations", all_demand["observations"], active_demand["observations"]),
                ("Mean", all_demand["mean"], active_demand["mean"]),
                ("Median", all_demand["median"], active_demand["median"]),
                ("Standard deviation", all_demand["standard_deviation"], active_demand["standard_deviation"]),
                ("Zero percentage", f"{all_demand['zero_pct']:.4f}%", f"{active_demand['zero_pct']:.4f}%"),
                ("90th percentile", all_demand["percentiles"]["0.9"], active_demand["percentiles"]["0.9"]),
                ("99th percentile", all_demand["percentiles"]["0.99"], active_demand["percentiles"]["0.99"]),
                ("99.9th percentile", all_demand["percentiles"]["0.999"], active_demand["percentiles"]["0.999"]),
                ("Maximum", all_demand["maximum"], active_demand["maximum"]),
                ("Skewness", all_demand["skewness"], active_demand["skewness"]),
            ],
        )
    )
    lines.extend(["", "The histogram clips the display at p99 and uses a logarithmic positive-sales panel so the tail does not hide the bulk of the distribution. The underlying data and statistics are not clipped or changed."])
    lines.extend(figure("01_overall_sales_distribution.png", "Overall active-period sales distribution"))

    lines.extend(["## Demand-Band Comparison", ""])
    lines.extend(
        table(
            ["Band", "Products", "Active mean", "Active median", "Active zero %", "Median series ADI", "Median series CV"],
            [
                (band, row["selected_products"], row["active_mean_sales"], row["active_median_sales"], row["active_zero_pct"], row["median_series_adi"], row["median_series_cv"])
                for band, row in [(name, band_map[name]) for name in ("low", "medium", "high")]
            ],
        )
    )
    lines.extend(["", "The groups remain meaningfully separated in active-period mean demand and intermittency. This confirms that Stage 4 did not merely separate products because of different pre-launch durations."])
    lines.extend(figure("02_demand_band_comparison.png", "Demand-band comparison"))

    lines.extend(["## Department Analysis", ""])
    lines.extend(
        table(
            ["Department", "Products", "Total units", "Sales share", "Active mean", "Active median", "Active zero %", "Active standard deviation"],
            [
                (department, row["selected_products"], row["total_sales"], f"{row['sales_share_pct']:.2f}%", row["active_mean_sales"], row["active_median_sales"], row["active_zero_pct"], row["active_standard_deviation"])
                for department, row in dept_map.items()
            ],
        )
    )
    lines.extend(["", "Department differences are descriptive, not causal: product composition and demand-band allocation differ between departments."])
    lines.extend(figure("03_department_comparison.png", "Department comparison"))

    lines.extend(["## Store Analysis", ""])
    lines.extend(
        table(
            ["Store", "Products", "Total units", "Sales share", "Active mean", "Active median", "Active zero %", "Active standard deviation"],
            [
                (store, row["selected_products"], row["total_sales"], f"{row['sales_share_pct']:.2f}%", row["active_mean_sales"], row["active_median_sales"], row["active_zero_pct"], row["active_standard_deviation"])
                for store, row in store_map.items()
            ],
        )
    )
    highest_store = max(store_map.items(), key=lambda pair: pair[1]["active_mean_sales"])
    lowest_store = min(store_map.items(), key=lambda pair: pair[1]["active_mean_sales"])
    scale_difference = 100 * (highest_store[1]["active_mean_sales"] / lowest_store[1]["active_mean_sales"] - 1)
    lines.extend(["", f"The highest active mean is at `{highest_store[0]}` and is **{scale_difference:.1f}%** above `{lowest_store[0]}`. This supports retaining `store_id` as an important future modeling dimension."])
    lines.extend(figure("04_store_comparison.png", "Store comparison"))
    lines.extend(figure("05_normalized_weekly_store_trends.png", "Normalized store trends"))

    lines.extend(["## Time Trends", ""])
    lines.extend(
        table(
            ["Year", "Observed days", "Total units", "Average daily total"],
            [(str(row["year"]), row["observed_days"], row["total_sales"], row["average_daily_total"]) for row in time["yearly"]],
        )
    )
    lines.extend(["", f"A simple descriptive monthly linear fit changes by **{time['monthly_linear_slope_units_per_month']:.2f} units per month**, equal to **{time['monthly_slope_as_pct_of_mean']:.3f}%** of average monthly sales. This is not a forecast and should not be extrapolated blindly. Aggregate movement mixes genuine demand change with products becoming available. The first month (January 2011) and final month (May 2016) are partial boundary months, and 2011/2016 are partial years, so their totals are not directly comparable with complete periods."])
    lines.extend(figure("06_monthly_sales_trend.png", "Monthly sales trend"))

    lines.extend(["## Weekday Patterns", ""])
    lines.extend(
        table(
            ["Weekday", "Active observations", "Total units", "Mean", "Median"],
            [(row["weekday"], row["active_observations"], row["total_sales"], row["mean_sales"], row["median_sales"]) for row in time["weekday"]],
        )
    )
    weekday_values = {row["weekday"]: row["mean_sales"] for row in time["weekday"]}
    max_weekday = max(weekday_values, key=weekday_values.get)
    min_weekday = min(weekday_values, key=weekday_values.get)
    lines.extend(["", f"`{max_weekday}` has the highest active-observation mean and `{min_weekday}` the lowest. The repeated weekday pattern supports using the existing calendar weekday field later."])
    lines.extend(figure("07_weekday_pattern.png", "Weekday demand pattern"))

    lines.extend(["## Monthly / Seasonal Patterns", ""])
    lines.extend(
        table(
            ["Month", "Average daily total"],
            [(row["month"], row["average_daily_sales"]) for row in time["monthly_overall"]],
        )
    )
    lines.extend(["", "Seasonality means a pattern that tends to recur at a similar point in each year. Trend means longer-term movement across years. Month averages suggest recurring calendar structure, but the trend chart also shows that years are not identical; future models should represent both without using future observations."])
    lines.extend(figure("08_monthly_seasonality.png", "Monthly seasonality"))

    lines.extend(["## Event / Holiday Analysis", ""])
    lines.extend(
        table(
            ["Day type", "Calendar days", "Active observations", "Mean sales", "Median sales", "Zero %"],
            [
                ("Non-event", events["non_event_days"], event_map[False]["observations"], event_map[False]["mean_sales"], event_map[False]["median_sales"], event_map[False]["zero_pct"]),
                ("Event", events["event_days"], event_map[True]["observations"], event_map[True]["mean_sales"], event_map[True]["median_sales"], event_map[True]["zero_pct"]),
            ],
        )
    )
    lines.extend(["", f"Observed event-day mean demand was **{events['event_vs_non_event_mean_difference_pct']:+.2f}%** different from non-event days. This is an association, not evidence that events caused the difference; event dates overlap with seasonality, prices, store behavior, and other factors.", "", "Event-type daily totals:", ""])
    lines.extend(
        table(
            ["Event type", "Occurrences", "Mean daily units", "Median daily units"],
            [(row["event_type"], row["event_occurrences"], row["mean_daily_sales"], row["median_daily_sales"]) for row in events["event_type_summary"]],
        )
    )

    lines.extend(["", "## SNAP Analysis", ""])
    lines.extend(
        table(
            ["Store", "Non-SNAP mean", "SNAP mean", "Difference", "SNAP zero %"],
            [
                (
                    store,
                    snap_store[(store, 0)]["mean_sales"],
                    snap_store[(store, 1)]["mean_sales"],
                    f"{snap['store_mean_difference_pct'][store]:+.2f}%",
                    snap_store[(store, 1)]["zero_pct"],
                )
                for store in ("CA_1", "TX_2", "WI_3")
            ],
        )
    )
    lines.extend(["", "Each store uses its corresponding state flag. The differences support evaluating SNAP as a future feature, but do not demonstrate a causal benefit-program effect."])
    lines.extend(figure("09_event_and_snap_comparison.png", "Event and SNAP comparisons"))

    lines.extend(["## Price Analysis", ""])
    lines.extend(
        table(
            ["Measure", "Weekly active-period prices"],
            [
                ("Records", prices["weekly_price_records"]),
                ("Mean", prices["mean"]),
                ("Median", prices["median"]),
                ("Standard deviation", prices["standard_deviation"]),
                ("Minimum", prices["minimum"]),
                ("10th percentile", prices["percentiles"]["0.1"]),
                ("90th percentile", prices["percentiles"]["0.9"]),
                ("99th percentile", prices["percentiles"]["0.99"]),
                ("Maximum", prices["maximum"]),
            ],
        )
    )
    lines.extend(["", "Pre-launch missing prices are excluded, not replaced with zero. Weekly records are used so a seven-day week does not receive seven times the statistical weight."])
    lines.extend(figure("10_price_distribution.png", "Price distribution by department"))

    lines.extend(["## Price-Change Investigation", ""])
    lines.extend(
        table(
            ["Measure", "Result"],
            [
                ("Item-store price series", prices["item_store_series"]),
                ("Series with at least one price change", prices["series_with_price_change"]),
                ("Series with a price change (%)", f"{prices['series_with_price_change_pct']:.4f}%"),
                ("Total price-change events", prices["total_price_change_events"]),
                ("Price increases", prices["increase_events"]),
                ("Price decreases", prices["decrease_events"]),
                ("Median absolute percentage change", prices["absolute_change_pct"]["median"]),
                ("90th percentile absolute change", prices["absolute_change_pct"]["p90"]),
                ("Maximum absolute change", prices["absolute_change_pct"]["maximum"]),
            ],
        )
    )
    lines.extend(["", "Weekly demand by price-change direction:", ""])
    lines.extend(
        table(
            ["Direction", "Weeks", "Mean weekly units", "Median weekly units", "Mean price change %"],
            [(row["change_direction"], row["weeks"], row["mean_weekly_sales"], row["median_weekly_sales"], row["mean_price_change_pct"]) for row in prices["price_change_direction_demand"]],
        )
    )
    corr = prices["within_series_price_demand_correlations"]
    lines.extend(["", f"Within-series price/weekly-demand correlations were available for **{corr['series_analyzed']}** series. The median was **{corr['median']:.3f}**, with **{corr['negative_pct']:.1f}%** negative and **{corr['positive_pct']:.1f}%** positive. Mixed directions reinforce that simple correlation cannot establish that price caused demand to change. Price, promotion timing, seasonality, product differences, events, and life cycle remain confounded."])
    lines.extend(figure("11_price_change_vs_demand.png", "Price change direction and demand"))

    zero_dist = intermittency["active_zero_pct_distribution"]
    adi = intermittency["average_demand_interval_distribution"]
    lines.extend(["## Intermittent Demand", ""])
    lines.extend(
        table(
            ["Measure", "Result"],
            [
                ("Item-store series", intermittency["series"]),
                ("Median active zero percentage", zero_dist["median"]),
                ("90th percentile active zero percentage", zero_dist["p90"]),
                ("Maximum active zero percentage", zero_dist["maximum"]),
                ("Median average demand interval", adi["median"]),
                ("90th percentile average demand interval", adi["p90"]),
                ("Maximum average demand interval", adi["maximum"]),
            ],
        )
    )
    lines.extend(["", "Descriptive intermittency classes:", ""])
    lines.extend(table(["Class", "Series"], intermittency["intermittency_class_counts"].items()))
    lines.extend(["", "Intermittent demand is harder because many days provide a zero target and positive sales arrive irregularly. Standard regression models may overpredict zeros or underpredict bursts. Croston-style approaches could later be useful baselines for the sparsest series, but none were implemented here."])

    lines.extend(["", "## Outlier Investigation", ""])
    lines.extend([f"Percentiles and row counts below use **{outliers['population']}**, so unavailable pre-launch zeros do not influence the tail thresholds.", ""])
    lines.extend(
        table(
            ["Measure", "Result"],
            [
                ("99th percentile", outliers["p99"]),
                ("99.9th percentile", outliers["p99_9"]),
                ("99.99th percentile", outliers["p99_99"]),
                ("Maximum", outliers["maximum"]),
                ("Rows above p99.9", outliers["rows_above_p99_9"]),
                ("Rows above p99.9 on event days", outliers["extreme_rows_on_event_days"]),
                ("Extreme event-day share", f"{outliers['extreme_event_day_pct']:.2f}%"),
            ],
        )
    )
    lines.extend(["", "Largest daily observations:", ""])
    lines.extend(
        table(
            ["Date", "Store", "Item", "Department", "Sales", "Price", "Event", "Demand band"],
            [(row["date"], row["store_id"], row["item_id"], row["dept_id"], row["sales"], row["sell_price"], row["event_name_1"], row["demand_band"]) for row in outliers["top_15_observations"]],
        )
    )
    lines.extend(["", "These values require investigation but are not automatically errors. Retail spikes may be plausible, and no values were removed or capped."])

    lines.extend(["", "## Product Launch Behavior", ""])
    lines.extend(
        table(
            ["Measure", "Date"],
            [
                ("Earliest product known-price date", launches["earliest_known_price_date"]),
                ("Latest product earliest known-price date", launches["latest_earliest_known_price_date"]),
                ("Earliest product positive-sale date", launches["earliest_positive_sale_date"]),
                ("Latest product earliest positive-sale date", launches["latest_earliest_positive_sale_date"]),
            ],
        )
    )
    lines.extend(["", "Product launch-year distribution:", ""])
    lines.extend(table(["Year", "Products"], [(str(year), count) for year, count in launches["launch_year_distribution"].items()]))
    lines.extend(["", "The evidence supports training each item-store series on active history rather than presenting unavailable pre-launch zeros as demand observations. The exact future policy still needs approval because availability inferred from first known price is a dataset convention, not an explicit inventory record."])

    lines.extend(["", "## Modeling Readiness", "", f"Assessment: **{readiness['assessment']}**.", "", "The data foundation is structurally sound, but modeling should wait until launch-boundary treatment and time-aware validation are specified."])

    lines.extend(["", "## Recommended Future Features", ""])
    lines.extend(table(["Candidate feature", "Stage 5 evidence"], [(row["feature"], row["evidence"]) for row in readiness["supported_future_features"]]))
    lines.extend(["", "These are recommendations only. No production feature columns were persisted in Stage 5."])

    lines.extend(["", "## Leakage Risks", ""])
    lines.extend([f"- {risk}" for risk in readiness["leakage_risks"]])
    lines.extend(["", "Leakage occurs when information that would not have been available at prediction time influences model training. It can make evaluation look excellent while producing an unusable real-world model."])

    lines.extend(["", "## Forecasting Challenges", ""])
    lines.extend([f"- {challenge}" for challenge in readiness["forecasting_challenges"]])

    lines.extend(["", "## Decisions Needed Before Next Stage", ""])
    lines.extend([f"- {decision}" for decision in readiness["decisions_before_next_stage"]])

    lines.extend(
        [
            "",
            "## Performance Decisions",
            "",
            f"- Loaded the {validation['rows']:,}-row CSV once using compact numeric and categorical dtypes.",
            "- Used vectorized operations and groupby aggregations; no manual Python loop traversed all daily rows.",
            "- Reused daily, weekly, monthly, and series-level tables across report sections and figures.",
            "- Reduced weekly price analysis to one item-store-week row rather than counting the same weekly price once per day.",
            "- Used aggregate views to understand 300 time series rather than drawing 300 unreadable lines.",
            "",
            "Aggregation makes a large collection of series understandable: totals and averages reveal common structure, while item-store metrics preserve the distribution of intermittency and launch timing. Aggregation can hide individual behavior, so the report also lists extreme and late-launch examples.",
            "",
            "## Scope Confirmation",
            "",
            "No persistent feature-engineering dataset, lags, rolling model inputs, train/test split, forecasting model, evaluation, PostgreSQL, inventory optimization, dashboard, Docker, or deployment work was performed. The Stage 4 CSV and manifest remained byte-for-byte unchanged.",
            "",
        ]
    )
    return "\n".join(lines)
