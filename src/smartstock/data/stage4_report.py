"""Render the Stage 4 transformation summary as a learner-friendly report."""

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


def render_stage4_report(summary: dict[str, Any]) -> str:
    """Render all required Stage 4 report sections."""

    eligibility = summary["eligibility"]
    demand = summary["demand_stratification"]
    selection = summary["selection"]
    transformation = summary["transformation"]
    calendar_join = summary["calendar_join"]
    price_join = summary["price_join"]
    missing = summary["missing_prices"]
    integrity = summary["integrity"]
    thresholds = demand["thresholds"]

    lines: list[str] = [
        "# SmartStock Stage 4 — Version 1 Subset Creation & Controlled Data Transformation",
        "",
        f"Generated at `{summary['created_at_utc']}` from the local official M5 raw files.",
        "",
        "> Stage 4 selected a reproducible 100-item FOODS subset, transformed only its 300 item-store rows to daily long format, and joined raw calendar and weekly price attributes. No forecasting features or price imputation were added.",
        "",
        "## Executive Summary",
        "",
        f"Version 1 contains **{selection['selected_items']:,} FOODS items** across **{', '.join(summary['stores'])}**, producing **{selection['item_store_pairs']:,} item-store series** and **{transformation['final_rows']:,} daily rows** from {transformation['date_min']} through {transformation['date_max']}. The tracked manifest fixes the item list and seed; the generated working CSV remains Git-ignored under `data/interim/`.",
        "",
        f"All calendar keys matched and neither join multiplied rows. Prices remain missing in **{missing['missing_rows']:,} rows ({missing['missing_pct']:.2f}%)**; those values were deliberately not filled.",
        "",
        "## Store Selection",
        "",
        "The frozen cross-state stores were used exactly as specified:",
        "",
    ]
    lines.extend(table(["State", "Store"], [("California", "CA_1"), ("Texas", "TX_2"), ("Wisconsin", "WI_3")]))

    failures = eligibility["independent_failure_counts"]
    lines.extend(
        [
            "",
            "## Eligibility Filtering",
            "",
            f"The starting population contained **{eligibility['starting_food_items']:,} FOODS items**. **{eligibility['eligible_items']:,}** passed all four rules and **{eligibility['ineligible_items']:,}** failed at least one. **{eligibility['items_failing_multiple_rules']:,}** items failed more than one rule.",
            "",
            "Rule failures are counted independently, so a product failing multiple rules appears in multiple rows. These counts must not be added as though filtering were sequential.",
            "",
        ]
    )
    lines.extend(
        table(
            ["Eligibility rule", "Independent failures", "Exact policy"],
            [
                ("A — Required stores", failures["required_stores"], "Structural sales row in CA_1, TX_2, and WI_3"),
                ("B — Extreme sparsity", failures["extreme_sparsity"], "Aggregated selected-store zero prevalence must be < 95%"),
                ("C — Useful history", failures["insufficient_remaining_history"], "1941 − earliest positive-sale day must be at least 730"),
                ("D — Price coverage", failures["insufficient_active_period_price_coverage"], "At least 70% active-period weekly price coverage in each selected store"),
            ],
        )
    )
    lines.extend(
        [
            "",
            "Price coverage is measured separately for every item-store pair. Its denominator is the distinct calendar weeks from that store's first positive sale through `d_1941`; weeks before launch are not penalized. Each of the three stores must reach 70%.",
            "",
            "## Demand Stratification",
            "",
            "The demand metric is **mean daily units across all three selected stores and all 1,941 days**:",
            "",
            "```text",
            "mean_daily_demand = total selected-store units / (3 × 1,941)",
            "```",
            "",
            "Tertiles were calculated from the eligible FOODS population. Boundary policy: low includes the lower threshold; medium is above the lower and includes the upper threshold; high is above the upper threshold.",
            "",
        ]
    )
    lines.extend(
        table(
            ["Band", "Mean daily demand definition", "Eligible items"],
            [
                ("Low", f"≤ {thresholds['low_upper_inclusive']:.10f}", demand["eligible_band_counts"].get("low", 0)),
                ("Medium", f"> {thresholds['low_upper_inclusive']:.10f} and ≤ {thresholds['medium_upper_inclusive']:.10f}", demand["eligible_band_counts"].get("medium", 0)),
                ("High", f"> {thresholds['medium_upper_inclusive']:.10f}", demand["eligible_band_counts"].get("high", 0)),
            ],
        )
    )
    lines.extend(["", "### Quota availability before sampling", ""])
    lines.extend(
        table(
            ["Department", "Demand band", "Available eligible items", "Required"],
            [
                (row["department"], row["demand_band"], row["available"], row["required"])
                for row in demand["quota_availability"]
            ],
        )
    )

    lines.extend(["", "## Final Product Selection", ""])
    lines.extend(
        table(
            ["Department", "Selected items"],
            selection["department_counts"].items(),
        )
    )
    lines.append("")
    lines.extend(
        table(
            ["Demand band", "Selected items"],
            selection["demand_band_counts"].items(),
        )
    )
    lines.extend(
        [
            "",
            f"Selection used `random_seed = {selection['random_seed']}` inside each department × demand-band cell. Sorting each candidate pool by `item_id` before sampling ensures the same source data produces the same 100 IDs. The exact list is stored in `config/v1_subset.json`.",
            "",
            "Stratified sampling preserves low-, medium-, and high-demand products rather than choosing only best sellers. This creates a more realistic forecasting challenge because retail portfolios contain intermittent and slow-moving items as well as fast movers.",
            "",
            "## Transformation",
            "",
            "```text",
            "1,437 selected-store FOODS candidates",
            "    -> eligibility + stratified selection",
            "100 items × 3 stores = 300 wide rows",
            "    -> melt d_1 ... d_1941",
            f"{transformation['final_rows']:,} daily rows",
            "    -> calendar join on d",
            "    -> price join on store_id + item_id + wm_yr_wk",
            "```",
            "",
            "Filtering before melting is the key performance decision. Melting the full M5 sales table would create about 59.2 million rows; melting 300 selected rows creates only 582,300.",
            "",
            "## Final Dataset",
            "",
        ]
    )
    lines.extend(
        table(
            ["Measure", "Result"],
            [
                ("Output", transformation["output_path"]),
                ("Rows", transformation["final_rows"]),
                ("Columns", transformation["final_columns"]),
                ("Unique items", selection["selected_items"]),
                ("Stores", ", ".join(summary["stores"])),
                ("Item-store pairs", selection["item_store_pairs"]),
                ("Date range", f"{transformation['date_min']} to {transformation['date_max']}"),
                ("In-memory size (MiB)", transformation["dataframe_memory_mib"]),
                ("CSV size (MiB)", transformation["csv_file_size_mib"]),
            ],
        )
    )

    lines.extend(["", "## Missing Prices", ""])
    lines.extend(
        table(
            ["Measure", "Rows"],
            [
                ("Total missing prices", missing["missing_rows"]),
                ("Missing percentage", f"{missing['missing_pct']:.6f}%"),
                ("Before first known price", missing["before_first_known_price_rows"]),
                ("On/after first known price", missing["on_or_after_first_known_price_rows"]),
                ("After last known price", missing["after_last_known_price_rows"]),
                ("Before first positive sale", missing["before_first_positive_sale_rows"]),
                ("On/after first positive sale", missing["on_or_after_first_positive_sale_rows"]),
                ("On/after launch with positive sales", missing["on_or_after_first_positive_with_positive_sales_rows"]),
                ("Products with at least one missing price", missing["products_with_missing_prices"]),
                ("Products without missing prices", missing["products_without_missing_prices"]),
            ],
        )
    )
    lines.extend(["", "### Missing prices by store", ""])
    lines.extend(
        table(
            ["Store", "Rows", "Missing", "Missing percentage"],
            [
                (row["store_id"], row["total_rows"], row["missing_rows"], f"{row['missing_pct']:.4f}%")
                for row in missing["by_store"]
            ],
        )
    )
    lines.extend(["", "### Missing prices by selected product", ""])
    lines.extend(
        table(
            ["Item", "Rows", "Missing", "Missing percentage"],
            [
                (row["item_id"], row["total_rows"], row["missing_rows"], f"{row['missing_pct']:.4f}%")
                for row in missing["by_product"]
            ],
        )
    )
    lines.extend(
        [
            "",
            "Missing prices remain `NaN`. Replacing an unknown price with zero would falsely imply the product was free, while forward-fill or mean imputation would introduce assumptions that require a later cleaning decision.",
            "",
            "## Validation",
            "",
        ]
    )
    lines.extend(table(["Integrity check", "Result"], integrity["checks"].items()))
    lines.extend(
        [
            "",
            f"Calendar join: {calendar_join['rows_before']:,} rows before and {calendar_join['rows_after']:,} after; {calendar_join['unmatched_rows']:,} unmatched; multiplier {calendar_join['row_multiplier']:.1f}.",
            "",
            f"Price join: {price_join['rows_before']:,} rows before and {price_join['rows_after']:,} after; {price_join['duplicate_price_keys']:,} duplicate source keys; multiplier {price_join['row_multiplier']:.1f}.",
            "",
            f"Raw source SHA-256 hashes were equal before and after execution: **{summary['raw_files']['unchanged']}**.",
            "",
            "## Decisions and Trade-offs",
            "",
            "- **Filter before melting:** avoids a needless 59.2-million-row intermediate and keeps laptop memory use controlled.",
            "- **Stratified selection:** gives all FOODS departments and all demand bands explicit representation; pure high-volume sampling would be biased.",
            "- **Fixed seed 42:** pseudorandom selection remains reproducible while avoiding manual cherry-picking.",
            "- **Composite price key:** week alone repeats across all products and stores. `(store_id, item_id, wm_yr_wk)` identifies the intended weekly price without multiplying rows.",
            "- **Missing prices preserved:** zero is a real price value with a different meaning from unknown; no unsupported imputation was introduced.",
            "- **Raw data immutable:** transformations can always be reproduced or corrected because official source files remain unchanged.",
            f"- **Safe dtype reduction:** repeated identifiers and labels used pandas categorical dtype in memory, reducing the final working DataFrame to {transformation['dataframe_memory_mib']:.2f} MiB. Their CSV values remain ordinary text.",
            "",
            "## Scope Confirmation",
            "",
            "No EDA, database work, feature engineering, price-change or promotion feature creation, holiday features, train/test splitting, modeling, forecasting, inventory logic, simulation, dashboard, Docker, or deployment work was performed.",
            "",
        ]
    )
    return "\n".join(lines)
