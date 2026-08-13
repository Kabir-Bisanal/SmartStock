"""Render the calculated Stage 3 M5 profile as beginner-readable Markdown."""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def display(value: Any) -> str:
    """Format profile values consistently for the Markdown report."""

    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    """Return a compact Markdown table as a list of lines."""

    def escape(value: Any) -> str:
        return display(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(escape(value) for value in row) + " |" for row in rows)
    return lines


def counts_table(counts: dict[str, int], label: str = "Value") -> list[str]:
    """Render a value-count mapping."""

    return markdown_table([label, "Count"], counts.items())


def render_stage3_report(profile: dict[str, Any]) -> str:
    """Render all required Stage 3 report sections from calculated values."""

    sales = profile["sales"]
    sales_values = sales["sales_values"]
    prices = profile["prices"]
    calendar = profile["calendar"]
    hierarchy = profile["hierarchy"]
    relationships = profile["relationships"]
    lines: list[str] = [
        "# SmartStock Stage 3 — M5 Data Understanding & Profiling",
        "",
        f"Generated from the local raw M5 files at `{profile['generated_at_utc']}`.",
        "",
        "> This is a structural and data-quality profile, not an exploratory analysis, cleaning pipeline, final subset, or production join. The raw files were read but not modified.",
        "",
        "## Executive Summary",
        "",
        f"The sales table contains **{sales['rows']:,} item-store rows**, **{sales['daily_column_count']:,} daily columns**, **{sales['unique_items']:,} items**, and **{sales['unique_stores']:,} stores**. Its day columns cover `{sales['first_daily_column']}` through `{sales['last_daily_column']}`. The price table contains **{prices['rows']:,} weekly item-store price records** across **{prices['unique_weeks']:,} weeks**. The calendar maps day labels and weeks to **{calendar['rows']:,} dates** from **{calendar['date_min']} through {calendar['date_max']}**.",
        "",
        f"All {relationships['sales_calendar']['sales_day_columns']:,} sales day labels match calendar day keys, all {relationships['prices_calendar']['price_weeks']:,} price weeks match calendar weeks, and all {relationships['sales_prices']['matched_item_store_pairs']:,} sales item-store combinations have at least one price record. The composite price key `(store_id, item_id, wm_yr_wk)` is {'unique' if relationships['sales_prices']['price_composite_key_is_unique'] else 'not unique'}.",
        "",
        f"Demand is strongly zero-heavy: **{sales_values['zero_prevalence_pct']:.2f}%** of daily cells contain zero. This is expected retail behavior, not automatically missing data. The report proposes three subset strategies but deliberately does not select the final stores or products.",
        "",
        "## Sales Dataset",
        "",
        "### Structure and cardinality",
        "",
    ]
    lines.extend(
        markdown_table(
            ["Measure", "Calculated value"],
            [
                ("File", "sales_train_evaluation.csv"),
                ("CSV file size (MiB)", sales["file_size_mib"]),
                ("Rows", sales["rows"]),
                ("Columns", sales["columns"]),
                ("Identifier columns", ", ".join(sales["identifier_columns"])),
                ("Daily sales columns", sales["daily_column_count"]),
                ("First / last daily column", f"{sales['first_daily_column']} / {sales['last_daily_column']}"),
                ("Daily sequence contiguous", sales["daily_columns_contiguous"]),
                ("Unique items", sales["unique_items"]),
                ("Unique stores", sales["unique_stores"]),
                ("Unique states", sales["unique_states"]),
                ("Unique categories", sales["unique_categories"]),
                ("Unique departments", sales["unique_departments"]),
                ("Unique item-store combinations", sales["unique_item_store_combinations"]),
                ("`id` unique", sales["id_is_unique"]),
                ("`(item_id, store_id)` unique", sales["item_store_is_unique"]),
                ("Default-inferred daily dtype", ", ".join(sales["source_inferred_daily_dtypes"])),
                ("Estimated default DataFrame memory (MiB)", sales["estimated_default_dataframe_memory_mib"]),
                ("Estimated int32 DataFrame memory (MiB)", sales["estimated_int32_dataframe_memory_mib"]),
            ],
        )
    )
    lines.extend(
        [
            "",
            "**Wide format** means that each item-store series is one row and each date is a separate column. This makes the raw file compact on disk but awkward for normal relational joins and potentially expensive to reshape.",
            "",
            "**Cardinality** means the number of distinct values in a field or key. A **composite key** uses more than one column; here `(item_id, store_id)` uniquely identifies each sales series.",
            "",
            "### Identifier missing values",
            "",
        ]
    )
    lines.extend(counts_table(sales["identifier_missing_values"], "Identifier"))
    lines.extend(["", "### Category counts", ""])
    lines.extend(
        markdown_table(
            ["Category", "Sales rows", "Unique items"],
            [
                (category, sales["category_row_counts"][category], item_count)
                for category, item_count in sales["category_item_counts"].items()
            ],
        )
    )
    lines.extend(["", "### Department counts", ""])
    lines.extend(
        markdown_table(
            ["Department", "Sales rows", "Unique items"],
            [
                (department, sales["department_row_counts"][department], item_count)
                for department, item_count in sales["department_item_counts"].items()
            ],
        )
    )
    lines.extend(["", "### Stores and states", ""])
    lines.extend(
        markdown_table(
            ["Store", "Item-store rows"],
            sales["store_row_counts"].items(),
        )
    )
    lines.extend([""])
    lines.extend(counts_table(sales["state_row_counts"], "State"))
    lines.extend(["", "### Daily sales-value validation", ""])
    lines.extend(
        markdown_table(
            ["Measure", "Calculated value"],
            [
                ("Daily cells scanned", sales_values["total_values"]),
                ("Missing daily values", sales_values["missing_values"]),
                ("Negative sales values", sales_values["negative_values"]),
                ("Zero sales values", sales_values["zero_values"]),
                ("Zero prevalence", f"{sales_values['zero_prevalence_pct']:.4f}%"),
                ("Minimum", sales_values["minimum"]),
                ("Maximum", sales_values["maximum"]),
                ("Profiler scan dtype", ", ".join(sales_values["profiling_dtypes"])),
                ("Items at least 95% zero", sales["items_at_least_95_pct_zero"]),
                ("FOODS items at least 95% zero", sales["foods_items_at_least_95_pct_zero"]),
            ],
        )
    )
    lines.extend(
        [
            "",
            "The CSV has no stored datatype; pandas normally infers these integer-valued day columns as `int64`. The profiler reads them as `int32` in chunks after validating their observed range. This reduces temporary memory without changing values.",
            "",
            "### Store-level sales structure",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Store", "State", "Items", "Total units", "Zero prevalence"],
            [
                (
                    row["store_id"],
                    row["state_id"],
                    row["unique_items"],
                    row["total_units"],
                    f"{row['zero_prevalence_pct']:.2f}%",
                )
                for row in sales["store_sales_summary"]
            ],
        )
    )

    lines.extend(["", "## Price Dataset", ""])
    lines.extend(
        markdown_table(
            ["Measure", "Calculated value"],
            [
                ("File", "sell_prices.csv"),
                ("CSV file size (MiB)", prices["file_size_mib"]),
                ("Rows", prices["rows"]),
                ("Columns", prices["columns"]),
                ("Optimized DataFrame memory (MiB)", prices["optimized_memory_mib"]),
                ("Unique stores", prices["unique_stores"]),
                ("Unique items", prices["unique_items"]),
                ("Unique item-store pairs", prices["unique_item_store_pairs"]),
                ("Unique composite price records", prices["unique_composite_price_records"]),
                ("Unique weeks", prices["unique_weeks"]),
                ("Earliest / latest week", f"{prices['earliest_week']} / {prices['latest_week']}"),
                ("Minimum price", prices["minimum_price"]),
                ("Maximum price", prices["maximum_price"]),
                ("Mean price", prices["mean_price"]),
                ("Median price", prices["median_price"]),
                ("Negative prices", prices["negative_prices"]),
                ("Zero prices", prices["zero_prices"]),
                ("Duplicate rows", prices["duplicate_rows"]),
                ("Duplicate composite keys", prices["duplicate_composite_keys"]),
                ("Possible raw item-store-week positions", prices["possible_raw_week_grid_positions"]),
                ("Missing raw item-store-week positions", prices["missing_raw_week_grid_positions"]),
                ("Overall raw week-grid coverage", f"{prices['overall_raw_week_grid_coverage_pct']:.2f}%"),
            ],
        )
    )
    lines.extend(["", "### Column datatypes", ""])
    lines.extend(
        markdown_table(
            ["Column", "Default inference", "Profiler dtype"],
            [
                (
                    column,
                    prices["source_inferred_dtypes"][column],
                    prices["profiling_dtypes"][column],
                )
                for column in ["store_id", "item_id", "wm_yr_wk", "sell_price"]
            ],
        )
    )
    lines.extend(["", "### Missing values", ""])
    lines.extend(counts_table(prices["missing_values"], "Column"))
    lines.extend(["", "### Price coverage by store", ""])
    lines.extend(
        markdown_table(
            ["Store", "Records", "Items", "Weeks", "Median weeks/item", "Raw week-grid coverage"],
            [
                (
                    row["store_id"],
                    row["price_records"],
                    row["unique_items"],
                    row["unique_weeks"],
                    row["median_weeks_per_item"],
                    f"{row['raw_week_grid_coverage_pct']:.2f}%",
                )
                for row in prices["store_coverage"]
            ],
        )
    )
    lines.extend(
        [
            "",
            "**Raw week-grid coverage** divides observed price rows by every possible item × calendar-week position for the relevant group. It is intentionally conservative: a pre-launch week is counted as uncovered, even when the absence is legitimate.",
            "",
            "### Price coverage by category",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Category", "Records", "Items", "Stores", "Weeks", "Raw week-grid coverage"],
            [
                (
                    row["cat_id"],
                    row["price_records"],
                    row["unique_items"],
                    row["unique_stores"],
                    row["unique_weeks"],
                    f"{row['raw_week_grid_coverage_pct']:.2f}%",
                )
                for row in prices["category_coverage"]
            ],
        )
    )
    distinct = prices["distinct_prices_per_item"]
    lines.extend(
        [
            "",
            "### Distinct prices per item",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Measure", "Calculated value"],
            [
                ("Minimum", distinct["minimum"]),
                ("Median", distinct["median"]),
                ("Mean", distinct["mean"]),
                ("Maximum", distinct["maximum"]),
                ("Items with one distinct price", distinct["items_with_one_distinct_price"]),
                ("Items with price changes", distinct["items_with_price_changes"]),
            ],
        )
    )

    lines.extend(["", "## Calendar Dataset", ""])
    lines.extend(
        markdown_table(
            ["Measure", "Calculated value"],
            [
                ("Rows", calendar["rows"]),
                ("Columns", calendar["columns"]),
                ("Memory (MiB)", calendar["memory_mib"]),
                ("Date range", f"{calendar['date_min']} to {calendar['date_max']}"),
                ("Unique `d` values", calendar["unique_d"]),
                ("`d` is unique", calendar["d_is_unique"]),
                ("Unique weeks", calendar["unique_weeks"]),
                ("Week range", f"{calendar['week_min']} to {calendar['week_max']}"),
                ("Years", ", ".join(str(value) for value in calendar["years"])),
                ("Months", ", ".join(str(value) for value in calendar["months"])),
                ("Weekdays", ", ".join(calendar["weekdays"])),
                ("Days with at least one event", calendar["event_days"]),
                ("Days with two events", calendar["two_event_days"]),
            ],
        )
    )
    lines.extend(["", "### Missing values", ""])
    lines.extend(counts_table(calendar["missing_values"], "Column"))
    lines.extend(["", "### Event-field value counts", ""])
    for column in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        lines.extend([f"#### `{column}`", ""])
        lines.extend(counts_table(calendar["event_value_counts"][column]))
        lines.append("")
    lines.extend(["### Combined event-type distribution", ""])
    lines.extend(counts_table(calendar["event_type_distribution"], "Event type"))
    lines.extend(["", "### SNAP flag coverage", ""])
    lines.extend(
        markdown_table(
            ["State", "SNAP-enabled days", "Calendar coverage"],
            [
                (state, values["enabled_days"], f"{values['coverage_pct']:.2f}%")
                for state, values in calendar["snap_coverage"].items()
            ],
        )
    )

    lines.extend(
        [
            "",
            "## Retail Hierarchy",
            "",
            "The real data demonstrates two linked dimensions rather than one single chain:",
            "",
            "```text",
            "Geography: state_id -> store_id",
            "Product:   cat_id -> dept_id -> item_id",
            "Sales:     store_id × item_id × daily columns",
            "```",
            "",
            "An item does not belong to a store in the product hierarchy; the sales table crosses the product dimension with the store dimension.",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Verified mapping", "Result"],
            [
                ("Each store maps to one state", hierarchy["store_maps_to_one_state"]),
                ("Each department maps to one category", hierarchy["department_maps_to_one_category"]),
                ("Each item maps to one department", hierarchy["item_maps_to_one_department"]),
                ("Each item maps to one category", hierarchy["item_maps_to_one_category"]),
            ],
        )
    )
    lines.extend(["", "Stores per state:", ""])
    lines.extend(counts_table(hierarchy["stores_per_state"], "State"))

    lines.extend(["", "## FOODS Category Analysis", ""])
    lines.extend(
        markdown_table(
            ["Measure", "Calculated value"],
            [
                ("Unique FOODS items", hierarchy["foods_unique_items"]),
                ("FOODS departments", ", ".join(hierarchy["foods_departments"])),
                ("FOODS items appearing in multiple stores", hierarchy["foods_items_in_multiple_stores"]),
                ("FOODS items appearing in every store", hierarchy["foods_items_in_every_store"]),
                ("Minimum stores per FOODS item", hierarchy["minimum_stores_per_food_item"]),
                ("Maximum stores per FOODS item", hierarchy["maximum_stores_per_food_item"]),
                ("FOODS items first positive after day 365", sales["foods_items_first_positive_after_day_365"]),
                ("FOODS items never positive", sales["foods_items_never_positive"]),
            ],
        )
    )
    lines.extend(["", "### FOODS products per department", ""])
    lines.extend(counts_table(hierarchy["foods_items_by_department"], "Department"))
    lines.extend(["", "### FOODS product availability by store", ""])
    lines.extend(counts_table(hierarchy["foods_items_by_store"], "Store"))
    lines.extend(
        [
            "",
            "Every FOODS item has a sales row in every store. This verifies structural availability in the sales matrix, but it does not prove continuous selling or price coverage. Some item-store series start selling late or contain mostly zeros.",
            "",
            "### Highest-zero FOODS items",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Item", "Zero prevalence", "First positive day", "Total units"],
            [
                (
                    row["item_id"],
                    f"{row['zero_prevalence_pct']:.2f}%",
                    row["first_positive_day"],
                    row["total_units"],
                )
                for row in sales["highest_zero_food_items"]
            ],
        )
    )

    sales_calendar = relationships["sales_calendar"]
    prices_calendar = relationships["prices_calendar"]
    sales_prices = relationships["sales_prices"]
    lines.extend(
        [
            "",
            "## Dataset Relationships",
            "",
            "### Sales day columns to calendar",
            "",
            "```text",
            "sales.d_x <-> calendar.d -> calendar.date and calendar.wm_yr_wk",
            "```",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Check", "Result"],
            [
                ("Sales day columns", sales_calendar["sales_day_columns"]),
                ("Matched sales day labels", sales_calendar["matched_sales_days"]),
                ("Unmatched sales day labels", len(sales_calendar["unmatched_sales_days"])),
                ("Calendar labels beyond sales", sales_calendar["calendar_days_not_in_sales"]),
                ("Observed sales date range", f"{sales_calendar['sales_date_min']} to {sales_calendar['sales_date_max']}"),
                ("Cardinality", sales_calendar["key_cardinality"]),
            ],
        )
    )
    lines.extend(
        [
            "",
            "### Weekly prices to calendar",
            "",
            "```text",
            "prices.wm_yr_wk <-> calendar.wm_yr_wk",
            "```",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Check", "Result"],
            [
                ("Price weeks", prices_calendar["price_weeks"]),
                ("Matched price weeks", prices_calendar["matched_price_weeks"]),
                ("Unmatched price weeks", len(prices_calendar["unmatched_price_weeks"])),
                ("Calendar weeks without prices", len(prices_calendar["calendar_weeks_without_prices"])),
                ("Calendar days per week", f"{prices_calendar['calendar_days_per_week_min']} to {prices_calendar['calendar_days_per_week_max']}"),
                ("Cardinality", prices_calendar["key_cardinality"]),
            ],
        )
    )
    lines.extend(
        [
            "",
            "Joining prices directly to calendar on week alone is technically many-to-many because both tables repeat the week. The safe later sequence is to map each sales day to its week, then join one price row using `(store_id, item_id, wm_yr_wk)`.",
            "",
            "### Sales identifiers to prices",
            "",
            "```text",
            "sales.(store_id, item_id) <-> prices.(store_id, item_id)",
            "prices unique row key: (store_id, item_id, wm_yr_wk)",
            "```",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Check", "Result"],
            [
                ("Matched items", sales_prices["matched_items"]),
                ("Sales items without prices", len(sales_prices["sales_items_without_prices"])),
                ("Price items without sales", len(sales_prices["price_items_without_sales"])),
                ("Matched stores", sales_prices["matched_stores"]),
                ("Sales stores without prices", len(sales_prices["sales_stores_without_prices"])),
                ("Price stores without sales", len(sales_prices["price_stores_without_sales"])),
                ("Matched item-store pairs", sales_prices["matched_item_store_pairs"]),
                ("Sales pairs without prices", sales_prices["sales_item_store_pairs_without_prices"]),
                ("Price pairs without sales", sales_prices["price_item_store_pairs_without_sales"]),
                ("Composite price key unique", sales_prices["price_composite_key_is_unique"]),
                ("Cardinality", sales_prices["key_cardinality"]),
            ],
        )
    )

    lines.extend(["", "## Data Quality Observations", ""])
    lines.extend(
        markdown_table(
            ["Classification", "Observation", "Why it matters"],
            [
                (row["classification"], row["observation"], row["implication"])
                for row in profile["quality_observations"]
            ],
        )
    )

    lines.extend(
        [
            "",
            "## Version 1 Subset Recommendations",
            "",
            "These are candidate strategies, not a final subset. Product-screen counts use transparent diagnostics: a product must have a sales row in all three stores; ‘active by day 365’ means positive demand occurred in each store within the first year; ‘below 95% zero’ is checked in each store; and 80% price coverage is measured against all calendar weeks. That price denominator penalizes legitimate late introductions, so it is a conservative screen rather than a cleaning rule.",
            "",
            "Selecting only the highest-volume products would bias the project toward fast movers and make results less representative. A final ~100-item subset should therefore be stratified across `FOODS_1`, `FOODS_2`, and `FOODS_3`, and ideally across demand/coverage bands rather than volume alone.",
            "",
        ]
    )
    for strategy in profile["subset_strategies"]:
        screen = strategy["product_screen"]
        lines.extend(
            [
                f"### {strategy['name']}",
                "",
                f"Candidate stores: **{', '.join(strategy['stores'])}**",
                "",
                f"Rationale: {strategy['rationale']}",
                "",
                f"Trade-off: {strategy['tradeoff']}",
                "",
            ]
        )
        lines.extend(
            markdown_table(
                ["Store", "State", "Raw price coverage", "Sales zero prevalence"],
                [
                    (
                        row["store_id"],
                        row["state_id"],
                        f"{row['raw_week_grid_coverage_pct']:.2f}%",
                        f"{row['zero_prevalence_pct']:.2f}%",
                    )
                    for row in strategy["store_evidence"]
                ],
            )
        )
        lines.extend(["", "Indicative FOODS product screen:", ""])
        lines.extend(
            markdown_table(
                ["Screen", "Items passing"],
                [
                    ("Present in all three sales stores", screen["foods_items_in_all_sales_stores"]),
                    ("Any price in all three stores", screen["foods_items_with_any_price_in_all_stores"]),
                    ("At least 80% raw price-week coverage in every store", screen["foods_items_with_at_least_80_pct_raw_week_coverage_in_all_stores"]),
                    ("Active by day 365 in every store", screen["foods_items_active_by_day_365_in_all_stores"]),
                    ("Below 95% zero in every store", screen["foods_items_below_95_pct_zero_in_all_stores"]),
                    ("All indicative screens", screen["foods_items_meeting_all_indicative_screens"]),
                ],
            )
        )
        lines.append("")

    performance = profile["performance"]
    lines.extend(
        [
            "## Performance and Memory Decisions",
            "",
            f"- Avoided a full sales melt. Converting {sales['rows']:,} rows × {sales['daily_column_count']:,} days to long format would create {sales_values['total_values']:,} sales rows before calendar or price columns are added.",
            f"- Loaded only the six sales identifier columns as one small DataFrame, then scanned daily columns once in chunks of {performance['sales_chunk_size_rows']:,} rows.",
            "- Used `int32` for the sales scan after checking the observed value range, rather than pandas' larger default `int64` inference.",
            "- Loaded the price table once with categorical identifiers, `int32` weeks, and `float32` prices; aggregate tables were calculated from that copy.",
            "- Did not calculate expensive per-day trends, correlations, promotion effects, or full descriptive statistics because they belong to EDA and add little structural evidence here.",
            "",
            "## Risks / Things to Investigate Later",
            "",
            "- Decide whether price absence before first sale represents product launch timing, store assortment timing, or another M5 convention before imputing anything.",
            "- Define the final subset's product coverage rule and department allocation with ChatGPT before extraction.",
            "- Preserve chronological ordering and product launch timing during later train/test design.",
            "- Design the Stage 4 wide-to-long process in bounded chunks; a naive melt would create tens of millions of rows.",
            "- Treat calendar days beyond `d_1941` as future-horizon support, not missing sales rows.",
            "- Verify how SNAP flags should be interpreted for each store state before feature engineering.",
            "",
            "## Scope Confirmation",
            "",
            "No raw data was cleaned or modified. No permanent wide-to-long transformation, final subset, PostgreSQL work, SQL schema, production ETL, feature engineering, modeling, inventory optimization, simulation, dashboard, Docker, or deployment work was performed.",
            "",
        ]
    )
    return "\n".join(lines)
