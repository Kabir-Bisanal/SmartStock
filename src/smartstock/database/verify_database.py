"""Run read-only connectivity and integrity checks against the SmartStock database."""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

from sqlalchemy import Engine, func, select

from smartstock.database.connection import (
    DatabaseSettings,
    create_database_engine,
    verify_connection,
)
from smartstock.database.loaders import EXPECTED_ROW_COUNTS, validate_loaded_database
from smartstock.database.schema import TABLES


def verify_database(
    engine: Engine,
    *,
    expected_counts: dict[str, int] | None = None,
    strict_v1_cardinality: bool = True,
) -> dict[str, Any]:
    """Verify connectivity, table counts, forecast coverage, and synthetic labeling."""

    verify_connection(engine)
    expected = expected_counts or EXPECTED_ROW_COUNTS
    loaded = validate_loaded_database(engine, expected_counts=expected)
    forecasts = TABLES["forecasts"]
    inventory = TABLES["inventory_recommendations"]
    with engine.connect() as connection:
        forecast_days = int(
            connection.execute(select(func.count(func.distinct(forecasts.c.target_date)))).scalar_one()
        )
        max_horizon = int(
            connection.execute(select(func.max(forecasts.c.horizon_day))).scalar_one() or 0
        )
        synthetic_rows = int(
            connection.execute(
                select(func.count()).select_from(inventory).where(
                    inventory.c.inventory_source == "synthetic_demo"
                )
            ).scalar_one()
        )
    if strict_v1_cardinality:
        if forecast_days != 30 or max_horizon != 30:
            raise ValueError(
                f"Expected a 30-day V1 forecast; found {forecast_days} dates and horizon {max_horizon}."
            )
        if synthetic_rows != expected["inventory_recommendations"]:
            raise ValueError("Inventory recommendations are not all labeled synthetic_demo.")
    return {
        **loaded,
        "connection": "ok",
        "forecast_days": forecast_days,
        "max_horizon_day": max_horizon,
        "synthetic_inventory_rows": synthetic_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        help="Optional PostgreSQL URL. Defaults to DATABASE_URL; credentials are never printed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = DatabaseSettings.from_environment()
    database_url = (args.database_url or settings.database_url or "").strip()
    if not database_url:
        print(
            "Database verification skipped: DATABASE_URL is not configured. "
            "CSV demo mode remains available."
        )
        return 2
    engine = create_database_engine(database_url)
    try:
        result = verify_database(engine)
    except Exception as exc:
        print(f"Database verification failed: {exc}")
        return 1
    finally:
        engine.dispose()
    print("SmartStock PostgreSQL verification passed")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
