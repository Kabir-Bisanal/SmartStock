"""Initialize or refresh the SmartStock PostgreSQL application database."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from smartstock.database.connection import DatabaseSettings, create_database_engine, verify_connection
from smartstock.database.loaders import load_database_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Delete SmartStock table rows and reload all generated artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = DatabaseSettings.from_environment()
    if not settings.database_url:
        raise SystemExit(
            "DATABASE_URL is not configured. Copy .env.example to .env, set a PostgreSQL URL, "
            "and try again. The Streamlit app can still run in CSV demo mode."
        )
    engine = create_database_engine(settings.database_url)
    try:
        verify_connection(engine)
        result = load_database_artifacts(engine, refresh=args.refresh)
    finally:
        engine.dispose()
    print("SmartStock database initialization complete")
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
