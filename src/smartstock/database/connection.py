"""Safe database configuration and SQLAlchemy engine creation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL, make_url


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUPPORTED_DATA_MODES = {"auto", "postgres", "csv"}


def load_local_env(path: Path | None = None) -> None:
    """Load simple KEY=VALUE entries from .env without overriding real environment values."""

    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env entry on line {line_number}; expected KEY=VALUE.")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            raise ValueError(f"Invalid empty key in .env on line {line_number}.")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class DatabaseSettings:
    """Runtime database/fallback settings sourced from environment variables."""

    database_url: str | None
    data_mode: str = "auto"

    @classmethod
    def from_environment(cls, *, load_dotenv: bool = True) -> "DatabaseSettings":
        if load_dotenv:
            load_local_env()
        mode = os.getenv("SMARTSTOCK_DATA_MODE", "auto").strip().lower() or "auto"
        if mode not in SUPPORTED_DATA_MODES:
            choices = ", ".join(sorted(SUPPORTED_DATA_MODES))
            raise ValueError(f"SMARTSTOCK_DATA_MODE must be one of: {choices}.")
        url = os.getenv("DATABASE_URL", "").strip() or None
        return cls(database_url=url, data_mode=mode)


def normalize_database_url(database_url: str, *, allow_sqlite: bool = False) -> URL:
    """Validate a URL without exposing credentials and select Psycopg 3 for PostgreSQL."""

    value = database_url.strip()
    if value.startswith("postgres://"):
        value = "postgresql+psycopg://" + value[len("postgres://") :]
    elif value.startswith("postgresql://"):
        value = "postgresql+psycopg://" + value[len("postgresql://") :]
    try:
        url = make_url(value)
    except Exception as exc:
        raise ValueError("DATABASE_URL is not a valid SQLAlchemy URL.") from exc
    if url.drivername == "sqlite" and allow_sqlite:
        return url
    if url.drivername != "postgresql+psycopg":
        raise ValueError(
            "DATABASE_URL must use PostgreSQL with Psycopg 3, for example "
            "postgresql+psycopg://user:password@localhost:5432/smartstock."
        )
    if not url.database:
        raise ValueError("DATABASE_URL must name a PostgreSQL database.")
    return url


def create_database_engine(
    database_url: str,
    *,
    allow_sqlite: bool = False,
    connect_timeout_seconds: int = 5,
) -> Engine:
    """Create a pooled SQLAlchemy engine; SQLite is allowed only for automated tests."""

    url = normalize_database_url(database_url, allow_sqlite=allow_sqlite)
    kwargs: dict[str, object] = {"future": True, "pool_pre_ping": True}
    if url.drivername == "postgresql+psycopg":
        kwargs["connect_args"] = {"connect_timeout": int(connect_timeout_seconds)}
    return create_engine(url, **kwargs)


def verify_connection(engine: Engine) -> None:
    """Raise an actionable exception when the database cannot answer a trivial query."""

    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
