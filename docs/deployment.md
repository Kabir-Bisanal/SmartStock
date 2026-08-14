# Deployment guide

SmartStock V1 supports three honest execution modes. No live cloud URL is claimed.

## Shared prerequisites

- Python 3.12
- The generated demo artifacts listed by `python -m smartstock.utils.health_check`
- A local clone with large M5/generated files restored outside Git

Install the project and verify it:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m smartstock.utils.environment_check
python -m smartstock.utils.health_check
```

## Mode A — local CSV demo

```powershell
$env:SMARTSTOCK_DATA_MODE="csv"
python -m streamlit run app/streamlit_app.py
```

This is read-only and requires no database. `scripts/run_demo.ps1` performs the checks and starts the same mode.

## Mode B — local PostgreSQL

Copy `.env.example` to the ignored `.env`, set `DATABASE_URL`, and leave `SMARTSTOCK_DATA_MODE=postgres` or `auto`.

```powershell
python -m smartstock.database.initialize_database
python -m smartstock.database.verify_database
python -m streamlit run app/streamlit_app.py
```

The loader validates file schemas/hashes, creates tables, and skips an already complete database. Use `--refresh` only when a deliberate reload is required. See `database_setup.md` for details.

## Mode C — Docker Compose full stack

Restore the generated artifact directories on the host, then optionally put non-default local PostgreSQL values in an ignored `.env`:

```powershell
docker compose config
docker compose up --build
```

Compose starts PostgreSQL, waits for its health check, runs the idempotent initializer, and starts Streamlit on `http://localhost:8501`. Generated data/model directories are mounted read-only into application containers; PostgreSQL data uses the named `smartstock_postgres_data` volume. The large raw Kaggle and feature files are excluded from the image.

Stop services with `docker compose down`. Add `--volumes` only when intentionally deleting the local database volume.

## Environment variables

| Name | Purpose |
|---|---|
| `SMARTSTOCK_DATA_MODE` | `auto`, `postgres`, or `csv`. |
| `DATABASE_URL` | SQLAlchemy PostgreSQL/Psycopg URL; keep credentials outside Git. |
| `POSTGRES_DB` | Compose database name. |
| `POSTGRES_USER` | Compose development user. |
| `POSTGRES_PASSWORD` | Compose development password; replace the placeholder. |

## Generic hosted containers

For a container platform, provide a managed PostgreSQL URL, persistent application artifacts (or an object-store download step), port 8501 routing, HTTPS, secret management, health monitoring, and backups. Authentication, CI/CD, scheduled refreshes, and observability remain V2 work. Deployment readiness means the repository has a reproducible container path; it does not mean production operations have been completed.
