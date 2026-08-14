"""Check whether the local environment can run SmartStock V1."""

from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TARGET_PYTHON = (3, 12)
REQUIRED_IMPORTS = {
    "numpy": "NumPy",
    "pandas": "Pandas",
    "matplotlib": "Matplotlib",
    "jupyterlab": "JupyterLab",
    "sklearn": "scikit-learn",
    "xgboost": "XGBoost",
    "streamlit": "Streamlit",
    "sqlalchemy": "SQLAlchemy",
    "psycopg": "Psycopg 3",
    "joblib": "Joblib",
}
EXPECTED_DIRECTORIES = ["app", "config", "data", "docs", "reports", "sql", "src", "tests"]
DEMO_ARTIFACTS = [
    "data/interim/smartstock_v1_long.csv",
    "data/processed/production/smartstock_v1_30day_forecast.csv",
    "data/processed/inventory/smartstock_v1_error_calibration.csv",
    "data/processed/inventory/smartstock_v1_inventory_recommendations.csv",
    "data/simulated/smartstock_v1_inventory_snapshot.csv",
    "models/smartstock_v1_deployment_forecaster.joblib",
    "config/v1_final_model.json",
    "config/v1_inventory_policy.json",
]


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    message: str


def _python_check() -> Check:
    current = sys.version_info[:2]
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if current == TARGET_PYTHON:
        return Check("Python", "PASS", f"Python {version}; target is 3.12.")
    if current in {(3, 11), (3, 13)}:
        return Check("Python", "WARN", f"Python {version}; SmartStock V1 targets 3.12.")
    return Check("Python", "FAIL", f"Python {version} is unsupported; use Python 3.12.")


def _import_checks(imports: dict[str, str]) -> Iterable[Check]:
    for module_name, label in imports.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "imported")
            yield Check(f"Import: {label}", "PASS", str(version))
        except Exception as exc:
            yield Check(f"Import: {label}", "FAIL", f"{exc.__class__.__name__}: {exc}")


def run_environment_check(
    project_root: Path = PROJECT_ROOT,
    *,
    imports: dict[str, str] | None = None,
) -> dict[str, object]:
    """Return structured PASS/WARN/FAIL results without requiring PostgreSQL."""

    checks: list[Check] = [_python_check()]
    checks.extend(_import_checks(imports or REQUIRED_IMPORTS))
    for relative in EXPECTED_DIRECTORIES:
        path = project_root / relative
        checks.append(Check(f"Directory: {relative}", "PASS" if path.is_dir() else "FAIL", str(path)))
    for relative in DEMO_ARTIFACTS:
        path = project_root / relative
        checks.append(Check(f"Demo artifact: {relative}", "PASS" if path.is_file() else "FAIL", "available" if path.is_file() else "missing"))
    database_url = os.getenv("DATABASE_URL", "").strip()
    checks.append(
        Check(
            "DATABASE_URL",
            "PASS" if database_url else "WARN",
            "configured (value hidden)" if database_url else "not configured; CSV demo mode remains available",
        )
    )
    counts = {status: sum(check.status == status for check in checks) for status in ("PASS", "WARN", "FAIL")}
    return {
        "target_python": "3.12",
        "checks": [asdict(check) for check in checks],
        "counts": counts,
        "csv_demo_ready": counts["FAIL"] == 0,
    }


def main() -> int:
    result = run_environment_check()
    print("SmartStock environment check")
    for check in result["checks"]:
        print(f"[{check['status']}] {check['name']}: {check['message']}")
    print("Summary:", json.dumps(result["counts"], sort_keys=True))
    print("CSV demo readiness:", "PASS" if result["csv_demo_ready"] else "FAIL")
    return 1 if result["counts"]["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
