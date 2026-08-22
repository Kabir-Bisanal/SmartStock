"""Validate SmartStock V1 data, model, configuration, and report integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from smartstock.database.loaders import ArtifactPaths, csv_row_count, validate_artifact_contracts
from smartstock.data.public_demo_bundle import PublicDemoPaths, validate_public_demo_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEVELOPMENT_ASSETS = [
    "data/raw/calendar.csv",
    "data/raw/sales_train_evaluation.csv",
    "data/raw/sell_prices.csv",
    "data/processed/smartstock_v1_features.csv",
    "config/v1_subset.json",
    "config/v1_features.json",
    "config/v1_validation.json",
]
DEMO_ASSETS = [
    "data/interim/smartstock_v1_long.csv",
    "data/processed/production/smartstock_v1_30day_forecast.csv",
    "data/processed/inventory/smartstock_v1_error_calibration.csv",
    "data/processed/inventory/smartstock_v1_inventory_recommendations.csv",
    "data/simulated/smartstock_v1_inventory_snapshot.csv",
    "models/smartstock_v1_deployment_forecaster.joblib",
    "config/v1_final_model.json",
    "config/v1_inventory_policy.json",
]
PUBLIC_DEMO_ASSETS = [
    "data/public_demo/history_120d.csv",
    "data/public_demo/production_forecasts.csv",
    "data/public_demo/inventory_snapshot.csv",
    "data/public_demo/inventory_recommendations.csv",
    "data/public_demo/metadata.json",
    "config/v1_inventory_policy.json",
]
CORE_REPORTS = [
    "reports/stage3_data_profile.md",
    "reports/stage4_transformation_report.md",
    "reports/stage5_eda_report.md",
    "reports/stage6_feature_engineering_report.md",
    "reports/stage7_baseline_evaluation_report.md",
    "reports/stage8_ridge_evaluation_report.md",
    "reports/stage9_final_model_report.md",
    "reports/stage9_test_evaluation_receipt.json",
    "reports/stage10_inventory_optimization_report.md",
    "reports/stage11_application_report.md",
    "reports/stage12_final_qa_report.md",
    "reports/stage12_summary.json",
]


@dataclass(frozen=True)
class HealthResult:
    name: str
    status: str
    scope: str
    message: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_checks(root: Path, paths: list[str], scope: str) -> list[HealthResult]:
    return [
        HealthResult(
            f"Asset: {relative}",
            "PASS" if (root / relative).is_file() else "FAIL",
            scope,
            "available" if (root / relative).is_file() else "missing",
        )
        for relative in paths
    ]


def _expected_hashes(root: Path) -> dict[str, str]:
    final_model = json.loads((root / "config/v1_final_model.json").read_text(encoding="utf-8"))
    stage10 = json.loads((root / "reports/stage10_summary.json").read_text(encoding="utf-8"))
    sources = final_model["source_hashes_at_selection"]
    return {
        "data/raw/sales_train_evaluation.csv": sources["raw_sales"],
        "data/raw/sell_prices.csv": sources["raw_prices"],
        "data/raw/calendar.csv": sources["raw_calendar"],
        "data/interim/smartstock_v1_long.csv": sources["interim_dataset"],
        "data/processed/smartstock_v1_features.csv": sources["feature_dataset"],
        "config/v1_subset.json": sources["subset_manifest"],
        "config/v1_features.json": sources["feature_manifest"],
        "config/v1_validation.json": sources["validation_manifest"],
        "config/v1_ridge.json": sources["ridge_config"],
        "config/v1_final_model.json": stage10["stage9_integrity"]["hashes_after"]["final_model_config"],
        "models/smartstock_v1_deployment_forecaster.joblib": stage10["deployment_forecast"]["artifact_sha256"],
        "data/processed/production/smartstock_v1_30day_forecast.csv": stage10["deployment_forecast"]["output_sha256"],
        "data/processed/inventory/smartstock_v1_error_calibration.csv": stage10["uncertainty_calibration"]["output_sha256"],
        "data/simulated/smartstock_v1_inventory_snapshot.csv": stage10["synthetic_inventory"]["output_sha256"],
        "data/processed/inventory/smartstock_v1_inventory_recommendations.csv": stage10["recommendation_output"]["sha256"],
    }


def run_health_check(
    project_root: Path = PROJECT_ROOT,
    *,
    verify_hashes: bool = True,
) -> dict[str, object]:
    """Return development and dashboard-demo readiness with explicit check scopes."""

    root = project_root.resolve()
    checks = [
        *_asset_checks(root, PUBLIC_DEMO_ASSETS, "dashboard_demo"),
        *_asset_checks(root, DEMO_ASSETS, "full_development"),
        *_asset_checks(root, DEVELOPMENT_ASSETS, "full_development"),
        *_asset_checks(root, CORE_REPORTS, "full_development"),
    ]
    version_path = root / "config/project_version.json"
    if version_path.is_file():
        metadata = json.loads(version_path.read_text(encoding="utf-8"))
        version_ok = metadata.get("version") == "1.0.0" and metadata.get("status") == "complete"
        checks.append(HealthResult("Project version metadata", "PASS" if version_ok else "FAIL", "dashboard_demo", f"version={metadata.get('version')}, status={metadata.get('status')}"))
    else:
        checks.append(HealthResult("Project version metadata", "FAIL", "dashboard_demo", "missing"))

    public_paths = PublicDemoPaths.from_directory(root / "data/public_demo")
    if all((root / relative).is_file() for relative in PUBLIC_DEMO_ASSETS):
        try:
            contract = validate_public_demo_bundle(public_paths)
            checks.append(
                HealthResult(
                    "Public deployment bundle contract",
                    "PASS",
                    "dashboard_demo",
                    json.dumps(contract["row_counts"], sort_keys=True),
                )
            )
        except Exception as exc:
            checks.append(
                HealthResult(
                    "Public deployment bundle contract", "FAIL", "dashboard_demo", str(exc)
                )
            )

    paths = ArtifactPaths(
        history=root / "data/interim/smartstock_v1_long.csv",
        forecasts=root / "data/processed/production/smartstock_v1_30day_forecast.csv",
        inventory_snapshot=root / "data/simulated/smartstock_v1_inventory_snapshot.csv",
        recommendations=root / "data/processed/inventory/smartstock_v1_inventory_recommendations.csv",
        final_model_config=root / "config/v1_final_model.json",
        inventory_policy=root / "config/v1_inventory_policy.json",
        deployment_model=root / "models/smartstock_v1_deployment_forecaster.joblib",
    )
    if all((root / relative).is_file() for relative in DEMO_ASSETS):
        try:
            contract = validate_artifact_contracts(paths)
            checks.append(HealthResult("Stage 10 application data contract", "PASS", "full_development", json.dumps(contract["row_counts"], sort_keys=True)))
        except Exception as exc:
            checks.append(HealthResult("Stage 10 application data contract", "FAIL", "full_development", str(exc)))

    feature_path = root / "data/processed/smartstock_v1_features.csv"
    if feature_path.is_file():
        feature_rows = csv_row_count(feature_path)
        checks.append(HealthResult("Feature dataset rows", "PASS" if feature_rows == 480_506 else "FAIL", "full_development", f"{feature_rows:,}; expected 480,506"))

    if verify_hashes:
        try:
            hashes = _expected_hashes(root)
            for relative, expected in hashes.items():
                path = root / relative
                if not path.is_file():
                    continue
                actual = file_sha256(path)
                scope = "full_development"
                checks.append(HealthResult(f"Frozen hash: {relative}", "PASS" if actual == expected else "FAIL", scope, actual))
        except Exception as exc:
            checks.append(HealthResult("Frozen hash manifest", "FAIL", "full_development", str(exc)))

    demo_checks = [check for check in checks if check.scope == "dashboard_demo"]
    development_checks = checks
    counts = {status: sum(check.status == status for check in checks) for status in ("PASS", "WARN", "FAIL")}
    return {
        "project_root": str(root),
        "checks": [asdict(check) for check in checks],
        "counts": counts,
        "dashboard_demo_ready": not any(check.status == "FAIL" for check in demo_checks),
        "full_development_ready": not any(check.status == "FAIL" for check in development_checks),
        "hash_verification_enabled": verify_hashes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-hashes", action="store_true", help="Skip slower SHA-256 comparisons.")
    parser.add_argument(
        "--deployment-only",
        action="store_true",
        help="Return success when the committed public dashboard bundle is ready, even if local development data is absent.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_health_check(verify_hashes=not args.skip_hashes)
    print("SmartStock project health check")
    for check in result["checks"]:
        print(f"[{check['status']}] [{check['scope']}] {check['name']}: {check['message']}")
    print("Dashboard demo readiness:", "PASS" if result["dashboard_demo_ready"] else "FAIL")
    print("Full development readiness:", "PASS" if result["full_development_ready"] else "FAIL")
    if args.deployment_only:
        return 0 if result["dashboard_demo_ready"] else 1
    return 0 if result["dashboard_demo_ready"] and result["full_development_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
