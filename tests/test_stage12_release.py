"""Focused tests for SmartStock V1 release metadata and readiness artifacts."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smartstock.utils.environment_check import run_environment_check  # noqa: E402
from smartstock.utils.health_check import run_health_check  # noqa: E402


class Stage12ReleaseTests(unittest.TestCase):
    def test_version_metadata_declares_completed_v1(self) -> None:
        metadata = json.loads(
            (PROJECT_ROOT / "config/project_version.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["version"], "1.0.0")
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["python_version_target"], "3.12")
        self.assertFalse(metadata["live_cloud_deployment_claim"])

    def test_environment_check_supports_csv_demo_without_database(self) -> None:
        with patch.dict(os.environ, {"DATABASE_URL": ""}, clear=False):
            result = run_environment_check(PROJECT_ROOT, imports={"json": "JSON"})
        self.assertTrue(result["csv_demo_ready"])
        database_check = next(
            check for check in result["checks"] if check["name"] == "DATABASE_URL"
        )
        self.assertEqual(database_check["status"], "WARN")

    def test_health_check_validates_dashboard_contract(self) -> None:
        result = run_health_check(PROJECT_ROOT, verify_hashes=False)
        self.assertTrue(result["dashboard_demo_ready"])
        self.assertTrue(result["full_development_ready"])
        contract = next(
            check
            for check in result["checks"]
            if check["name"] == "Stage 10 application data contract"
        )
        self.assertEqual(contract["status"], "PASS")

    def test_container_configuration_excludes_large_data(self) -> None:
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
        self.assertIn("FROM python:3.12-slim", dockerfile)
        self.assertIn("USER smartstock", dockerfile)
        self.assertIn("condition: service_healthy", compose)
        self.assertIn("condition: service_completed_successfully", compose)
        self.assertIn("/app/data/interim:ro", compose)
        for excluded in ("data/raw", "data/interim", "data/processed", "models"):
            self.assertIn(excluded, dockerignore)

    def test_required_documentation_and_readme_sections_exist(self) -> None:
        required_docs = [
            "architecture.md", "pipeline.md", "forecasting.md", "inventory.md",
            "application.md", "deployment.md", "portfolio_summary.md",
            "resume_bullets.md", "interview_guide.md", "release_checklist.md",
        ]
        for name in required_docs:
            self.assertTrue((PROJECT_ROOT / "docs" / name).is_file(), name)
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        for heading in (
            "## Architecture", "## Results", "## Quick Start", "## Docker",
            "## Limitations", "## Future Improvements",
        ):
            self.assertIn(heading, readme)
        self.assertIn("synthetic demo assumptions", readme)

    def test_packaging_targets_python_312_and_src_layout(self) -> None:
        metadata = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
        app_requirements = (PROJECT_ROOT / "requirements-app.txt").read_text(encoding="utf-8")
        self.assertIn('requires-python = ">=3.12,<3.13"', metadata)
        self.assertIn('where = ["src"]', metadata)
        self.assertIn("-e .", requirements)
        self.assertIn("streamlit", app_requirements)
        self.assertNotIn("jupyterlab", app_requirements)


if __name__ == "__main__":
    unittest.main()
