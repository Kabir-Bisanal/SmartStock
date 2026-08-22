"""Build SmartStock's compact, deterministic public deployment bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smartstock.data.public_demo_bundle import build_public_demo_bundle  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(build_public_demo_bundle(), indent=2, sort_keys=True))
