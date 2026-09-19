from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEED_DIR = REPO / "data" / "speed"
ACCELERATION_DIR = REPO / "data" / "acceleration"

needs_data = pytest.mark.skipif(not (SPEED_DIR / "manifest.jsonl").exists(),
                                reason="supplied data not present (it is gitignored)")
needs_acceleration_data = pytest.mark.skipif(not (ACCELERATION_DIR / "manifest.jsonl").exists(),
                                             reason="acceleration data not present (it is gitignored)")
