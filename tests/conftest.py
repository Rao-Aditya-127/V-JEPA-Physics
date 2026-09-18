from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEED_DIR = REPO / "data" / "speed"

needs_data = pytest.mark.skipif(not (SPEED_DIR / "manifest.jsonl").exists(),
                                reason="supplied data not present (it is gitignored)")
