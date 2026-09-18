from pathlib import Path

from conftest import SPEED_DIR, needs_data
from vjepa_physics.data import load_records, subset


@needs_data
def test_speed_records():
    records = load_records(SPEED_DIR)
    assert len(records) == 1536
    assert records.clip_id.is_unique and records.clip_id.is_monotonic_increasing
    assert records.speed_mps.nunique() == 64
    assert (records.acceleration_mps2 == 0).all()
    for column in ("theta_degrees", "start_x", "start_y", "video"):
        assert column in records
    assert all(Path(p).exists() for p in records.video_path.head(20))


@needs_data
def test_subset_is_seeded_and_spans_the_label_range():
    records = load_records(SPEED_DIR)
    a, b = subset(records, 100, seed=0), subset(records, 100, seed=0)
    assert a.clip_id.tolist() == b.clip_id.tolist()
    # The dataset is sorted by speed; the first 100 rows cover only a few values.
    assert records.head(100).speed_mps.nunique() < 10
    assert a.speed_mps.nunique() > 40
