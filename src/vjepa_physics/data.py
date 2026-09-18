"""Read a dataset's manifest and metadata into one records table.

One row per clip, sorted by clip id. Every metadata field is kept, including the
nuisance variables (theta, start position), because they are how we check that a
probe is not reading something other than its target.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_records(dataset_dir: str | Path) -> pd.DataFrame:
    """Parse `<dataset_dir>/manifest.jsonl`, resolving paths relative to the
    dataset directory as DATA.md specifies."""
    dataset_dir = Path(dataset_dir)
    manifest = dataset_dir / "manifest.jsonl"
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        meta = json.loads((dataset_dir / entry["metadata"]).read_text(encoding="utf-8"))
        if meta.pop("id") != entry["id"]:
            raise ValueError(f"manifest id {entry['id']} does not match its metadata file")
        start_x, start_y = meta.pop("start_position_xy_m")
        rows.append({
            "clip_id": entry["id"],
            "video": entry["video"],
            "video_path": str(dataset_dir / entry["video"]),
            **meta,
            "start_x": start_x,
            "start_y": start_y,
        })

    records = pd.DataFrame(rows).sort_values("clip_id").reset_index(drop=True)
    if records["clip_id"].duplicated().any():
        raise ValueError(f"duplicate clip ids in {manifest}")
    return records


def subset(records: pd.DataFrame, limit: int | None, seed: int = 0) -> pd.DataFrame:
    """A seeded random subset for debug runs.

    Not the first N rows: the datasets are sorted by label, so the first 100 speed
    clips would cover only the slowest few speeds.
    """
    if limit is None or limit >= len(records):
        return records
    return records.sample(n=limit, random_state=seed).sort_values("clip_id").reset_index(drop=True)
