"""Extract and cache pooled V-JEPA 2 features for one dataset.

    python scripts/01_extract.py --dataset speed               # full run -> artifacts/features/speed
    python scripts/01_extract.py --dataset speed --limit 100   # debug  -> artifacts/features/speed_n100

Resumes automatically if interrupted. The same command works for the
`acceleration` and `direction` datasets -- extraction does not depend on the target.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without pip install

import torch

from vjepa_physics.config import dataset_dir, features_dir, load_config
from vjepa_physics.data import load_records, subset
from vjepa_physics.encoder import FrozenVJEPA2
from vjepa_physics.extract import extract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="speed", choices=["speed", "acceleration", "direction"])
    parser.add_argument("--limit", type=int, default=None, help="seeded random subset of N clips")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pre, ext = cfg["preprocessing"], cfg["extraction"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    records = subset(load_records(dataset_dir(cfg, args.dataset)), args.limit, seed=cfg["probing"]["seed"])
    encoder = FrozenVJEPA2(cfg["model"]["id"], device, ext["autocast_dtype"],
                           pre["do_resize"], pre["do_center_crop"])
    out = extract(
        records, encoder, features_dir(cfg, args.dataset, args.limit),
        batch_size=args.batch_size or ext["batch_size"], decoder=pre["decoder"],
        num_frames=pre["num_frames"], frame_size=pre["frame_size"],
        pixel_size=ext["pixel_baseline_size"], decode_workers=ext["decode_workers"],
        overwrite=args.overwrite, extra_meta={"dataset": args.dataset, "limit": args.limit},
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
