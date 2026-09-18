"""Run the frozen encoder over a dataset once and cache the pooled features.

Output directory layout:

    labels.csv    one row per clip, aligned with the arrays by row index
    pooled.npy    (N, num_layers + 1, hidden)  float16   index 0 = patch embeddings
    pixels.npy    (N, frames * size * size)    float16   [OURS] raw-pixel baseline
    done.npy      (N,) bool                               which rows are written
    meta.json     how these features were produced

The arrays are memory-mapped and written batch by batch, with `done.npy` flushed
after each batch, so a crashed run resumes where it stopped instead of restarting.
"""

from __future__ import annotations

import json
import platform
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import tqdm

from .config import REPO_ROOT
from .encoder import FrozenVJEPA2
from .video import load_clip, pixel_features, resolve_decoder

ALIGNMENT_COLUMNS = ["clip_id", "video"]


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except Exception:
        return {"commit": None, "dirty": None}


def _open_arrays(out_dir: Path, records: pd.DataFrame, n_states: int, hidden: int,
                 pixel_dim: int, overwrite: bool):
    """Open existing arrays for resuming, or create fresh ones."""
    paths = {name: out_dir / f"{name}.npy" for name in ("pooled", "pixels", "done")}
    shapes = {"pooled": (len(records), n_states, hidden), "pixels": (len(records), pixel_dim)}

    resumable = not overwrite and all(p.exists() for p in paths.values()) \
        and (out_dir / "labels.csv").exists()
    if resumable:
        existing = pd.read_csv(out_dir / "labels.csv")
        same_clips = existing[ALIGNMENT_COLUMNS].equals(records[ALIGNMENT_COLUMNS].reset_index(drop=True))
        pooled = np.load(paths["pooled"], mmap_mode="r+")
        pixels = np.load(paths["pixels"], mmap_mode="r+")
        if not same_clips or pooled.shape != shapes["pooled"] or pixels.shape != shapes["pixels"]:
            raise RuntimeError(
                f"{out_dir} holds features for a different clip set or shape. "
                "Pass overwrite=True (--overwrite) to replace them."
            )
        return pooled, pixels, np.load(paths["done"])

    out_dir.mkdir(parents=True, exist_ok=True)
    records.drop(columns=["video_path"]).to_csv(out_dir / "labels.csv", index=False)
    pooled = np.lib.format.open_memmap(paths["pooled"], mode="w+", dtype=np.float16,
                                       shape=shapes["pooled"])
    pixels = np.lib.format.open_memmap(paths["pixels"], mode="w+", dtype=np.float16,
                                       shape=shapes["pixels"])
    done = np.zeros(len(records), dtype=bool)
    np.save(paths["done"], done)
    return pooled, pixels, done


def extract(records: pd.DataFrame, encoder: FrozenVJEPA2, out_dir: str | Path, *,
            batch_size: int = 8, decoder: str = "auto", num_frames: int = 16,
            frame_size: int = 256, pixel_size: int = 16, decode_workers: int = 4,
            overwrite: bool = False, extra_meta: dict | None = None) -> Path:
    out_dir = Path(out_dir)
    records = records.reset_index(drop=True)
    backend = resolve_decoder(decoder)
    n_states = encoder.num_layers + 1
    pixel_dim = num_frames * pixel_size * pixel_size

    pooled, pixels, done = _open_arrays(out_dir, records, n_states, encoder.hidden_size,
                                        pixel_dim, overwrite)
    todo = np.flatnonzero(~done)
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    print(f"{out_dir.name}: {done.sum()}/{len(records)} already done, "
          f"{len(todo)} to extract with decoder={backend}")
    if len(todo) == 0 and (out_dir / "meta.json").exists():
        return out_dir   # nothing new: keep the metadata describing the run that made these

    def decode(row: int):
        frames = load_clip(records.at[row, "video_path"], backend, num_frames, frame_size)
        return frames, pixel_features(frames, pixel_size)

    started = datetime.now(timezone.utc)
    with ThreadPoolExecutor(max_workers=decode_workers) as pool:
        # Decode batch k+1 on worker threads while the encoder runs batch k.
        pending = [pool.submit(decode, row) for row in batches[0]] if batches else []
        for k, rows in enumerate(tqdm(batches, desc="extract", unit="batch")):
            decoded = [future.result() for future in pending]
            if k + 1 < len(batches):
                pending = [pool.submit(decode, row) for row in batches[k + 1]]

            clips = [frames for frames, _ in decoded]
            feats = encoder.pooled(clips).numpy()
            if not np.isfinite(feats).all():
                raise FloatingPointError(f"non-finite features in batch rows {rows.tolist()}")

            pooled[rows] = feats.astype(np.float16)
            pixels[rows] = torch.stack([pix for _, pix in decoded]).numpy().astype(np.float16)
            pooled.flush()
            pixels.flush()
            done[rows] = True
            np.save(out_dir / "done.npy", done)

    meta = {
        "model_id": encoder.model_id,
        "n_clips": int(len(records)),
        "complete": bool(done.all()),
        "pooled_shape": list(pooled.shape),
        "pooled_layout": "index 0 = patch embeddings; index l+1 = output of block l "
                         "(paper layer l); pre-final-LayerNorm",
        "pooling": "mean over all space-time tokens, computed in float32, stored float16",
        "preprocessing": {"do_resize": encoder.do_resize, "do_center_crop": encoder.do_center_crop,
                          "image_mean": list(encoder.processor.image_mean),
                          "image_std": list(encoder.processor.image_std),
                          "num_frames": num_frames, "frame_size": frame_size},
        "decoder": backend,
        "pixel_baseline": {"size": pixel_size, "dims": pixel_dim, "form": "grayscale, area-downsampled"},
        "device": str(encoder.device),
        "gpu": torch.cuda.get_device_name(0) if encoder.device.type == "cuda" else None,
        "autocast_dtype": str(encoder.autocast_dtype) if encoder.device.type == "cuda" else None,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "python": platform.python_version()},
        "git": git_state(),
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra_meta or {}),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir
