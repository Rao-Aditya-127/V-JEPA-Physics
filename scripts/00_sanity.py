"""Smoke test the extraction path before paying for a full run.

    python scripts/00_sanity.py                 # auto device, 40-clip viability check
    python scripts/00_sanity.py --n-viability 24

Checks, in order -- each one guards a bug that would otherwise surface hours
later as a flat or nonsensical curve:

  1. decode      shape, dtype, and the blue disk really is blue (RGB order)
  2. preprocess  output == manual (x/255 - mean)/std, so no resize/crop happened
  3. forward     25 hidden states of (B, 2048, 1024); hidden_states[l+1] is block l's
                 output (verified with hooks); hidden_states[24] != last_hidden_state
  4. viability   on a small subset, is speed linearly readable at all?

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without pip install

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from vjepa_physics.config import dataset_dir, load_config
from vjepa_physics.data import load_records, subset
from vjepa_physics.encoder import FrozenVJEPA2
from vjepa_physics.features import hidden_index
from vjepa_physics.video import load_clip, resolve_decoder

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def disk_mask(frames: torch.Tensor) -> torch.Tensor:
    """Pixels that differ clearly from the (uniform) background colour."""
    background = frames.flatten(2).median(dim=2).values[..., None, None]     # (T, C, 1, 1)
    return (frames.int() - background.int()).abs().sum(dim=1) > 60            # (T, H, W)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-viability", type=int, default=40)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pre, ext = cfg["preprocessing"], cfg["extraction"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    decoder = resolve_decoder(pre["decoder"])
    records = load_records(dataset_dir(cfg, cfg["variables"]["speed"]["dataset"]))
    print(f"device={device}  decoder={decoder}  clips={len(records)}")

    # ---- 1. decode --------------------------------------------------------------
    print("\n1. decode")
    frames = load_clip(records.video_path[0], decoder, pre["num_frames"], pre["frame_size"])
    check("shape (16, 3, 256, 256)", tuple(frames.shape) == (16, 3, 256, 256), str(tuple(frames.shape)))
    check("dtype uint8", frames.dtype == torch.uint8, str(frames.dtype))
    mask = disk_mask(frames)
    rgb_on_disk = frames.permute(1, 0, 2, 3)[:, mask].float().mean(dim=1)     # mean R, G, B on the disk
    r, g, b = rgb_on_disk.tolist()
    # DATA.md says "blue disk", but the files hold RGB (210, 105, 41) -- verified
    # against an independent PyAV decode. A channel swap here would read ~(41, 105, 210).
    check("RGB order: disk is the file's true colour ~(210, 105, 41)",
          abs(r - 210) < 15 and abs(g - 105) < 15 and abs(b - 41) < 15,
          f"disk pixels={int(mask.sum())}, mean R={r:.0f} G={g:.0f} B={b:.0f}")

    # ---- load model ---------------------------------------------------------------
    print(f"\nloading {cfg['model']['id']} ...")
    enc = FrozenVJEPA2(cfg["model"]["id"], device, ext["autocast_dtype"],
                       pre["do_resize"], pre["do_center_crop"])

    # ---- 2. preprocess ------------------------------------------------------------
    print("\n2. preprocess")
    pixels = enc.preprocess([frames])
    mean = torch.tensor(enc.processor.image_mean).view(1, 1, 3, 1, 1)
    std = torch.tensor(enc.processor.image_std).view(1, 1, 3, 1, 1)
    manual = (frames[None].float() / 255.0 - mean) / std
    check("shape (1, 16, 3, 256, 256)", tuple(pixels.shape) == (1, 16, 3, 256, 256), str(tuple(pixels.shape)))
    check("equals manual normalisation (no resize, no crop)",
          torch.allclose(pixels, manual, atol=1e-5), f"max diff {(pixels - manual).abs().max():.2e}")
    check("normalisation constants are ImageNet's",
          np.allclose(enc.processor.image_mean, [0.485, 0.456, 0.406])
          and np.allclose(enc.processor.image_std, [0.229, 0.224, 0.225]),
          f"mean={list(enc.processor.image_mean)} std={list(enc.processor.image_std)}")

    # ---- 3. forward ---------------------------------------------------------------
    print("\n3. forward")
    clips = [frames, load_clip(records.video_path[1], decoder, pre["num_frames"], pre["frame_size"])]
    captured: dict[int, torch.Tensor] = {}
    hooks = [enc.model.encoder.embeddings.register_forward_hook(
        lambda m, i, o: captured.__setitem__(0, o))]
    for l, block in enumerate(enc.model.encoder.layer):
        hooks.append(block.register_forward_hook(
            lambda m, i, o, l=l: captured.__setitem__(l + 1, o[0] if isinstance(o, tuple) else o)))
    states = enc.hidden_states(clips)
    for hook in hooks:
        hook.remove()

    check(f"{enc.num_layers + 1} hidden states", len(states) == enc.num_layers + 1, str(len(states)))
    check("each (2, 2048, 1024)", all(tuple(s.shape) == (2, 2048, 1024) for s in states),
          str(tuple(states[0].shape)))
    check("all finite", all(torch.isfinite(s).all() for s in states))
    check("hidden_states[0] is the patch embedding output", torch.equal(states[0], captured[0]))
    check("hidden_states[l+1] is block l's output, for every l",
          all(torch.equal(states[hidden_index(l)], captured[hidden_index(l)]) for l in range(enc.num_layers)))
    with torch.inference_mode(), enc._autocast():
        last = enc.model(pixel_values_videos=enc.preprocess(clips).to(enc.device),
                         skip_predictor=True).last_hidden_state
    check("hidden_states[24] != last_hidden_state (final LayerNorm)",
          not torch.allclose(states[-1].float(), last.float(), atol=1e-3),
          "so we use hidden_states throughout, never last_hidden_state")

    # ---- 4. viability -------------------------------------------------------------
    n = args.n_viability
    print(f"\n4. viability on {n} clips (seeded random subset)")
    sample = subset(records, n, seed=cfg["probing"]["seed"])
    feats, t0 = [], time.time()
    batch = ext["batch_size"]
    for i in range(0, n, batch):
        rows = sample.video_path[i:i + batch]
        feats.append(enc.pooled([load_clip(p, decoder, pre["num_frames"], pre["frame_size"]) for p in rows]))
    per_clip = (time.time() - t0) / n
    pooled = torch.cat(feats).numpy()
    speed = sample.speed_mps.to_numpy()

    print(f"   {'paper layer':>11}  {'LOO ridge R2':>12}  {'NN-speed corr':>13}")
    best = -np.inf
    for layer in (0, 4, 8, 16, 23):
        X = StandardScaler().fit_transform(pooled[:, hidden_index(layer), :])
        ridge = RidgeCV(alphas=np.logspace(-2, 5, 15), store_cv_results=True).fit(X, speed)
        loo_mse = ridge.cv_results_[:, list(ridge.alphas).index(ridge.alpha_)].mean()
        r2 = 1 - loo_mse / speed.var()
        dists = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(dists, np.inf)
        nn_corr = np.corrcoef(speed, speed[dists.argmin(1)])[0, 1]
        best = max(best, r2)
        print(f"   {layer:>11}  {r2:>12.3f}  {nn_corr:>13.3f}")
    check("speed is linearly readable at some layer (LOO R2 > 0.3)", best > 0.3,
          f"best {best:.3f}. Quick check only: alpha chosen on the same LOO errors.")

    est_min = per_clip * len(records) / 60
    print(f"\n   {per_clip:.2f} s/clip on {device} (incl. decode)  ->  "
          f"full speed extraction ~{est_min:.0f} min")

    print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
