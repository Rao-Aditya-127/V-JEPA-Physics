"""Part 2: the activation-manifold figures.

    python scripts/figures/manifold.py --variable direction

Refits fold 0's manifold (seconds) and draws three figures:

    manifold_<variable>.png          the curve, its centroids and the clips around it
    manifold_chord_<variable>.png    the curve against the straight paths across it
    manifold_residual_<variable>.png how far held-out clips sit from curve vs line

Everything is drawn in a frame built from the CENTROIDS, not the clips. The clips'
own principal directions are dominated by start position and the other physical
variables -- in that frame the manifold is nearly edge-on and cannot be seen.

Needs no GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # works without pip install

import numpy as np
import pandas as pd

from vjepa_physics.config import features_dir, load_config
from vjepa_physics.features import load_features
from vjepa_physics.folds import grouped_folds
from vjepa_physics.manifold import (
    centroids, choose_smoothing, chord, components_for_variance, fit_manifold,
)
from vjepa_physics.plots import (
    manifold_3d_figure, manifold_figure, manifold_overview_figure, manifold_residual_figure,
    manifold_vs_chord_figure,
)
from vjepa_physics.probes import standardize

UNITS = {"speed": " m/s", "acceleration": " m/s²", "direction": "°"}


def straight_line(X, y, k, pca_mean, components):
    values, means, _ = centroids((X - pca_mean) @ components.T, y)
    fits = [np.polyfit(values, means[:, j], 1) for j in range(k)]
    return lambda v: np.stack([np.polyval(f, np.atleast_1d(v)) for f in fits], 1) @ components + pca_mean


def fit_fold0(cfg, variable):
    """Fold 0's manifold, plus everything needed to draw it in its centroid frame."""
    var_cfg, probing, mf = cfg["variables"][variable], cfg["probing"], cfg["manifold"]
    circular = var_cfg.get("circular", False)
    period = 360.0 if circular else None
    feats = load_features(features_dir(cfg, var_cfg["dataset"]))
    label = feats.labels[var_cfg["target"]].to_numpy(dtype=np.float64)
    X_all = feats.layer(cfg["nullspace"]["layer"]).astype(np.float64)
    fold = grouped_folds(label, probing["n_folds"], probing["inner_val_every"])[0]
    train = np.sort(np.concatenate([fold.fit, fold.val]))
    X_fit, X_val, X_test, X_train = standardize(
        X_all[fold.fit], X_all[fold.val], X_all[fold.test], X_all[train])
    y_train, y_test = label[train], label[fold.test]
    k = components_for_variance(X_train, y_train, mf["centroid_variance"])
    smoothing, _ = choose_smoothing(X_fit, label[fold.fit], X_val, label[fold.val], k=k,
                                    grid=[float(s) for s in mf["smoothing_grid"]],
                                    periodic=circular, period=period)
    manifold = fit_manifold(X_train, y_train, k=k, smoothing=smoothing,
                            periodic=circular, period=period)
    values, means, _ = centroids(X_train, y_train)
    frame_mean = means.mean(axis=0)
    frame = np.linalg.svd(means - frame_mean, full_matrices=False)[2][:3]
    to_frame = lambda Z: (np.atleast_2d(Z) - frame_mean) @ frame.T
    return dict(manifold=manifold, values=values, means=means, to_frame=to_frame,
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                k=k, smoothing=smoothing, circular=circular)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variable", default="direction",
                        choices=["speed", "acceleration", "direction", "all"],
                        help="'all' draws only the three-panel overview")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    figures_dir = cfg["paths"]["artifacts"] / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.variable == "all":
        panels = {}
        for v in ("speed", "acceleration", "direction"):
            got = fit_fold0(cfg, v)
            thin = np.arange(0, len(got["X_train"]), 3)
            curve = got["to_frame"](got["manifold"].evaluate(got["manifold"]._grid(600)))
            panels[v] = (got["to_frame"](got["X_train"])[thin], got["y_train"][thin],
                         got["to_frame"](got["means"]), got["values"], curve,
                         UNITS[v], got["circular"])
            print(f'  {v}: PCA k={got["k"]}, smoothing {got["smoothing"]:g}')
        print(f'wrote {manifold_overview_figure(panels, figures_dir / "manifold_all.png")}')
        return

    var_cfg, probing, mf = cfg["variables"][args.variable], cfg["probing"], cfg["manifold"]
    layer = cfg["nullspace"]["layer"]
    circular = var_cfg.get("circular", False)
    period = 360.0 if circular else None

    feat_dir = features_dir(cfg, var_cfg["dataset"])
    feats = load_features(feat_dir)
    label = feats.labels[var_cfg["target"]].to_numpy(dtype=np.float64)
    X_all = feats.layer(layer).astype(np.float64)

    fold = grouped_folds(label, probing["n_folds"], probing["inner_val_every"])[0]
    train = np.sort(np.concatenate([fold.fit, fold.val]))
    X_fit, X_val, X_test, X_train = standardize(
        X_all[fold.fit], X_all[fold.val], X_all[fold.test], X_all[train])
    y_train, y_test = label[train], label[fold.test]

    k = components_for_variance(X_train, y_train, mf["centroid_variance"])
    smoothing, _ = choose_smoothing(X_fit, label[fold.fit], X_val, label[fold.val], k=k,
                                    grid=[float(s) for s in mf["smoothing_grid"]],
                                    periodic=circular, period=period)
    manifold = fit_manifold(X_train, y_train, k=k, smoothing=smoothing,
                            periodic=circular, period=period)
    line = straight_line(X_train, y_train, k, manifold.pca_mean, manifold.components)
    print(f"{args.variable}: fold 0, PCA k={k}, smoothing {smoothing:g}")

    # The drawing frame: the plane the centroids themselves occupy.
    values, means, _ = centroids(X_train, y_train)
    frame_mean = means.mean(axis=0)
    frame = np.linalg.svd(means - frame_mean, full_matrices=False)[2][:3]
    to_frame = lambda Z: (np.atleast_2d(Z) - frame_mean) @ frame.T

    grid = manifold._grid(600)
    curve = to_frame(manifold.evaluate(grid))
    figures = cfg["paths"]["artifacts"] / "results" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    unit = UNITS[args.variable]

    thin = np.arange(0, len(X_train), 3)          # a sample, so the 3D view stays readable
    a3 = manifold_3d_figure(to_frame(X_train)[thin], y_train[thin], to_frame(means), values,
                            curve, figures / f"manifold3d_{args.variable}.png",
                            args.variable, unit, circular)
    a = manifold_figure(to_frame(X_train), y_train, to_frame(means), values, curve, values,
                        figures / f"manifold_{args.variable}.png", args.variable, unit, circular)

    # Straight paths between points on the curve: a short hop, a half-range jump, and
    # the extreme pair. For direction the extreme pair is opposite ends of the circle.
    lo, mid, hi = values[0], values[len(values) // 2], values[-1]
    quarter = values[len(values) // 4]
    pairs = [(lo, quarter), (lo, mid), (quarter, values[3 * len(values) // 4])]
    chords = [to_frame(chord(manifold, v0, v1, np.linspace(0, 1, 41))) for v0, v1 in pairs]
    names = [f"{v0:g}{unit.strip()} to {v1:g}{unit.strip()}" for v0, v1 in pairs]
    b = manifold_vs_chord_figure(curve, chords, names,
                                 figures / f"manifold_chord_{args.variable}.png", args.variable)

    d_curve = np.linalg.norm(X_test - manifold.evaluate(y_test), axis=1)
    d_line = np.linalg.norm(X_test - line(y_test), axis=1)
    within = np.mean([np.linalg.norm(X_train[y_train == v] - X_train[y_train == v].mean(0),
                                     axis=1).mean() for v in values])
    c = manifold_residual_figure(y_test, d_curve, d_line,
                                 figures / f"manifold_residual_{args.variable}.png",
                                 args.variable, unit, scatter=within)

    print(f"  curve {d_curve.mean():.2f}   line {d_line.mean():.2f}   "
          f"within-value spread {within:.2f}")
    for path in (a3, a, b, c):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
