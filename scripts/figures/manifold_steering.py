"""Part 2, stage 2: the steering-comparison figures.

    python scripts/figures/manifold_steering.py --variable direction

Reads the run from 06_manifold_steering.py and draws four figures:

    steer_paths_<v>.png     the two routes drawn on the manifold (cf. paper Fig. 7c)
    steer_tracking_<v>.png  does the readout follow, along the journey?
    steer_journey_<v>.png   the same, against how far the clip had to travel
    steer_subspace_<v>.png  Part 1.3's method alongside, for the brief's comparison

The first three show linear against manifold steering -- the comparison the Goodfire
paper makes. The fourth adds Part 1.3's subspace method, which the paper never
discusses and which the brief asks us to compare against, so it gets its own figure
rather than crowding the reproduction.

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
    centroids, choose_smoothing, components_for_variance, fit_manifold, short_delta, steering_step,
)
from vjepa_physics.plots import (
    steering_journey_figure, steering_paths_figure, steering_tracking_figure,
)
from vjepa_physics.probes import standardize

UNITS = {"speed": " m/s", "acceleration": " m/s²", "direction": "°"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variable", default="direction", choices=["speed", "acceleration", "direction"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    var_cfg, probing, mf = cfg["variables"][args.variable], cfg["probing"], cfg["manifold"]
    circular = var_cfg.get("circular", False)
    period = 360.0 if circular else None
    target = cfg["steering"]["target"][args.variable]
    unit = UNITS[args.variable]

    base = cfg["paths"]["artifacts"] / "results" / var_cfg["dataset"] / "manifold_steering"
    if not (base / "summary.csv").exists():
        sys.exit(f"missing {base/'summary.csv'}: run 06_manifold_steering.py --variable {args.variable}")
    summary = pd.read_csv(base / "summary.csv")
    binned = pd.read_csv(base / "summary_by_journey.csv")
    figures = cfg["paths"]["artifacts"] / "results" / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # Refit fold 0 so the illustrative path figure can be drawn on the real manifold.
    feats = load_features(features_dir(cfg, var_cfg["dataset"]))
    label = feats.labels[var_cfg["target"]].to_numpy(dtype=np.float64)
    X_all = feats.layer(cfg["nullspace"]["layer"]).astype(np.float64)
    fold = grouped_folds(label, probing["n_folds"], probing["inner_val_every"])[0]
    train = np.sort(np.concatenate([fold.fit, fold.val]))
    X_fit, X_val, X_train = standardize(X_all[fold.fit], X_all[fold.val], X_all[train])
    y_train = label[train]
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
    curve = to_frame(manifold.evaluate(manifold._grid(600)))

    # The longest journey available: the state furthest from the target.
    start = float(values[np.argmax(np.abs(short_delta(values, target, period)))])
    origin = manifold.evaluate([start])
    paths = {}
    for method in ("linear", "manifold"):
        points = [steering_step(manifold, origin, np.array([start]), target, t, method)[0][0]
                  for t in np.linspace(0, 1, 21)]
        paths[method] = to_frame(np.stack(points))
    a = steering_paths_figure(curve, paths, figures / f"steer_paths_{args.variable}.png",
                              args.variable, unit,
                              title=f"{args.variable}: {start:g}{unit.strip()} to "
                                    f"{target:g}{unit.strip()}, two routes")

    b = steering_tracking_figure(summary, figures / f"steer_tracking_{args.variable}.png",
                                 args.variable, unit, confidence=circular)
    c = steering_journey_figure(binned, figures / f"steer_journey_{args.variable}.png",
                                args.variable, unit)
    d = steering_tracking_figure(summary, figures / f"steer_subspace_{args.variable}.png",
                                 args.variable, unit, methods=("linear", "manifold", "subspace"),
                                 confidence=circular,
                                 title=f"{args.variable}: with Part 1.3's method alongside")

    for path in (a, b, c, d):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
