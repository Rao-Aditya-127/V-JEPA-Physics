"""Part 2: fit and evaluate the activation manifold for one variable.

    python scripts/05_manifold.py --variable direction

Fits the curve the activations actually occupy, parameterised by the physical
variable, following the Goodfire paper's App. A.3 / B.1: PCA, group by value, average,
spline. Direction gets a periodic spline so its circle closes.

Four evaluations, all on clips whose label value the manifold never saw:

  E1  reconstruction   how far is a held-out clip from the curve at its own value,
                       against two baselines -- the global mean, and the nearest
                       centroid the fit did see
  E2  unseen values    does the curve land on the true centroid of a value it was
                       never given?
  E3  curve vs line    does the curve beat the best straight line? If not, the
                       curvature is not real and manifold steering cannot help
  E4  coordinate       read the variable off a clip by projecting it onto the curve,
                       with no probe at all, and compare against the Part 1.1 probe

Smoothing is chosen per fold on the inner validation split, never by eye. Needs no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without pip install

import numpy as np
import pandas as pd

from vjepa_physics.config import features_dir, load_config
from vjepa_physics.extract import git_state
from vjepa_physics.features import load_features
from vjepa_physics.folds import check_folds, grouped_folds
from vjepa_physics.manifold import (
    centroids, choose_smoothing, chord, components_for_variance, fit_manifold, save_manifold,
)
from vjepa_physics.probes import standardize


def circular_gap(a, b, period):
    return np.abs((np.asarray(a) - np.asarray(b) + period / 2) % period - period / 2)


def straight_line(X, y, k, pca_mean, components):
    """The best straight line through the centroids -- E3's baseline.

    Degree-1 least squares per PCA coordinate. If this reconstructs held-out clips as
    well as the spline does, the manifold has no curvature worth steering along.
    """
    values, means, _ = centroids((X - pca_mean) @ components.T, y)
    fits = [np.polyfit(values, means[:, j], 1) for j in range(k)]
    return lambda v: np.stack([np.polyval(f, np.atleast_1d(v)) for f in fits], 1) @ components + pca_mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variable", default="direction", choices=["speed", "acceleration", "direction"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    var_cfg, probing, mf = cfg["variables"][args.variable], cfg["probing"], cfg["manifold"]
    layer = cfg["nullspace"]["layer"]
    circular = var_cfg.get("circular", False)
    period = 360.0 if circular else None

    feat_dir = features_dir(cfg, var_cfg["dataset"], args.limit)
    feats = load_features(feat_dir)
    label = feats.labels[var_cfg["target"]].to_numpy(dtype=np.float64)
    X_all = feats.layer(layer).astype(np.float64)

    out_dir = cfg["paths"]["artifacts"] / "results" / feat_dir.name / "manifold"
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = grouped_folds(label, probing["n_folds"], probing["inner_val_every"])
    check_folds(folds, len(label), label)
    grid = [float(s) for s in mf["smoothing_grid"]]

    print(f"{feat_dir.name}: layer {layer}, {len(folds)} grouped folds, "
          f"{'periodic' if circular else 'natural'} cubic spline, "
          f"PCA to {mf['centroid_variance']:.0%} of centroid variance, "
          f"smoothing chosen from {len(grid)} candidates")

    started, rows, per_fold = time.time(), [], []
    for f, fold in enumerate(folds):
        train = np.sort(np.concatenate([fold.fit, fold.val]))
        # Standardised exactly as in Parts 1.1-1.3, so the same probes could judge it.
        X_fit, X_val, X_test, X_train = standardize(
            X_all[fold.fit], X_all[fold.val], X_all[fold.test], X_all[train])
        y_fit, y_val, y_test, y_train = (label[fold.fit], label[fold.val],
                                         label[fold.test], label[train])

        k = components_for_variance(X_train, y_train, mf["centroid_variance"])
        smoothing, trace = choose_smoothing(X_fit, y_fit, X_val, y_val, k=k, grid=grid,
                                            periodic=circular, period=period)
        manifold = fit_manifold(X_train, y_train, k=k, smoothing=smoothing,
                                periodic=circular, period=period)
        line = straight_line(X_train, y_train, k, manifold.pca_mean, manifold.components)

        # Which held-out values are inside the fitted range? The rest are extrapolation,
        # which is a different question and is reported separately.
        held = np.unique(y_test)
        inside = (np.ones_like(held, dtype=bool) if circular else
                  (held >= y_train.min()) & (held <= y_train.max()))
        is_interp = np.isin(y_test, held[inside])

        # E1 reconstruction, and E3 against the straight line
        d_curve = np.linalg.norm(X_test - manifold.evaluate(y_test), axis=1)
        d_line = np.linalg.norm(X_test - line(y_test), axis=1)
        d_mean = np.linalg.norm(X_test - X_train.mean(0), axis=1)
        train_values, train_means, _ = centroids(X_train, y_train)
        gap = (circular_gap(y_test[:, None], train_values[None], 360.0) if circular
               else np.abs(y_test[:, None] - train_values[None]))
        d_nearest = np.linalg.norm(X_test - train_means[gap.argmin(1)], axis=1)

        # E2 unseen values: does the curve land on the true centroid?
        test_values, test_means, _ = centroids(X_test, y_test)
        e2_curve = np.linalg.norm(test_means - manifold.evaluate(test_values), axis=1)
        e2_nearest = np.linalg.norm(
            test_means - train_means[(circular_gap(test_values[:, None], train_values[None], 360.0)
                                      if circular else
                                      np.abs(test_values[:, None] - train_values[None])).argmin(1)], axis=1)

        # E4 coordinate recovery, probe-free
        coord = manifold.coordinate(X_test)
        e4 = (circular_gap(coord, y_test, 360.0) if circular else np.abs(coord - y_test))

        # E3b how far a straight path between two values strays from the curve
        ends = [(train_values[0], train_values[len(train_values) // 2]),
                (train_values[0], train_values[-1]),
                (train_values[len(train_values) // 4], train_values[3 * len(train_values) // 4])]
        strays = []
        curve_points = manifold.evaluate(manifold._grid(1000))
        for v0, v1 in ends:
            path = chord(manifold, v0, v1, np.linspace(0, 1, 21))
            off = np.linalg.norm(path[:, None] - curve_points[None], axis=-1).min(1)
            strays.append(off.max())
        radius = np.linalg.norm(curve_points - curve_points.mean(0), axis=1).mean()

        row = {"fold": f, "k": k, "smoothing": smoothing, "n_train": len(train),
               "n_test": len(fold.test), "arc_length": manifold.arc_length(), "radius": radius,
               "recon_curve": d_curve[is_interp].mean(), "recon_line": d_line[is_interp].mean(),
               "recon_mean": d_mean[is_interp].mean(), "recon_nearest": d_nearest[is_interp].mean(),
               "recon_curve_extrap": d_curve[~is_interp].mean() if (~is_interp).any() else np.nan,
               "n_extrapolated_values": int((~inside).sum()),
               "centroid_curve": e2_curve[inside].mean(), "centroid_nearest": e2_nearest[inside].mean(),
               "coordinate_error": e4.mean(), "chord_stray_max": max(strays),
               "chord_stray_relative": max(strays) / radius}
        rows.append(row)
        per_fold.append(pd.DataFrame(trace).assign(fold=f))
        if f == 0:
            save_manifold(manifold, out_dir / "manifold_fold0.npz")

        unit = "deg" if circular else ""
        print(f"\n  fold {f}: PCA k={k}, smoothing {smoothing:g}, {len(fold.test)} held-out clips "
              f"({len(held)} values, {int((~inside).sum())} extrapolated)")
        print(f"    E1 reconstruction   curve {row['recon_curve']:6.2f}   "
              f"nearest centroid {row['recon_nearest']:6.2f}   global mean {row['recon_mean']:6.2f}")
        print(f"    E3 curve vs line    curve {row['recon_curve']:6.2f}   "
              f"straight line {row['recon_line']:6.2f}   "
              f"{'CURVE WINS' if row['recon_curve'] < row['recon_line'] else 'no gain'}")
        print(f"    E2 unseen centroid  curve {row['centroid_curve']:6.2f}   "
              f"nearest centroid {row['centroid_nearest']:6.2f}")
        print(f"    E4 coordinate       {row['coordinate_error']:6.2f} {unit}   (probe-free readout)")
        print(f"    chord strays {row['chord_stray_relative']:.2f}x the curve's radius")

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "evaluation.csv", index=False)
    pd.concat(per_fold, ignore_index=True).to_csv(out_dir / "smoothing_trace.csv", index=False)
    summary = results.drop(columns=["fold"]).agg(["mean", "std"]).T
    summary.to_csv(out_dir / "summary.csv")

    m, s = results.mean(), results.std(ddof=1)
    print(f"\n  over {len(folds)} folds")
    print(f"    reconstruction  curve {m.recon_curve:6.2f} +/- {s.recon_curve:.2f}   "
          f"line {m.recon_line:6.2f} +/- {s.recon_line:.2f}   "
          f"({100*(1-m.recon_curve/m.recon_line):+.1f}% vs the line)")
    print(f"    unseen centroid curve {m.centroid_curve:6.2f} +/- {s.centroid_curve:.2f}   "
          f"nearest {m.centroid_nearest:6.2f}")
    print(f"    coordinate      {m.coordinate_error:6.2f} +/- {s.coordinate_error:.2f}")

    run = {"variable": args.variable, "layer": layer, "circular": circular,
           "centroid_variance": mf["centroid_variance"], "smoothing_grid": grid,
           "k_per_fold": results.k.tolist(), "smoothing_per_fold": results.smoothing.tolist(),
           "n_folds": len(folds), "minutes": round((time.time() - started) / 60, 2),
           "features": str(feat_dir), "git": git_state()}
    (out_dir / "run.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    print(f"\ndone in {run['minutes']} min -> {out_dir}")
    if m.recon_curve >= m.recon_line:
        print("note: the curve does not beat a straight line for this variable -- the "
              "curvature is not doing any work, and spline steering cannot be expected to "
              "improve on linear steering here.")


if __name__ == "__main__":
    main()
