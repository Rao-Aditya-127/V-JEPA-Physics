"""Part 2, stage 2: steer along the manifold, and compare against Part 1.3.

    python scripts/06_manifold_steering.py --variable direction

Every held-out clip is moved from its own value to a target, in 21 steps, by three
methods:

    linear      x + t (s(v1) - s(v0))        Goodfire eq. 1, the straight route
    manifold    x + s(v(t)) - s(v0)          Goodfire eq. 2, along the fitted curve
    subspace    V c*(v(t)) + x_perp          Part 1.3's method, re-solved at each step

The first two agree exactly at t = 0 and t = 1 and differ only in between, which is
the paper's whole claim -- same endpoints, different journeys. So the headline measure
is not where a path arrives but what happens on the way: at each step a probe trained
only on held-out clips is asked what it reads, and compared against where the clip is
supposed to be.

`subspace` is an extension of paper C.12 rather than C.12 itself: that method jumps
straight to a target, and is turned into a path here so the three can share an axis.

Results are binned by how far each clip had to travel, because the methods are
expected to agree for short moves and diverge for long ones.

Needs no GPU. Run 05_manifold.py and 03_nullspace.py first.
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
    choose_smoothing, components_for_variance, fit_manifold, short_delta, steering_step,
)
from vjepa_physics.nullspace import predict
from vjepa_physics.probes import circular_mae, fit_probe, r2_score, sincos
from vjepa_physics.steering import clean_weights, stack_probes, steering_basis


def read_value(probe, X, circular):
    """What the evaluation probe reads, as a value, plus its confidence.

    For direction the probe outputs a vector whose LENGTH says how much directional
    signal is present at all -- it collapses towards zero in states the encoder never
    produces. The scalars have no such quantity, so confidence is undefined there.
    """
    out = predict(probe, X)
    if circular:
        return (np.degrees(np.arctan2(out[:, 0], out[:, 1])) % 360.0,
                np.linalg.norm(out, axis=1))
    return out.ravel(), np.full(len(out), np.nan)


def subspace_solver(weights, biases, basis, ranks, n_probes, circular):
    """Part 1.3's construction, prepared once so each target is a single solve.

    c* is linear in the target, so the QR and the matrix inverse are done once and
    every waypoint costs one matrix-vector product instead of a fresh factorisation.
    """
    W, b = stack_probes(clean_weights(weights, basis, ranks), biases, n_probes)
    V = steering_basis(W)
    A_inv = np.linalg.pinv(W @ V)
    outputs = W.shape[0] // min(n_probes, len(weights))

    def steer(X, values):
        y = sincos(np.asarray(values)) if circular else np.asarray(values)[:, None]
        targets = np.tile(y, (1, len(W) // outputs))             # (n, K*outputs)
        c_star = (targets - b) @ A_inv.T                         # (n, dims)
        return (X - (X @ V) @ V.T) + c_star @ V.T

    return steer, V.shape[1]


def best_probe_count(results_dir: Path, target: float, fallback: int = 20) -> int:
    """How many probes Part 1.3 found best for this variable -- the fairest version of
    its method to compare against, rather than a number picked here."""
    path = results_dir / "steering" / "summary.csv"
    if not path.exists():
        return fallback
    summary = pd.read_csv(path)
    head = summary[np.isclose(summary.target, target) &
                   (summary.n_folds == summary.n_folds.max())]
    return int(head.loc[head.mae_to_target_mean.idxmin()].n_probes) if len(head) else fallback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variable", default="direction", choices=["speed", "acceleration", "direction"])
    parser.add_argument("--target", type=float, default=None)
    parser.add_argument("--waypoints", type=int, default=21)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    var_cfg, probing, mf, ns = (cfg["variables"][args.variable], cfg["probing"],
                                cfg["manifold"], cfg["nullspace"])
    layer, epochs = ns["layer"], var_cfg["epochs"]
    circular = var_cfg.get("circular", False)
    period = 360.0 if circular else None
    target = args.target if args.target is not None else cfg["steering"]["target"][args.variable]

    feat_dir = features_dir(cfg, var_cfg["dataset"])
    feats = load_features(feat_dir)
    label = feats.labels[var_cfg["target"]].to_numpy(dtype=np.float64)
    X_all = feats.layer(layer).astype(np.float64)

    results = cfg["paths"]["artifacts"] / "results" / feat_dir.name
    out_dir = results / "manifold_steering"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = np.load(results / "nullspace" / "bases.npz")
    rounds = pd.read_csv(results / "nullspace" / "rounds.csv")
    n_probes = best_probe_count(results, target)

    folds = grouped_folds(label, probing["n_folds"], probing["inner_val_every"])
    check_folds(folds, len(label), label)
    ts = np.linspace(0.0, 1.0, args.waypoints)
    unit = "deg" if circular else ""

    print(f"{feat_dir.name}: layer {layer}, target {target}{unit}, {args.waypoints} waypoints, "
          f"{len(folds)} folds, subspace method using {n_probes} probes (Part 1.3's best)")

    started, rows, eval_r2s = time.time(), [], []
    for f, fold in enumerate(folds):
        train = np.sort(np.concatenate([fold.fit, fold.val]))
        mean, std = X_all[train].mean(0), X_all[train].std(0) + 1e-6
        X_fit, X_val = (X_all[fold.fit] - mean) / std, (X_all[fold.val] - mean) / std
        X_train, X_test = (X_all[train] - mean) / std, (X_all[fold.test] - mean) / std
        y_train, y_test = label[train], label[fold.test]

        k = components_for_variance(X_train, y_train, mf["centroid_variance"])
        smoothing, _ = choose_smoothing(X_fit, label[fold.fit], X_val, label[fold.val], k=k,
                                        grid=[float(s) for s in mf["smoothing_grid"]],
                                        periodic=circular, period=period)
        manifold = fit_manifold(X_train, y_train, k=k, smoothing=smoothing,
                                periodic=circular, period=period)

        # The judge: trained on held-out clips only, so it knows nothing of the
        # manifold, the steering probes, or the activations that built either.
        y_probe = sincos(y_test) if circular else y_test
        probe = fit_probe(X_test, y_probe, lr=ns["learning_rate"], wd=ns["weight_decay"],
                          epochs=epochs, batch_size=probing["batch_size"],
                          seed=probing["seed"], device=probing["device"])
        eval_r2 = float(r2_score(y_probe.reshape(len(y_test), -1), predict(probe, X_test)))
        eval_r2s.append(eval_r2)

        ranks = rounds[rounds.fold == f].sort_values("round").rank_this_round.tolist()
        subspace, dims = subspace_solver(saved[f"fold{f}_weights"], saved[f"fold{f}_biases"],
                                         saved[f"fold{f}_basis"], ranks, n_probes, circular)
        journey = np.abs(short_delta(y_test, target, period))
        print(f"\n  fold {f}: {len(y_test)} clips, evaluation probe R2 {eval_r2:.3f}, "
              f"PCA k={k}, subspace {dims} dims, journeys {journey.min():.1f}-{journey.max():.1f}{unit}")

        for t in ts:
            for method in ("linear", "manifold", "subspace"):
                if method == "subspace":
                    intended = y_test + t * short_delta(y_test, target, period)
                    steered = subspace(X_test, intended)
                else:
                    steered, intended = steering_step(manifold, X_test, y_test, target, t, method)
                got, confidence = read_value(probe, steered, circular)
                err = (np.abs((got - intended + 180) % 360 - 180) if circular
                       else np.abs(got - intended))
                to_target = (np.abs((got - target + 180) % 360 - 180) if circular
                             else np.abs(got - target))
                rows.append(pd.DataFrame({
                    "fold": f, "method": method, "t": t, "journey": journey,
                    "tracking_error": err, "error_to_target": to_target,
                    "confidence": confidence, "eval_r2": eval_r2}))

        mid = [r for r in rows[-3 * len(ts):]]
        del mid

    per_clip = pd.concat(rows, ignore_index=True)
    per_clip.to_csv(out_dir / "paths.csv.gz", index=False, compression="gzip")

    # Binned by how far the clip had to travel: the methods should agree for short
    # moves and diverge for long ones.
    edges = ([0, 45, 90, 135, 180.01] if circular else
             list(np.quantile(per_clip.journey, [0, 0.25, 0.5, 0.75, 1.0]) + [0, 0, 0, 0, 1e-9]))
    per_clip["journey_bin"] = pd.cut(per_clip.journey, edges, include_lowest=True)
    summary = (per_clip.groupby(["method", "t"], observed=True)
               .agg(tracking_error=("tracking_error", "mean"),
                    tracking_std=("tracking_error", "std"),
                    error_to_target=("error_to_target", "mean"),
                    confidence=("confidence", "mean"), n=("tracking_error", "size")).reset_index())
    summary.to_csv(out_dir / "summary.csv", index=False)
    binned = (per_clip.groupby(["method", "t", "journey_bin"], observed=True)
              .agg(tracking_error=("tracking_error", "mean"),
                   confidence=("confidence", "mean"), n=("tracking_error", "size")).reset_index())
    binned.to_csv(out_dir / "summary_by_journey.csv", index=False)

    print(f"\n  mean tracking error along the path (how far the readout is from where "
          f"the clip should be)")
    print(f"    {'t':>5s}" + "".join(f"{m:>12s}" for m in ("linear", "manifold", "subspace")))
    for t in ts[::4]:
        line = f"    {t:5.2f}"
        for method in ("linear", "manifold", "subspace"):
            v = summary[(summary.method == method) & np.isclose(summary.t, t)].tracking_error
            line += f"{float(v.iloc[0]):12.2f}"
        print(line)

    worst = summary[np.isclose(summary.t, 0.5)].set_index("method").tracking_error
    print(f"\n  at the midpoint: linear {worst['linear']:.2f}{unit}   "
          f"manifold {worst['manifold']:.2f}{unit}   subspace {worst['subspace']:.2f}{unit}")
    if circular:
        conf = summary[np.isclose(summary.t, 0.5)].set_index("method").confidence
        print(f"  confidence there: linear {conf['linear']:.3f}   manifold {conf['manifold']:.3f}"
              f"   subspace {conf['subspace']:.3f}   (a real clip is about 1.0)")

    run = {"variable": args.variable, "layer": layer, "target": target,
           "waypoints": args.waypoints, "n_probes_subspace": n_probes,
           "evaluation_probe_r2": {"mean": float(np.mean(eval_r2s)), "per_fold": eval_r2s},
           "n_folds": len(folds), "minutes": round((time.time() - started) / 60, 2),
           "git": git_state()}
    (out_dir / "run.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    print(f"\ndone in {run['minutes']} min -> {out_dir}")


if __name__ == "__main__":
    main()
