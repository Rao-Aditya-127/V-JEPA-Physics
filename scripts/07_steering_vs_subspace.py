"""Part 2: spline steering against Part 1.3's subspace steering, head to head.

    python scripts/07_steering_vs_subspace.py --variable direction

The brief asks for this comparison specifically -- "compare spline steering with the
multi-probe subspace method from Part 1, including strengths, limitations, and
failure cases". 06_manifold_steering.py answered a different question (Goodfire's:
straight line or curve?) and judged everything with one linear probe. This script
compares the two methods on their own terms.

The methods differ in one respect that everything else follows from:

    subspace   x* = V c*(target) + x_perp     REPLACE: c* is the same for every clip
    manifold   x* = x + s(v1) - s(v0)         SHIFT:   the move depends on the start

Both are applied as a single jump to a target -- how Part 1.3 was built, and a
native use of the manifold too -- over the same targets Part 1.3's multi-target check
used, plus each clip's own value ("self": asked to change nothing). Every steered
clip is then judged five ways:

    probe_err    a linear probe trained on the held-out clips only (as 06 and 1.3)
    knn_err      a probe-free judge: the value of the 10 nearest real clips
    realism      distance to the nearest real clip, over the typical real clip's
    edit         how far the activation moved, as a fraction of its own length
    side shifts  how far probes for the OTHER variables (start position, motion
                 type, or direction) moved -- steering one variable should not
                 move them

and a third method is added for practicality: manifold steering needs the clip's
current value v0, which 06 took from the label. `manifold_est` estimates it with a
probe trained on the training clips, as anyone steering an unlabelled clip must.

Two judges, because each can favour one side: the subspace method is built to make
linear probes read the target, and the manifold is built from real clips' averages.
Side probes and the v0 probe are ridge (alpha = 100, the repo default) fitted on the
training clips; k = 10 for the neighbour judge. Both were fixed on unsteered clips
before any steering result was seen.

Needs no GPU. Run 03_nullspace.py and 05_manifold.py first (06 is not required).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works without pip install

import numpy as np
import pandas as pd

from vjepa_physics.config import features_dir, load_config
from vjepa_physics.extract import git_state
from vjepa_physics.features import load_features
from vjepa_physics.folds import check_folds, grouped_folds
from vjepa_physics.manifold import (
    choose_smoothing, components_for_variance, fit_manifold, short_delta, steering_step,
)
from vjepa_physics.nullspace import fit_ridge, predict
from vjepa_physics.probes import fit_probe, r2_score, sincos

K_NEIGHBOURS = 10
SIDE_ALPHA = 100.0


def _steering_helpers():
    """subspace_solver, best_probe_count and read_value from 06, loaded by path so
    both scripts steer with literally the same code."""
    spec = importlib.util.spec_from_file_location("manifold_steering",
                                                  ROOT / "scripts" / "06_manifold_steering.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def angle(out: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(out[:, 0], out[:, 1])) % 360.0


def gap(a, b, circular: bool) -> np.ndarray:
    return np.abs((a - b + 180.0) % 360.0 - 180.0) if circular else np.abs(a - b)


def targets_for(label: np.ndarray, headline: float, n: int, circular: bool) -> list[float]:
    """Part 1.3's multi-target set (04_steering.py), so both methods face the same targets."""
    extra = (np.linspace(0, 360, n, endpoint=False) if circular else
             np.linspace(np.quantile(label, 0.1), np.quantile(label, 0.9), n))
    return [headline] + [float(t) for t in extra if not np.isclose(t, headline)]


class NeighbourJudge:
    """Reads a value off the nearest real clips -- no probe, no training."""

    def __init__(self, X_ref: np.ndarray, y_ref: np.ndarray, circular: bool, k: int = K_NEIGHBOURS):
        self.X, self.y, self.circular, self.k = X_ref, y_ref, circular, k
        self.sq = (X_ref ** 2).sum(1)

    def __call__(self, Q: np.ndarray, exclude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Value read from the k nearest real clips, and the distance to the nearest.
        `exclude` is each query's own unsteered clip, which would otherwise trivially
        be among its neighbours."""
        d2 = (Q ** 2).sum(1)[:, None] - 2 * Q @ self.X.T + self.sq[None, :]
        d2[np.arange(len(Q)), exclude] = np.inf
        nearest = np.argpartition(d2, self.k, axis=1)[:, :self.k]
        values = self.y[nearest]
        if self.circular:
            r = np.radians(values)
            read = np.degrees(np.arctan2(np.sin(r).mean(1), np.cos(r).mean(1))) % 360.0
        else:
            read = np.median(values, axis=1)
        return read, np.sqrt(np.maximum(d2.min(axis=1), 0.0))


def side_variables(labels: pd.DataFrame, variable: str) -> dict:
    """The factors that vary inside this dataset but are NOT being steered.

    The direction dataset's speed field is unreliable (750 clips labelled 0 m/s move),
    so it is not used.
    """
    sides = {"start_x": ("metres", labels.start_x.to_numpy(float)),
             "start_y": ("metres", labels.start_y.to_numpy(float))}
    if variable == "direction":
        sides["motion"] = ("flip", (labels.motion == "acceleration").to_numpy(float))
    else:
        sides["direction"] = ("degrees", labels.theta_degrees.to_numpy(float))
    return sides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variable", default="direction", choices=["speed", "acceleration", "direction"])
    parser.add_argument("--multi-target", type=int, default=8)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    helpers = _steering_helpers()
    cfg = load_config(args.config)
    var_cfg, probing, mf, ns = (cfg["variables"][args.variable], cfg["probing"],
                                cfg["manifold"], cfg["nullspace"])
    layer, epochs = ns["layer"], var_cfg["epochs"]
    circular = var_cfg.get("circular", False)
    period = 360.0 if circular else None
    headline = cfg["steering"]["target"][args.variable]

    feat_dir = features_dir(cfg, var_cfg["dataset"])
    feats = load_features(feat_dir)
    label = feats.labels[var_cfg["target"]].to_numpy(dtype=np.float64)
    X_all = feats.layer(layer).astype(np.float64)
    sides = side_variables(feats.labels, args.variable)

    results = cfg["paths"]["artifacts"] / "results" / feat_dir.name
    out_dir = results / "steering_vs_subspace"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = np.load(results / "nullspace" / "bases.npz")
    rounds = pd.read_csv(results / "nullspace" / "rounds.csv")
    n_probes = helpers.best_probe_count(results, headline)
    targets = targets_for(label, headline, args.multi_target, circular)

    folds = grouped_folds(label, probing["n_folds"], probing["inner_val_every"])
    check_folds(folds, len(label), label)
    unit = "deg" if circular else ""
    print(f"{feat_dir.name}: layer {layer}, {len(targets)} targets + 'self', {len(folds)} folds, "
          f"subspace from {n_probes} probes (Part 1.3's best), judges: held-out probe + "
          f"{K_NEIGHBOURS} nearest real clips")

    started, rows, fold_info = time.time(), [], []
    for f, fold in enumerate(folds):
        train = np.sort(np.concatenate([fold.fit, fold.val]))
        mean, std = X_all[train].mean(0), X_all[train].std(0) + 1e-6
        Z = (X_all - mean) / std                                  # every clip, fold's scaling
        X_fit, X_val, X_train, X_test = Z[fold.fit], Z[fold.val], Z[train], Z[fold.test]
        y_train, y_test = label[train], label[fold.test]

        # ---- the two steering methods, built exactly as 06 builds them
        k = components_for_variance(X_train, y_train, mf["centroid_variance"])
        smoothing, _ = choose_smoothing(X_fit, label[fold.fit], X_val, label[fold.val], k=k,
                                        grid=[float(s) for s in mf["smoothing_grid"]],
                                        periodic=circular, period=period)
        manifold = fit_manifold(X_train, y_train, k=k, smoothing=smoothing,
                                periodic=circular, period=period)
        ranks = rounds[rounds.fold == f].sort_values("round").rank_this_round.tolist()
        subspace, dims = helpers.subspace_solver(saved[f"fold{f}_weights"], saved[f"fold{f}_biases"],
                                                 saved[f"fold{f}_basis"], ranks, n_probes, circular)

        # ---- judge 1: linear probe on the held-out clips only (06 / Part 1.3's judge)
        y_probe = sincos(y_test) if circular else y_test
        judge = fit_probe(X_test, y_probe, lr=ns["learning_rate"], wd=ns["weight_decay"],
                          epochs=epochs, batch_size=probing["batch_size"],
                          seed=probing["seed"], device=probing["device"])
        judge_r2 = float(np.mean(r2_score(y_probe.reshape(len(y_test), -1), predict(judge, X_test))))

        # ---- judge 2: the nearest real clips, among all clips of this dataset
        neighbours = NeighbourJudge(Z, label, circular)
        own = fold.test                                           # row of each test clip in Z
        real_read, real_nn = neighbours(X_test, own)
        typical_nn = float(np.median(real_nn))

        # ---- the practical manifold: v0 estimated by a probe fitted on training clips
        v0_probe = fit_ridge(X_train, sincos(y_train) if circular else y_train, alpha=SIDE_ALPHA)
        v0_out = predict(v0_probe, X_test)
        v0_est = angle(v0_out) if circular else np.clip(v0_out.ravel(), manifold.values[0],
                                                        manifold.values[-1])

        # ---- side probes: the variables that should NOT move
        side_probes, side_base = {}, {}
        for name, (kind, values) in sides.items():
            y_side = sincos(values[train]) if kind == "degrees" else values[train]
            probe = fit_ridge(X_train, y_side, alpha=SIDE_ALPHA)
            side_probes[name] = (kind, probe)
            side_base[name] = predict(probe, X_test)

        def side_shift(name, X_steered):
            kind, probe = side_probes[name]
            before, after = side_base[name], predict(probe, X_steered)
            if kind == "degrees":
                return gap(angle(after), angle(before), True)
            if kind == "flip":
                return ((after.ravel() > 0.5) != (before.ravel() > 0.5)).astype(float)
            return np.abs(after.ravel() - before.ravel())

        info = {"fold": f, "k": k, "smoothing": smoothing, "subspace_dims": dims,
                "judge_r2": judge_r2, "typical_nearest_real": typical_nn,
                "neighbour_err_on_real": float(gap(real_read, y_test, circular).mean()),
                "v0_probe_err": float(gap(v0_est, y_test, circular).mean())}
        for name, (kind, values) in sides.items():
            base = side_base[name]
            if kind == "degrees":
                info[f"side_{name}_err"] = float(gap(angle(base), values[fold.test], True).mean())
            elif kind == "flip":
                info[f"side_{name}_acc"] = float(((base.ravel() > 0.5) == (values[fold.test] > 0.5)).mean())
            else:
                info[f"side_{name}_r2"] = float(r2_score(values[fold.test], base.ravel()))
        fold_info.append(info)
        print(f"\n  fold {f}: {len(y_test)} clips, PCA k={k}, subspace {dims} dims, "
              f"judge R2 {judge_r2:.3f}, neighbours read real clips to {info['neighbour_err_on_real']:.2f}{unit}, "
              f"v0 probe {info['v0_probe_err']:.2f}{unit}")

        # ---- steer: every target, plus each clip's own value
        norm = np.linalg.norm(X_test, axis=1)
        for target in ["self"] + targets:
            goal = y_test if target == "self" else np.full(len(y_test), float(target))
            methods = {"manifold": steering_step(manifold, X_test, y_test, goal, 1.0, "manifold")[0],
                       "subspace": subspace(X_test, goal)}
            if target != "self":
                methods["manifold_est"] = steering_step(manifold, X_test, v0_est, goal, 1.0, "manifold")[0]
            for method, X_star in methods.items():
                read, conf = helpers.read_value(judge, X_star, circular)
                knn_read, nn_dist = neighbours(X_star, own)
                frame = {"fold": f, "method": method, "target": str(target),
                         "journey": np.abs(short_delta(y_test, goal, period)),
                         "probe_err": gap(read, goal, circular),
                         "knn_err": gap(knn_read, goal, circular),
                         "realism": nn_dist / typical_nn,
                         "edit": np.linalg.norm(X_star - X_test, axis=1) / norm,
                         "confidence": conf}
                for name in sides:
                    frame[f"side_{name}"] = side_shift(name, X_star)
                rows.append(pd.DataFrame(frame))

    per_clip = pd.concat(rows, ignore_index=True)
    per_clip.to_csv(out_dir / "per_clip.csv.gz", index=False, compression="gzip")
    per_clip["kind"] = np.where(per_clip.target == "self", "self", "targets")

    metrics = ["probe_err", "knn_err", "realism", "edit", "confidence"] + [f"side_{n}" for n in sides]
    summary = per_clip.groupby(["kind", "method"])[metrics].mean().reset_index()
    summary.to_csv(out_dir / "summary.csv", index=False)

    moved = per_clip[per_clip.kind == "targets"].copy()
    edges = ([0, 45, 90, 135, 180.01] if circular else
             list(np.quantile(moved.journey, [0, 0.25, 0.5, 0.75, 1.0]) + [0, 0, 0, 0, 1e-9]))
    moved["journey_bin"] = pd.cut(moved.journey, edges, include_lowest=True)
    binned = moved.groupby(["journey_bin", "method"], observed=True)[metrics].mean().reset_index()
    binned.to_csv(out_dir / "summary_by_journey.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\n  steering to the {len(targets)} targets (mean over clips and targets)")
    print(summary[summary.kind == "targets"].drop(columns="kind").round(3).to_string(index=False))
    print(f"\n  asked to change nothing (target = the clip's own value)")
    print(summary[summary.kind == "self"].drop(columns="kind").round(3).to_string(index=False))
    print(f"\n  by distance travelled: probe_err / knn_err / realism")
    print(binned[["journey_bin", "method", "probe_err", "knn_err", "realism"]].round(3).to_string(index=False))

    run = {"variable": args.variable, "layer": layer, "targets": targets,
           "n_probes_subspace": n_probes, "k_neighbours": K_NEIGHBOURS, "side_alpha": SIDE_ALPHA,
           "sides": list(sides), "folds": fold_info,
           "minutes": round((time.time() - started) / 60, 2), "git": git_state()}
    (out_dir / "run.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    print(f"\ndone in {run['minutes']} min -> {out_dir}")


if __name__ == "__main__":
    main()
