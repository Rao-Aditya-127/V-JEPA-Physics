"""Part 1.2: iterative nullspace probing (paper Appendix C.11), one variable.

    python scripts/03_nullspace.py --variable speed
    python scripts/03_nullspace.py --variable direction --limit 100   # debug subset

At one layer (8 by default, the Physics Emergence Zone of paper Fig. 4c), train a
probe, delete the subspace it reads from, and train another, until the score reaches
chance. The number of rounds survived is the variable's effective dimensionality.

Unlike Part 1.1 there is no hyperparameter sweep: C.11 fixes lr = 1e-3 and
wd = 1e-4. The round-0 check compares that fixed setting against the configuration
Part 1.1's sweep selected at this layer, so a curve that starts low because the
probe is badly tuned cannot be mistaken for a variable that is hard to read (N2).

The run repeats over the same 5 grouped folds as Part 1.1 -- each is a grouped
80/20 split whose test values never appear in training -- which gives the error
bands the paper's Figures 4c and 23 show but C.11's single fixed split cannot.

Needs no GPU: it reads the cached features from 01_extract.py.
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
from vjepa_physics.folds import check_folds, grouped_folds, random_folds
from vjepa_physics.nullspace import dimensionality, predict, run_sequence
from vjepa_physics.plots import nullspace_figure
from vjepa_physics.probes import fit_probe, r2_score, standardize


def round_zero_check(X_fit, y_fit, X_test, y_test, c11: tuple[float, float],
                     swept: tuple[float, float] | None, epochs: int, probing: dict,
                     device: str) -> dict:
    """N2: does C.11's fixed (lr, wd) read the variable as well as Part 1.1's choice?

    C.11 uses wd = 1e-4, a hundred times below the smallest weight decay in the
    Appendix B grid that Part 1.1 swept -- an inconsistency inside the paper. If the
    fixed setting scores much lower here, every dimensionality number afterwards is
    an underestimate, so it is worth three seconds to find out.
    """
    out = {}
    for name, (lr, wd) in [("c11", c11)] + ([("swept", swept)] if swept else []):
        probe = fit_probe(X_fit, y_fit, lr=lr, wd=wd, epochs=epochs,
                          batch_size=probing["batch_size"], seed=probing["seed"], device=device)
        out[name] = {"lr": lr, "wd": wd,
                     "test_r2": float(r2_score(y_test.reshape(len(y_test), -1),
                                               predict(probe, X_test)))}
    if "swept" in out:
        out["gap"] = out["swept"]["test_r2"] - out["c11"]["test_r2"]
    return out


def swept_config(results_dir: Path, layer: int) -> tuple[float, float] | None:
    """The (lr, wd) Part 1.1's sweep picked most often at this layer, if it ran."""
    path = results_dir / "per_fold.csv"
    if not path.exists():
        return None
    rows = pd.read_csv(path)
    rows = rows[(rows.condition == "main") & (rows.layer == layer)]
    if rows.empty:
        return None
    lr, wd = rows.groupby(["lr", "wd"]).size().idxmax()
    return float(lr), float(wd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variable", default="speed", choices=["speed", "acceleration", "direction"])
    parser.add_argument("--layer", type=int, default=None, help="paper layer (default: config)")
    parser.add_argument("--max-rounds", type=int, default=None, help="override the config cap")
    parser.add_argument("--folds", type=int, default=None, help="run only the first N folds")
    parser.add_argument("--probe", default="adam", choices=["adam", "ridge"],
                        help="[OURS] diagnostic: ridge is closed-form, so its direction "
                             "carries no leftover random initialisation")
    parser.add_argument("--dims-per-round", type=int, default=None,
                        help="[OURS] diagnostic: delete only this many basis columns per "
                             "round (1 splits direction's sin/cos pair across rounds)")
    parser.add_argument("--no-stop", action="store_true",
                        help="run the full round cap instead of stopping at chance, as paper Fig. 4c does")
    parser.add_argument("--ungrouped", action="store_true",
                        help="C.11's literal split: random 80/20, so test values are seen in training")
    parser.add_argument("--limit", type=int, default=None, help="use features from a --limit extraction")
    parser.add_argument("--device", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    var_cfg, probing, ns = cfg["variables"][args.variable], cfg["probing"], cfg["nullspace"]
    device = args.device or probing["device"]
    layer = args.layer if args.layer is not None else ns["layer"]
    epochs = var_cfg["epochs"]
    circular = var_cfg.get("circular", False)
    lr, wd = ns["learning_rate"], ns["weight_decay"]
    max_rounds = args.max_rounds or ns["max_rounds"][args.variable]
    stop_r2, patience = ns["stop_r2"][args.variable], ns["patience"]
    if args.no_stop:
        stop_r2 = float("-inf")

    feat_dir = features_dir(cfg, var_cfg["dataset"], args.limit)
    feats = load_features(feat_dir)
    label = feats.labels[var_cfg["target"]].to_numpy(dtype=np.float64)
    from vjepa_physics.probes import sincos                       # local: only direction needs it
    y = sincos(label) if circular else label
    X_all = feats.layer(layer)

    results = cfg["paths"]["artifacts"] / "results" / feat_dir.name
    # Diagnostics never overwrite the faithful C.11 run: they get their own directory.
    tag = "".join(["" if args.probe == "adam" else f"_{args.probe}",
                   "" if args.dims_per_round is None else f"_dims{args.dims_per_round}",
                   "_nostop" if args.no_stop else ""])
    out_dir = results / f"nullspace{tag}"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    folds = (random_folds(len(label), probing["n_folds"], seed=probing["seed"]) if args.ungrouped
             else grouped_folds(label, probing["n_folds"], probing["inner_val_every"]))
    check_folds(folds, len(label), None if args.ungrouped else label)
    if args.folds:
        folds = folds[:args.folds]

    print(f"{feat_dir.name}: layer {layer}, {len(label)} clips, "
          f"{'random' if args.ungrouped else 'grouped'} 80/20 x {len(folds)} folds, "
          f"probe={args.probe}"
          f"{'' if args.dims_per_round is None else f', {args.dims_per_round} dim(s)/round'}, "
          f"Adam lr={lr} wd={wd}, {epochs} epochs, stop at R2<{stop_r2} for {patience} rounds, "
          f"cap {max_rounds}, device={device}")

    started = time.time()
    all_rounds, dims_rows, saved, checks = [], [], {}, {}
    for k, fold in enumerate(folds):
        # No inner validation split: C.11 fixes the hyperparameters, so fit + val are
        # both training data and the 80/20 ratio is the paper's.
        train = np.sort(np.concatenate([fold.fit, fold.val]))
        X_train, X_test = standardize(X_all[train], X_all[fold.test])
        y_train, y_test = y[train], y[fold.test]
        angles = label[fold.test] if circular else None
        print(f"\n  fold {k}: train {len(train)}, test {len(fold.test)}")

        if k == 0 and args.probe == "adam":
            checks = round_zero_check(X_train, y_train, X_test, y_test, (lr, wd),
                                      swept_config(results, layer), epochs, probing, device)
            line = f"    round-0 check (N2): C.11 lr={lr} wd={wd} -> R2 {checks['c11']['test_r2']:.3f}"
            if "swept" in checks:
                s = checks["swept"]
                line += (f" | 1.1's lr={s['lr']} wd={s['wd']} -> R2 {s['test_r2']:.3f}"
                         f" | gap {checks['gap']:+.3f}")
            print(line)

        seq = run_sequence(X_train, y_train, X_test, y_test, lr=lr, wd=wd, epochs=epochs,
                           batch_size=probing["batch_size"], seed=probing["seed"],
                           max_rounds=max_rounds, stop_r2=stop_r2, patience=patience,
                           angles_test=angles, device=device, probe_fit=args.probe,
                           dims_per_round=args.dims_per_round)
        rounds = seq.rounds.assign(fold=k)
        all_rounds.append(rounds)
        for name, threshold in ns["thresholds"][args.variable].items():
            dims_rows.append({"fold": k, "rule": name, **dimensionality(rounds, threshold, patience)})
        # Everything Part 1.3 needs to steer in this subspace, in activation units.
        saved[f"fold{k}_basis"] = seq.basis
        saved[f"fold{k}_weights"] = np.stack(seq.weights)
        saved[f"fold{k}_mean"] = X_all[train].mean(axis=0)
        saved[f"fold{k}_std"] = X_all[train].std(axis=0) + 1e-6
        saved[f"fold{k}_test_clips"] = feats.labels.clip_id.to_numpy()[fold.test]

    per_round = pd.concat(all_rounds, ignore_index=True)
    per_round.to_csv(out_dir / "rounds.csv", index=False)

    metrics = ["test_r2", "train_r2", "test_mae", "mae_vs_baseline", "leakage", "variance_removed"]
    metrics += [c for c in ("test_circ_mae", "test_acc15") if c in per_round]
    metrics += [c for c in per_round.columns if c.startswith("stale_")]   # the same probe, re-scored
    aggregates = {f"{m}_{stat}": (m, stat) for m in metrics for stat in ("mean", "std")}
    summary = (per_round.groupby(["round", "dims_removed"])
               .agg(n_folds=("fold", "count"), **aggregates).reset_index())
    summary.to_csv(out_dir / "summary.csv", index=False)

    dims = pd.DataFrame(dims_rows)
    dims.to_csv(out_dir / "dimensionality.csv", index=False)
    np.savez_compressed(out_dir / "bases.npz", **saved)

    print("\n  effective dimensionality (mean over folds)")
    for rule, group in dims.groupby("rule"):
        reached = "" if group.reached.all() else f"  [{(~group.reached).sum()} fold(s) hit the cap]"
        print(f"    {rule:6s} R2<{group.threshold.iloc[0]:<5}  K = {group.K.mean():5.1f} "
              f"+/- {group.K.std(ddof=1):.1f}   dimensions = {group.dims.mean():5.1f}{reached}")

    figure = nullspace_figure({args.variable: summary}, out_dir / "figures" / "nullspace.png",
                              title=f"V-JEPA 2-L layer {layer}: {args.variable}")

    run = {"variable": args.variable, "layer": layer, "features": str(feat_dir),
           "features_meta": feats.meta, "n_clips": int(len(label)), "circular": circular,
           "split": "random 80/20" if args.ungrouped else "grouped 80/20", "n_folds": len(folds),
           "lr": lr, "weight_decay": wd, "epochs": epochs, "batch_size": probing["batch_size"],
           "seed": probing["seed"], "max_rounds": max_rounds, "stop_r2": stop_r2,
           "patience": patience, "thresholds": ns["thresholds"][args.variable],
           "probe": args.probe, "dims_per_round": args.dims_per_round,
           "round_zero_check": checks, "rounds_run": {int(k): int(v) for k, v in
                                                      per_round.groupby("fold").size().items()},
           "minutes": round((time.time() - started) / 60, 2), "git": git_state()}
    (out_dir / "run.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")

    print(f"\ndone in {run['minutes']} min -> {out_dir}\nwrote {figure}")
    if checks.get("gap", 0) > 0.05:
        print(f"note: C.11's fixed hyperparameters score {checks['gap']:.3f} below Part 1.1's "
              "selected ones at round 0 -- the dimensionality below is a lower bound.")
    hit_cap = per_round.groupby("fold").size().eq(max_rounds)
    if hit_cap.any():
        print(f"note: {int(hit_cap.sum())}/{len(folds)} folds reached the {max_rounds}-round cap "
              "without falling to chance -- K is a lower bound; raise --max-rounds.")


if __name__ == "__main__":
    main()
