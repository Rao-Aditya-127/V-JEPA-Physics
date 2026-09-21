"""Part 1.3: multi-probe subspace steering (paper Appendix C.12), one variable.

    python scripts/04_steering.py --variable direction
    python scripts/04_steering.py --variable speed --multi-target 8

Overwrite a physical variable in the activations and ask an independent reader what
it sees. The steering subspace comes from Part 1.2's probe sequence, so run
03_nullspace.py first.

The protocol is C.12 p. 33, which exists to stop the measurement being circular --
solving for the steering probes and then asking those probes what they read proves
only that the solve worked:

    1. split            the fold's grouped 80/20, as in Parts 1.1 and 1.2
    2. steering probes  Part 1.2's sequence, trained on the 80%
    3. evaluation probe trained on the held-out 20% ONLY
    4. steer            the held-out activations, using the training-set subspace
    5. evaluate         what does the evaluation probe read?

For every probe count N the sweep records the error to the target and the error to
the clip's true label. The second must RISE as the first falls: if both improved we
would be writing the target into a corner of the space nothing else uses, rather than
overwriting the representation.

Direction reproduces paper Fig. 24. Speed and acceleration are [OURS] -- the paper
never steers them -- and exist because Part 2 has to compare spline steering against
this method for all three.

Needs no GPU.
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
from vjepa_physics.nullspace import predict
from vjepa_physics.probes import circular_mae, fit_probe, r2_score, sincos
from vjepa_physics.steering import build, displacement


def evaluation_probe(X_test: np.ndarray, y_test: np.ndarray, epochs: int, probing: dict,
                     device: str, half: bool, seed: int):
    """C.12 step 3: a probe trained on held-out clips only, to judge the steering.

    It must never have seen the steering probes or the activations that built the
    subspace -- otherwise "the probe reads the target" says nothing.

    `half` splits the held-out clips again, fitting on one half and judging on the
    other. The paper fits on all of them and judges on the same ones; that is the
    default so the numbers stay comparable, but it leaves the evaluator free to have
    memorised the very clips it scores.
    """
    index = np.arange(len(X_test))
    if half:
        index = np.random.default_rng(seed).permutation(index)
        fit_idx, judge_idx = np.sort(index[: len(index) // 2]), np.sort(index[len(index) // 2:])
    else:
        fit_idx = judge_idx = index
    probe = fit_probe(X_test[fit_idx], y_test[fit_idx], lr=probing["learning_rate"],
                      wd=probing["weight_decay"], epochs=epochs,
                      batch_size=probing["batch_size"], seed=seed, device=device)
    r2 = float(r2_score(y_test[judge_idx].reshape(len(judge_idx), -1),
                        predict(probe, X_test[judge_idx])))
    return probe, judge_idx, r2


def errors(pred: np.ndarray, target_value: float, truth: np.ndarray, circular: bool) -> tuple:
    """Mean absolute error to the steering target, and to the clips' own labels."""
    if circular:
        to_target = circular_mae(np.full(len(pred), target_value), pred)
        to_truth = circular_mae(truth, pred)
    else:
        flat = pred.ravel()
        to_target = float(np.abs(flat - target_value).mean())
        to_truth = float(np.abs(flat - truth).mean())
    return to_target, to_truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variable", default="direction", choices=["speed", "acceleration", "direction"])
    parser.add_argument("--target", type=float, default=None,
                        help="value to steer every clip to (default: config, 90 deg for direction)")
    parser.add_argument("--multi-target", type=int, default=0,
                        help="[OURS] also sweep this many evenly spaced targets, to check the "
                             "result is not special to one of them")
    parser.add_argument("--max-probes", type=int, default=None, help="cap the sweep")
    parser.add_argument("--eval-half", action="store_true",
                        help="[OURS] fit the evaluation probe on half the held-out clips and "
                             "judge on the other half; the paper uses all of them for both")
    parser.add_argument("--run", default="nullspace", help="which Part 1.2 run to take probes from")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    var_cfg, probing, ns = cfg["variables"][args.variable], cfg["probing"], cfg["nullspace"]
    steer_cfg = cfg["steering"]
    device = args.device or probing["device"]
    layer, epochs = ns["layer"], var_cfg["epochs"]
    circular = var_cfg.get("circular", False)
    probe_cfg = {"learning_rate": ns["learning_rate"], "weight_decay": ns["weight_decay"],
                 "batch_size": probing["batch_size"]}

    feat_dir = features_dir(cfg, var_cfg["dataset"], args.limit)
    feats = load_features(feat_dir)
    label = feats.labels[var_cfg["target"]].to_numpy(dtype=np.float64)
    X_all = feats.layer(layer)

    results = cfg["paths"]["artifacts"] / "results" / feat_dir.name
    source = results / args.run
    if not (source / "bases.npz").exists():
        sys.exit(f"missing {source/'bases.npz'} -- run 03_nullspace.py --variable {args.variable}")
    saved = np.load(source / "bases.npz")
    if f"fold0_biases" not in saved:
        sys.exit(f"{source/'bases.npz'} predates saved biases; re-run 03_nullspace.py")
    rounds = pd.read_csv(source / "rounds.csv")

    out_dir = results / "steering"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    headline = args.target if args.target is not None else steer_cfg["target"][args.variable]
    targets = [headline]
    if args.multi_target:
        extra = (np.linspace(0, 360, args.multi_target, endpoint=False) if circular else
                 np.linspace(np.quantile(label, 0.1), np.quantile(label, 0.9), args.multi_target))
        targets += [float(t) for t in extra if not np.isclose(t, headline)]

    folds = grouped_folds(label, probing["n_folds"], probing["inner_val_every"])
    check_folds(folds, len(label), label)

    print(f"{feat_dir.name}: layer {layer}, steering probes from {args.run}, "
          f"{len(folds)} folds, target {headline}{' deg' if circular else ''}, "
          f"{len(targets)} target(s), evaluation probe on "
          f"{'half the' if args.eval_half else 'all'} held-out clips")

    started, rows, eval_r2s = time.time(), [], []
    for k, fold in enumerate(folds):
        W, b = saved[f"fold{k}_weights"], saved[f"fold{k}_biases"]
        basis = saved[f"fold{k}_basis"]
        ranks = rounds[rounds.fold == k].sort_values("round").rank_this_round.tolist()
        mean, std = saved[f"fold{k}_mean"], saved[f"fold{k}_std"]

        # The same standardisation the steering probes were fitted under -- training
        # statistics only, and the space the subspace lives in.
        X_test = (X_all[fold.test] - mean) / std
        truth = label[fold.test]
        y_test = sincos(truth) if circular else truth

        probe, judge, eval_r2 = evaluation_probe(X_test, y_test, epochs, probe_cfg, device,
                                                 args.eval_half, probing["seed"])
        eval_r2s.append(eval_r2)
        X_judge, truth_judge = X_test[judge], truth[judge]

        max_probes = min(args.max_probes or len(W), len(W))
        print(f"\n  fold {k}: {len(fold.test)} held-out clips, evaluation probe R2 = {eval_r2:.3f}, "
              f"{len(W)} steering probes available")

        for target in targets:
            y_star = sincos(np.array([target]))[0] if circular else np.array([target])
            base_pred = predict(probe, X_judge)
            to_target, to_truth = errors(base_pred, target, truth_judge, circular)
            rows.append({"fold": k, "target": target, "n_probes": 0, "dims": 0,
                         "mae_to_target": to_target, "mae_to_truth": to_truth,
                         "displacement": 0.0, "eval_r2": eval_r2})
            if target == headline:
                print(f"    no steering          MAE to target {to_target:7.2f}   "
                      f"to truth {to_truth:6.2f}")

            for n in range(1, max_probes + 1):
                plan = build(W, b, y_star, n, basis=basis, ranks=ranks)
                X_star = plan.apply(X_judge)
                to_target, to_truth = errors(predict(probe, X_star), target, truth_judge, circular)
                rows.append({"fold": k, "target": target, "n_probes": n, "dims": plan.dims,
                             "mae_to_target": to_target, "mae_to_truth": to_truth,
                             "displacement": displacement(X_judge, X_star), "eval_r2": eval_r2})
                if target == headline and n in (1, 5, 10, 20, max_probes):
                    print(f"    {n:3d} probes ({plan.dims:3d} dims)  MAE to target {to_target:7.2f}   "
                          f"to truth {to_truth:6.2f}   moved {rows[-1]['displacement']:.2f}x")

    sweep = pd.DataFrame(rows)
    sweep.to_csv(out_dir / "sweep.csv", index=False)
    metrics = ["mae_to_target", "mae_to_truth", "displacement"]
    aggregates = {f"{m}_{s}": (m, s) for m in metrics for s in ("mean", "std")}
    summary = (sweep.groupby(["target", "n_probes", "dims"])
               .agg(n_folds=("fold", "count"), **aggregates).reset_index())
    summary.to_csv(out_dir / "summary.csv", index=False)

    # Folds stop at different probe counts, so the tail of the sweep is averaged over
    # fewer and fewer of them. Report only where every fold contributed.
    head = summary[(summary.target == headline) &
                   (summary.n_folds == summary.n_folds.max())].sort_values("n_probes")
    best = head.loc[head.mae_to_target_mean.idxmin()]
    print(f"\n  best: {int(best.n_probes)} probes ({int(best.dims)} dims) -> "
          f"MAE {best.mae_to_target_mean:.2f} +/- {best.mae_to_target_std:.2f} to target, "
          f"from {head.mae_to_target_mean.iloc[0]:.2f} unsteered")

    run = {"variable": args.variable, "layer": layer, "probes_from": str(source),
           "targets": targets, "headline_target": headline, "eval_half": args.eval_half,
           "evaluation_probe_r2": {"mean": float(np.mean(eval_r2s)), "per_fold": eval_r2s},
           "n_folds": len(folds), "probe": probe_cfg, "epochs": epochs,
           "minutes": round((time.time() - started) / 60, 2), "git": git_state()}
    (out_dir / "run.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    print(f"\ndone in {run['minutes']} min -> {out_dir}")
    if np.mean(eval_r2s) < 0.9:
        print(f"note: the evaluation probe only reaches R2 {np.mean(eval_r2s):.3f} unsteered "
              "-- it is a weak judge, so read the steering numbers with care.")


if __name__ == "__main__":
    main()
