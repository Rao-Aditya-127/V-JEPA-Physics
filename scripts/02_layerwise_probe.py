"""Part 1.1: layer-wise linear probing (paper Appendix B), one variable.

    python scripts/02_layerwise_probe.py --variable speed               # the brief's experiment
    python scripts/02_layerwise_probe.py --variable speed --limit 100   # debug subset

For every paper layer 0..23 and every one of 5 grouped folds, train the 20-config
Adam sweep on the fit split, choose on the inner validation split, report on the
held-out fold. By default only that runs:

    main       encoder features, grouped CV       -> the reproduction (Figure 1)

plus a convergence check (U6): does doubling the epochs change the answer? That
check is part of doing the core correctly, not an extra experiment.

Additional checks beyond the brief are opt-in, e.g.
`--conditions main embedding random_cv shuffled pixels`:

    embedding  [OURS] the patch embedding, before block 0
    random_cv  [OURS] ungrouped CV                -> how much seeing test values helps
    shuffled   [OURS] labels permuted             -> no signal: R^2 at or below 0
    pixels     [OURS] raw downsampled frames      -> what the encoder adds

When any of them are in the results, controls.png and error_by_value.png are drawn too.

Needs no GPU -- it reads the cached features from 01_extract.py.
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
from vjepa_physics.features import layer_fraction, load_features
from vjepa_physics.folds import check_folds, grouped_folds, random_folds, save_folds
from vjepa_physics.plots import controls_figure, error_by_value_figure, layerwise_figure
from vjepa_physics.probes import standardize, sweep

UNITS = {"speed": "m/s", "acceleration": "m/s²"}
CORE = ("main",)
EXTRAS = ("embedding", "random_cv", "shuffled", "pixels")     # [OURS], beyond the brief
CONDITIONS = CORE + EXTRAS


def evaluate(features: dict, y: np.ndarray, folds, probing: dict, epochs: int, device: str,
             condition: str, num_layers: int, per_feature: bool = True):
    """Run the sweep for every feature set x fold. Returns per-fold rows and
    out-of-fold predictions (each clip predicted by the fold that held it out)."""
    rows, oof = [], {}
    lrs, wds = probing["learning_rates"], probing["weight_decays"]
    for layer, X in features.items():
        started = time.time()
        pred = np.full(len(y), np.nan)
        for k, fold in enumerate(folds):
            Xf, Xv, Xt = standardize(X[fold.fit], X[fold.val], X[fold.test], per_feature=per_feature)
            result = sweep(Xf, y[fold.fit], Xv, y[fold.val], Xt, y[fold.test],
                           learning_rates=lrs, weight_decays=wds, epochs=epochs,
                           batch_size=probing["batch_size"], seed=probing["seed"], device=device)
            best = result.best
            lr, wd = result.configs[best]
            pred[fold.test] = result.test_pred[best][:, 0]
            rows.append({
                "condition": condition, "layer": layer,
                "layer_fraction": layer_fraction(layer, num_layers) if layer is not None else np.nan,
                "fold": k, "lr": lr, "wd": wd, "lr_at_grid_edge": lr in (min(lrs), max(lrs)),
                "val_r2": float(result.val_r2[best]), "test_r2": float(result.test_r2[best]),
                "test_mae": float(result.test_mae[best]),
                "n_fit": len(fold.fit), "n_val": len(fold.val), "n_test": len(fold.test),
            })
        oof[layer] = pred
        r2 = [r["test_r2"] for r in rows if r["layer"] == layer and r["condition"] == condition]
        name = "pixels" if layer is None else ("embedding" if layer == -1 else f"layer {layer:2d}")
        print(f"  {condition:9s} {name:10s}  R2 = {np.mean(r2):6.3f} +/- {np.std(r2, ddof=1):.3f}"
              f"   ({time.time() - started:4.1f} s)")
    return rows, oof


def convergence_check(pooled_layer, y, fold, probing, epochs, device, layers):
    """U6: if doubling the epochs moves R^2, the sweep is measuring optimisation."""
    rows = []
    for layer in layers:
        X = pooled_layer(layer)
        Xf, Xv, Xt = standardize(X[fold.fit], X[fold.val], X[fold.test])
        out = {"layer": layer}
        for mult in (1, 2):
            result = sweep(Xf, y[fold.fit], Xv, y[fold.val], Xt, y[fold.test],
                           learning_rates=probing["learning_rates"],
                           weight_decays=probing["weight_decays"], epochs=epochs * mult,
                           batch_size=probing["batch_size"], seed=probing["seed"], device=device)
            out[f"test_r2_{mult}x"] = float(result.test_r2[result.best])
        out["delta"] = out["test_r2_2x"] - out["test_r2_1x"]
        rows.append(out)
        print(f"  layer {layer:2d}: R2 {out['test_r2_1x']:.3f} at {epochs} epochs -> "
              f"{out['test_r2_2x']:.3f} at {2 * epochs}  (delta {out['delta']:+.3f})")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variable", default="speed", choices=["speed", "acceleration"])
    parser.add_argument("--limit", type=int, default=None, help="use features from a --limit extraction")
    parser.add_argument("--conditions", nargs="+", default=list(CORE), choices=CONDITIONS,
                        help="default: main only. Extras: " + ", ".join(EXTRAS))
    parser.add_argument("--control-stride", type=int, default=1,
                        help="probe every Nth layer for random_cv/shuffled (1 = all)")
    parser.add_argument("--no-convergence-check", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    var_cfg, probing = cfg["variables"][args.variable], cfg["probing"]
    device = args.device or probing["device"]
    epochs = var_cfg["epochs"]

    feat_dir = features_dir(cfg, var_cfg["dataset"], args.limit)
    feats = load_features(feat_dir)
    y = feats.labels[var_cfg["target"]].to_numpy(dtype=np.float64)
    clip_ids = feats.labels.clip_id.to_numpy()
    L = feats.num_layers
    out_dir = cfg["paths"]["artifacts"] / "results" / feat_dir.name
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    splits_dir = cfg["paths"]["artifacts"] / "splits"
    print(f"{feat_dir.name}: {len(y)} clips, {len(np.unique(y))} distinct {var_cfg['target']} values, "
          f"{L} layers, device={device}, epochs={epochs}")

    # Folds -- built once, checked, written to disk.
    grouped = grouped_folds(y, probing["n_folds"], probing["inner_val_every"])
    check_folds(grouped, len(y), y)
    save_folds(grouped, clip_ids, splits_dir / f"{feat_dir.name}_grouped.json", "grouped by label value")
    ungrouped = random_folds(len(y), probing["n_folds"], seed=probing["seed"])
    check_folds(ungrouped, len(y))
    save_folds(ungrouped, clip_ids, splits_dir / f"{feat_dir.name}_random.json", "random over clips")

    all_layers = list(range(L))
    control_layers = all_layers[::args.control_stride]
    rows, oof_main = [], {}
    started = time.time()

    if "main" in args.conditions:
        print("\nmain: paper layers 0..23, grouped CV")
        r, oof_main = evaluate({l: feats.layer(l) for l in all_layers}, y, grouped, probing,
                               epochs, device, "main", L)
        rows += r
    if "embedding" in args.conditions:
        print("\nembedding [OURS]: patch embedding before block 0, grouped CV")
        r, _ = evaluate({-1: feats.embeddings()}, y, grouped, probing, epochs, device, "embedding", L)
        rows += r
    if "random_cv" in args.conditions:
        print("\nrandom_cv [OURS]: ungrouped CV")
        r, _ = evaluate({l: feats.layer(l) for l in control_layers}, y, ungrouped, probing,
                        epochs, device, "random_cv", L)
        rows += r
    if "shuffled" in args.conditions:
        print("\nshuffled [OURS]: permuted labels, grouped by the permuted values")
        y_shuffled = np.random.default_rng(probing["seed"]).permutation(y)
        folds_shuffled = grouped_folds(y_shuffled, probing["n_folds"], probing["inner_val_every"])
        r, _ = evaluate({l: feats.layer(l) for l in control_layers}, y_shuffled, folds_shuffled,
                        probing, epochs, device, "shuffled", L)
        rows += r
    if "pixels" in args.conditions:
        print("\npixels [OURS]: raw downsampled frames, grouped CV")
        r, _ = evaluate({None: feats.pixels}, y, grouped, probing, epochs, device, "pixels", L,
                        per_feature=False)   # shared scale: see probes.standardize
        rows += r

    # Merge with earlier results: re-running one condition replaces only that condition.
    per_fold = pd.DataFrame(rows)
    per_fold_path = out_dir / "per_fold.csv"
    if per_fold_path.exists():
        earlier = pd.read_csv(per_fold_path)
        kept = earlier[~earlier.condition.isin(args.conditions)]
        if len(kept):
            print(f"\nkeeping earlier results for: {sorted(kept.condition.unique())}")
        per_fold = pd.concat([kept, per_fold], ignore_index=True)
    per_fold.to_csv(per_fold_path, index=False)
    summary = (per_fold.groupby(["condition", "layer", "layer_fraction"], dropna=False)
               .agg(r2_mean=("test_r2", "mean"), r2_std=("test_r2", "std"),
                    mae_mean=("test_mae", "mean"), mae_std=("test_mae", "std"),
                    n_folds=("fold", "count"), lr_edge_picks=("lr_at_grid_edge", "sum"))
               .reset_index())
    summary.to_csv(out_dir / "summary.csv", index=False)

    convergence = None
    if not args.no_convergence_check:
        print("\nconvergence check (U6), fold 0")
        convergence = convergence_check(feats.layer, y, grouped[0], probing, epochs, device,
                                        [l for l in (0, 8, 16) if l < L])
        convergence.to_csv(out_dir / "convergence.csv", index=False)

    figures = out_dir / "figures"
    oof_path = out_dir / "oof_main.npz"
    if oof_main:
        np.savez(oof_path, y=y, clip_id=clip_ids,
                 layers=np.array(list(oof_main)), predictions=np.stack(list(oof_main.values())))
    extras_present = summary.condition.isin(EXTRAS).any()
    if (summary.condition == "main").any():
        layerwise_figure(summary, figures / "layerwise.png", args.variable, L)
        if extras_present:
            controls_figure(summary, figures / "controls.png", args.variable, L)
    if extras_present and oof_path.exists():
        saved = np.load(oof_path)
        error_by_value_figure(saved["y"], dict(zip(saved["layers"].tolist(), saved["predictions"])),
                              figures / "error_by_value.png", args.variable, UNITS[args.variable])

    edge = int(per_fold.lr_at_grid_edge.sum())
    run = {"features": str(feat_dir), "features_meta": feats.meta, "variable": args.variable,
           "target": var_cfg["target"], "epochs": epochs, "probing": probing,
           "conditions_in_results": sorted(per_fold.condition.unique()),
           "conditions_this_run": args.conditions, "control_stride": args.control_stride,
           "n_clips": int(len(y)), "minutes": round((time.time() - started) / 60, 2),
           "selections_at_lr_grid_edge": f"{edge}/{len(per_fold)}"}
    (out_dir / "run.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")

    print(f"\ndone in {run['minutes']} min -> {out_dir}")
    # Grid-edge picks matter only where there is signal. Under shuffled labels the least-trained
    # config (lowest lr) overfits least, so it is *expected* to win there.
    for condition, group in per_fold.groupby("condition"):
        n_edge = int(group.lr_at_grid_edge.sum())
        if n_edge and condition != "shuffled":
            lrs = sorted(group.loc[group.lr_at_grid_edge, "lr"].unique())
            print(f"note: {condition}: {n_edge}/{len(group)} selections at the lr grid edge {lrs} -- "
                  "the optimum may lie outside the paper's range.")


if __name__ == "__main__":
    main()
