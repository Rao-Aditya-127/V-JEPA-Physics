"""Part 1.2: our version of the paper's Figure 4c (direction encoding redundancy).

    python scripts/figures/figure4c.py

Reads the direction run produced by

    python scripts/03_nullspace.py --variable direction --dims-per-round 1 \\
                                   --max-rounds 100 --no-stop

-- one component removed per round, a fixed 100 probes, no early stop, which is what
Fig. 4c plots -- and draws the two scorings side by side:

    left    the probe, then the SAME probe re-scored after its own readout subspace
            has been removed, interleaved. A sawtooth.
    right   only ever a freshly retrained probe. Smooth.

Same activations, same removals, same metric; the panels differ only in when the
probe is scored. The paper reads its sawtooth as evidence of paired sin/cos features,
so the comparison is the point of the figure.

Also prints the arithmetic floor a collapsed probe scores, which is where the dips
land: with the sin readout zeroed, atan2 can only emit two angles, so accuracy is the
fraction of the dataset's directions lying within the tolerance of either one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # works without pip install

import numpy as np
import pandas as pd

from vjepa_physics.config import features_dir, load_config
from vjepa_physics.plots import figure4c


def dead_component_floor(angles: np.ndarray, tol: float = 15.0) -> float:
    """Accuracy a probe scores once one of its two outputs is zeroed.

    atan2(0, cos) is 0 deg or 180 deg and nothing else, so the probe is right exactly
    on the directions within `tol` of those two. 8 equally spaced directions give
    2/8 = 25%; our 64 give 15.6%.
    """
    reachable = np.array([0.0, 180.0])
    within = [np.abs((angles - r + 180.0) % 360.0 - 180.0) <= tol for r in reachable]
    return float((within[0] | within[1]).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="nullspace_dims1_nostop",
                        help="results subdirectory written by 03_nullspace.py")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    var_cfg = cfg["variables"]["direction"]
    base = cfg["paths"]["artifacts"] / "results" / var_cfg["dataset"] / args.run
    if not (base / "summary.csv").exists():
        sys.exit(f"missing {base/'summary.csv'} -- run:\n"
                 "  python scripts/03_nullspace.py --variable direction "
                 "--dims-per-round 1 --max-rounds 100 --no-stop")
    summary = pd.read_csv(base / "summary.csv")
    if "stale_acc15_mean" not in summary:
        sys.exit(f"{base/'summary.csv'} predates the stale-scoring columns; re-run 03_nullspace.py")

    labels = pd.read_csv(features_dir(cfg, var_cfg["dataset"]) / "labels.csv")
    angles = labels[var_cfg["target"]].to_numpy(dtype=float)
    floor = dead_component_floor(np.unique(angles))

    figures = cfg["paths"]["artifacts"] / "results" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    probe = "ridge" if "ridge" in args.run else "Adam (C.11)"
    name = "fig4c_direction" + ("_ridge" if "ridge" in args.run else "")
    out = figure4c(summary, figures / f"{name}.png", floor=floor,
                   title=f"V-JEPA 2-L layer {cfg['nullspace']['layer']}: direction encoding "
                         f"redundancy  [{probe} probe]")

    rounds = summary[summary.n_folds == summary.n_folds.max()]
    fresh, stale = rounds.test_acc15_mean.to_numpy(), rounds.stale_acc15_mean.to_numpy()
    print(f"{len(rounds)} probes x {int(rounds.n_folds.max())} folds, one component removed per round\n")
    print(f"  retrained : {100*fresh[0]:5.1f}% at probe 0 -> {100*fresh[-1]:5.1f}% at probe {len(fresh)-1}")
    print(f"  stale     : {100*stale.mean():5.1f}% on average, spread "
          f"{100*stale.std():.1f}  (a floor, not a measurement)")
    print(f"  arithmetic floor for {len(np.unique(angles))} directions: {100*floor:.1f}%"
          f"   (8 directions, as in the paper, would give 25.0%)")

    def period2(series: np.ndarray) -> float:
        detrended = series - np.poly1d(np.polyfit(np.arange(len(series)), series, 3))(np.arange(len(series)))
        power = np.abs(np.fft.rfft(detrended)) ** 2
        return float(power[-1] / power[1:].mean())

    interleaved = np.empty(2 * len(fresh))
    interleaved[0::2], interleaved[1::2] = fresh, stale
    print(f"\n  sawtooth strength (power at period 2, relative to the average frequency):")
    print(f"    retrained only : {period2(fresh):7.2f}x   (a sawtooth needs >> 1)")
    print(f"    interleaved    : {period2(interleaved):7.2f}x")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
