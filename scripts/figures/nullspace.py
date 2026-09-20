"""Part 1.2: the combined nullspace figures (paper Figs. 4c, 22, 23).

    python scripts/figures/nullspace.py

Reads the three nullspace runs from 03_nullspace.py and draws:

    nullspace.png        all three variables' R2 against dimensions removed
    nullspace_zoom.png   the first 40 dimensions, where the paper's sawtooth would be
    sawtooth.png         direction alone, with the accuracy-within-15 degrees of Fig. 4c

It also prints a dimensionality table against the paper's Table 3, and a jaggedness
statistic: the paper reports that direction's curve is a sawtooth and speed's is not
(Fig. 23), so the claim is worth measuring rather than eyeballing. Needs no GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # works without pip install

import numpy as np
import pandas as pd

from vjepa_physics.config import load_config
from vjepa_physics.plots import POLAR_STYLE, nullspace_figure, sawtooth_figure

PAPER_TABLE3_LAYER8 = {"speed": 28, "direction": 136, "acceleration": None}


def jaggedness(summary: pd.DataFrame) -> dict:
    """How much the curve zig-zags, beyond its overall downward trend.

    A sawtooth alternates: down, up, down, up. `sign_flips` counts how often the
    round-to-round change reverses, as a fraction of the rounds -- near 0.5 for a
    clean sawtooth, near 0 for a smooth decline. `mean_rise` is the average size of
    the upward steps, which a monotone curve does not have at all.
    """
    r2 = summary.sort_values("dims_removed").test_r2_mean.to_numpy()
    step = np.diff(r2)
    if len(step) < 2:
        return {"sign_flips": np.nan, "mean_rise": np.nan, "rises": 0}
    flips = int((np.sign(step[1:]) * np.sign(step[:-1]) < 0).sum())
    rises = step[step > 0]
    return {"sign_flips": flips / len(step), "mean_rise": float(rises.mean()) if len(rises) else 0.0,
            "rises": int(len(rises))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = cfg["paths"]["artifacts"] / "results"
    summaries, dims = {}, []
    for variable in POLAR_STYLE:
        base = results / cfg["variables"][variable]["dataset"] / "nullspace"
        if not (base / "summary.csv").exists():
            sys.exit(f"missing {base/'summary.csv'}: run 03_nullspace.py --variable {variable}")
        summaries[variable] = pd.read_csv(base / "summary.csv")
        table = pd.read_csv(base / "dimensionality.csv").assign(variable=variable)
        dims.append(table)

    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    layer = cfg["nullspace"]["layer"]
    full = nullspace_figure(summaries, figures / "nullspace.png",
                            title=f"V-JEPA 2-L layer {layer}: iterative nullspace projection")
    zoom = nullspace_figure(summaries, figures / "nullspace_zoom.png", xmax=40, ylim=(0.0, 1.02),
                            title=f"Layer {layer}: the first 40 dimensions removed")
    saw = sawtooth_figure(summaries["direction"], figures / "sawtooth.png", xmax=40,
                          title=f"Layer {layer}: direction, round by round")

    print(f"effective dimensionality at layer {layer}  (mean +/- std over 5 grouped folds)\n")
    print(f"{'variable':14s} {'rule':6s} {'R2 <':>6s} {'K':>12s} {'dimensions':>12s}  paper Table 3")
    frame = pd.concat(dims, ignore_index=True)
    for (variable, rule), group in frame.groupby(["variable", "rule"], sort=False):
        paper = PAPER_TABLE3_LAYER8[variable]
        note = f"{paper}" if paper else "not measured"
        capped = "" if group.reached.all() else f"  [{(~group.reached).sum()} fold(s) hit the cap]"
        print(f"{variable:14s} {rule:6s} {group.threshold.iloc[0]:>6} "
              f"{group.K.mean():6.1f} +/- {group.K.std(ddof=1):3.1f} "
              f"{group.dims.mean():6.1f} +/- {group.dims.std(ddof=1):3.1f}  {note}{capped}")

    print("\njaggedness of the curve (paper Fig. 23: direction sawtooths, speed does not)")
    print(f"{'variable':14s} {'sign flips':>12s} {'upward steps':>14s} {'mean rise':>12s}")
    for variable, summary in summaries.items():
        j = jaggedness(summary)
        print(f"{variable:14s} {j['sign_flips']:11.2f} {j['rises']:14d} {j['mean_rise']:12.4f}")

    print(f"\nwrote {full}\nwrote {zoom}\nwrote {saw}")


if __name__ == "__main__":
    main()
