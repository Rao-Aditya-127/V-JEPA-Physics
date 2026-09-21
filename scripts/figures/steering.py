"""Part 1.3: the steering figures (paper Fig. 24).

    python scripts/figures/steering.py

Reads the three steering runs from 04_steering.py and draws:

    steering_<variable>.png   error to target and to truth, against probes steered
    steering_overlay.png      all three, each normalised by its own unsteered error

It also prints the comparison against the paper's Fig. 24 numbers (direction only --
the paper never steers speed or acceleration) and the multi-target check, if the runs
were made with --multi-target. Needs no GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # works without pip install

import numpy as np
import pandas as pd

from vjepa_physics.config import load_config
from vjepa_physics.plots import POLAR_STYLE, steering_figure, steering_overlay

UNITS = {"speed": " m/s", "acceleration": " m/s²", "direction": "°"}
PAPER_FIG24 = {"direction": {"baseline": 82.9, "at_20_probes": 11.9}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = cfg["paths"]["artifacts"] / "results"
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    headline, written = {}, []
    print(f"{'variable':14s} {'unsteered':>10s} {'best':>10s} {'at':>10s} {'closed':>8s}  paper Fig. 24")
    for variable in POLAR_STYLE:
        base = results / cfg["variables"][variable]["dataset"] / "steering"
        if not (base / "summary.csv").exists():
            sys.exit(f"missing {base/'summary.csv'}: run 04_steering.py --variable {variable}")
        summary = pd.read_csv(base / "summary.csv")
        target = cfg["steering"]["target"][variable]
        # Only probe counts every fold reached -- the tail is averaged over fewer folds
        # and its error bands would not be comparable.
        head = summary[np.isclose(summary.target, target) &
                       (summary.n_folds == summary.n_folds.max())].sort_values("n_probes")
        headline[variable] = head

        unsteered = float(head.mae_to_target_mean.iloc[0])
        best = head.loc[head.mae_to_target_mean.idxmin()]
        paper = PAPER_FIG24.get(variable)
        note = (f"{paper['baseline']} -> {paper['at_20_probes']} at 20 probes" if paper
                else "[OURS] not measured in the paper")
        print(f"{variable:14s} {unsteered:10.2f} {best.mae_to_target_mean:10.2f} "
              f"{int(best.n_probes):7d} pr {1 - best.mae_to_target_mean/unsteered:7.1%}  {note}")

        written.append(steering_figure(head, figures / f"steering_{variable}.png", variable,
                                       UNITS[variable], target, baseline=unsteered))

    written.append(steering_overlay(headline, figures / "steering_overlay.png"))

    # The honesty check: error to the clip's own label must rise as error to the target falls.
    print("\nas error to the target falls, error to the true label must rise:")
    for variable, head in headline.items():
        best = head.loc[head.mae_to_target_mean.idxmin()]
        print(f"  {variable:13s} to target {head.mae_to_target_mean.iloc[0]:6.2f} -> "
              f"{best.mae_to_target_mean:6.2f}   to truth {head.mae_to_truth_mean.iloc[0]:6.2f} -> "
              f"{best.mae_to_truth_mean:6.2f}   "
              f"{'OK' if best.mae_to_truth_mean > head.mae_to_truth_mean.iloc[0] else 'NOT RISING'}")

    # Robustness: does the result depend on which target was chosen?
    for variable in POLAR_STYLE:
        base = results / cfg["variables"][variable]["dataset"] / "steering"
        summary = pd.read_csv(base / "summary.csv")
        if summary.target.nunique() < 2:
            continue
        closed = []
        for target, group in summary.groupby("target"):
            group = group[group.n_folds == group.n_folds.max()].sort_values("n_probes")
            closed.append(1 - group.mae_to_target_mean.min() / group.mae_to_target_mean.iloc[0])
        print(f"  {variable:13s} over {len(closed)} targets: gap closed "
              f"{np.mean(closed):.1%} +/- {np.std(closed):.1%}")

    print()
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
