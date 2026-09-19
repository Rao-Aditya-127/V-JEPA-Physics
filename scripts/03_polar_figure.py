"""Part 1.1: the combined layer-wise figure, reproducing the paper's Figure 2c.

    python scripts/03_polar_figure.py

Reads the layer-wise probing results for speed, direction and acceleration (from
02_layerwise_probe.py) and draws all three curves on one axis, styled after the
paper so the two figures can be compared side by side. Writes two versions: the
full 0-1 scale, as in the paper, and a zoomed view (R² 0.75-1.00) that shows the
detail. Needs no GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without pip install

import pandas as pd

from vjepa_physics.config import load_config
from vjepa_physics.plots import POLAR_STYLE, polar_figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = cfg["paths"]["artifacts"] / "results"
    summaries = {}
    for variable in POLAR_STYLE:
        path = results / cfg["variables"][variable]["dataset"] / "summary.csv"
        if not path.exists():
            sys.exit(f"missing {path}: run 02_layerwise_probe.py --variable {variable} first")
        summary = pd.read_csv(path)
        summaries[variable] = summary[(summary.condition == "main") & (summary.layer >= 0)]
        if len(summaries[variable]) != cfg["model"]["num_layers"]:
            sys.exit(f"{path} has {len(summaries[variable])} layers, "
                     f"expected {cfg['model']['num_layers']}")

    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    num_layers = cfg["model"]["num_layers"]

    # Full scale: the figure to set beside the paper's.
    full = polar_figure(summaries, figures / "fig2c_polar.png", num_layers)
    # Zoomed: shows the detail the full scale hides. The title says it's zoomed,
    # because the axis no longer starts at zero and differences look larger.
    zoom = polar_figure(summaries, figures / "fig2c_polar_zoom.png", num_layers,
                        ylim=(0.75, 1.0), yticks=(0.75, 0.80, 0.85, 0.90, 0.95, 1.00),
                        title="V-JEPA 2-L: Polar (zoomed, R² 0.75–1.00)")
    print(f"wrote {full}\nwrote {zoom}")


if __name__ == "__main__":
    main()
