"""Figures, in one consistent style.

Colours follow the validated reference palette (dataviz skill): categorical slots
in fixed order, a one-hue ramp for ordered series, text in ink tokens rather than
series colours, recessive grid and axes, one y-axis per chart.

    Figure 1  layerwise.png   the reproduction curve (paper Fig. 2c, one variable)
    Figure 2  controls.png    [OURS] the same curve against its controls
    Figure 3  error_by_value.png  [OURS] held-out error vs. true label value
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"            # categorical slots 1-3
ORDINAL_BLUES = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]      # ramp steps 250/400/550/700

PAPER_LAYER = 8                                                  # the paper's emergence-zone marker


def _style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11, loc="left", fontweight="bold", pad=16)


def _reference_lines(ax, num_layers: int) -> None:
    ax.axhline(0, color=AXIS, linewidth=1.2, zorder=1)
    x8 = PAPER_LAYER / (num_layers - 1)
    ax.axvline(x8, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    # Label sits above the plot frame (x in data units, y in axes units), clear of all data.
    ax.text(x8, 1.0, "layer 8", transform=ax.get_xaxis_transform(), color=MUTED,
            fontsize=8, ha="center", va="bottom")


def _save(fig, path: Path) -> Path:
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")   # keeps outside legends in frame
    plt.close(fig)
    return path


def _curve(ax, frame: pd.DataFrame, color: str, label: str, marker: str = "o",
           dashed: bool = False, z: int = 3) -> None:
    frame = frame.sort_values("layer_fraction")
    x, mean, std = frame.layer_fraction, frame.r2_mean, frame.r2_std.fillna(0)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0, zorder=z - 1)
    ax.plot(x, mean, color=color, linewidth=2, linestyle=(0, (5, 3)) if dashed else "-",
            marker=marker, markersize=4.5, markerfacecolor=SURFACE if dashed else color,
            markeredgecolor=color if dashed else SURFACE, markeredgewidth=1, label=label, zorder=z)


def _ylim(ax, *series: pd.Series) -> None:
    low = min(float((s).min()) for s in series)
    ax.set_ylim(min(-0.05, low - 0.05), 1.02)


def layerwise_figure(summary: pd.DataFrame, path: Path, variable: str, num_layers: int = 24) -> Path:
    """Figure 1 -- the reproduction target: main condition, paper layers only."""
    main = summary[(summary.condition == "main") & (summary.layer >= 0)]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    _reference_lines(ax, num_layers)
    _curve(ax, main, BLUE, variable)
    _ylim(ax, main.r2_mean - main.r2_std)
    ax.set_xlim(-0.02, 1.02)
    _style(ax, "Layer fraction", "Held-out R²  (5-fold grouped CV, mean ± std)",
           f"V-JEPA 2-L: {variable}")
    return _save(fig, path)


def controls_figure(summary: pd.DataFrame, path: Path, variable: str, num_layers: int = 24) -> Path:
    """Figure 2 -- [OURS] the reproduction curve against its controls."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    _reference_lines(ax, num_layers)
    lows = []

    # Ungrouped first and dashed, so the reproduction curve stays visible on top when they coincide.
    random_cv = summary[summary.condition == "random_cv"]
    if len(random_cv):
        _curve(ax, random_cv, ORANGE, "ungrouped CV (test values seen in training)", marker="s",
               dashed=True, z=3)
        lows.append(random_cv.r2_mean - random_cv.r2_std)

    main = summary[(summary.condition == "main") & (summary.layer >= 0)]
    _curve(ax, main, BLUE, "grouped CV (reproduction)", z=5)
    lows.append(main.r2_mean - main.r2_std)

    shuffled = summary[summary.condition == "shuffled"]
    if len(shuffled):
        _curve(ax, shuffled, MUTED, "shuffled labels (no-signal null, expect ≤ 0)", marker="^")
        lows.append(shuffled.r2_mean - shuffled.r2_std)

    pixels = summary[summary.condition == "pixels"]
    if len(pixels):
        mean, std = float(pixels.r2_mean.iloc[0]), float(pixels.r2_std.iloc[0])
        ax.axhspan(mean - std, mean + std, color=AQUA, alpha=0.15, linewidth=0, zorder=1)
        ax.axhline(mean, color=AQUA, linewidth=2, linestyle=(0, (6, 3)), label="raw pixels", zorder=2)
        lows.append(pd.Series([mean - std]))

    embedding = summary[summary.condition == "embedding"]
    if len(embedding):
        x = -1 / (num_layers - 1)
        ax.errorbar(x, embedding.r2_mean, yerr=embedding.r2_std, fmt="D", color=BLUE,
                    markerfacecolor=SURFACE, markersize=6, markeredgewidth=1.5, capsize=0,
                    label="patch embedding (before block 0)", zorder=4)

    _ylim(ax, *lows)
    ax.set_xlim(-0.08, 1.02)
    _style(ax, "Layer fraction", "Held-out R²  (mean ± std over folds)", f"{variable}: controls")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2)
    return _save(fig, path)


def error_by_value_figure(y: np.ndarray, oof: dict[int, np.ndarray], path: Path, variable: str,
                          unit: str, layers=(0, 8, 16, 23)) -> Path:
    """Figure 3 -- [OURS] mean |error| at each true label value, from held-out predictions.
    Shows *where* in the label range the readout fails, which one averaged MAE hides."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    values = np.unique(y)
    for color, layer in zip(ORDINAL_BLUES, [l for l in layers if l in oof]):
        err = np.abs(oof[layer] - y)
        per_value = [err[y == v].mean() for v in values]
        ax.plot(values, per_value, color=color, linewidth=2, label=f"layer {layer}", zorder=3)
    ax.set_ylim(bottom=0)
    _style(ax, f"True {variable} ({unit})", f"Mean |error| ({unit}), held-out",
           f"{variable}: where the readout fails")
    legend = ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, title="paper layer",
                       title_fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    legend.get_title().set_color(INK_2)
    return _save(fig, path)
