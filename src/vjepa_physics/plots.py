"""Figures, in one consistent style.

Colours follow the validated reference palette (dataviz skill): categorical slots
in fixed order, a one-hue ramp for ordered series, text in ink tokens rather than
series colours, recessive grid and axes, one y-axis per chart.

    Figure 1  layerwise.png   the reproduction curve (paper Fig. 2c, one variable)
    Figure 2  controls.png    [OURS] the same curve against its controls
    Figure 3  error_by_value.png  [OURS] held-out error vs. true label value
    Fig. 2c   fig2c_polar.png  all three variables on one axis, styled after the paper
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
RED = "#e34948"                                                  # categorical slot 8
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


# Styled after the paper's Figure 2c so the two can be compared side by side:
# the same colour per variable, thick lines, ±1 std bands, a dashed layer-8 line in
# the legend, a boxed legend, a full frame and 0-1 axes. Two deliberate differences:
# the y-label says "Held-out" (our scores come from test folds never used for model
# selection), and staggered markers let colour-blind readers tell the lines apart
# (red and teal are close under deuteranopia; the palette validator flags ΔE 6.9).
POLAR_STYLE = {                       # variable -> (colour, marker, legend label)
    "speed": (BLUE, "o", "Speed"),
    "direction": (RED, "^", "Direction"),
    "acceleration": (AQUA, "s", "Acceleration"),
}


def polar_figure(summaries: dict[str, pd.DataFrame], path: Path, num_layers: int = 24,
                 ylim: tuple[float, float] = (0.0, 1.05),
                 yticks: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                 title: str = "V-JEPA 2-L: Polar") -> Path:
    """The reproduction of the paper's Figure 2c: every variable's layer-wise curve.

    `summaries` maps a variable name to its summary.csv, main condition only. The
    defaults give the paper's full 0-1 scale; a narrower `ylim` gives a zoomed view,
    and its `title` should then say so, since the axis no longer starts at zero.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=150)
    for offset, (variable, (color, marker, label)) in enumerate(POLAR_STYLE.items()):
        frame = summaries[variable].sort_values("layer_fraction")
        x, mean, std = frame.layer_fraction, frame.r2_mean, frame.r2_std.fillna(0)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=2.5, marker=marker, markersize=5.5,
                markevery=(offset, 3), label=label)            # staggered: markers never coincide
    ax.axvline(PAPER_LAYER / (num_layers - 1), color="#9a9a9a", linestyle="--", linewidth=1.3,
               label=f"Layer {PAPER_LAYER}")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(*ylim)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks(list(yticks))
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(INK_2)
    ax.tick_params(colors=INK_2, labelcolor=INK)
    ax.set_xlabel("Layer Fraction", color=INK, fontsize=12, fontweight="bold")
    ax.set_ylabel("Held-out R²", color=INK, fontsize=12, fontweight="bold")
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=10, labelcolor=INK)
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    return _save(fig, path)


def nullspace_figure(summaries: dict[str, pd.DataFrame], path: Path,
                     title: str = "Iterative nullspace projection",
                     ylim: tuple[float, float] = (-0.05, 1.05),
                     thresholds: dict[str, float] | None = None,
                     xmax: int | None = None) -> Path:
    """Part 1.2 (paper Fig. 4c / 23): held-out R² against dimensions removed.

    `summaries` maps a variable to its nullspace summary.csv. Only rounds where every
    fold is still running are drawn, so the band always covers the same folds. A
    curve that drops at once means the variable lives in a few directions; a slow
    decay means it is written redundantly. Direction's sawtooth -- the paper's
    signature of sin/cos feature pairs -- shows up as the jagged red line.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.3), dpi=150)
    for variable, frame in summaries.items():
        color, marker, label = POLAR_STYLE.get(variable, (BLUE, "o", variable))
        frame = frame[frame.n_folds == frame.n_folds.max()].sort_values("dims_removed")
        if xmax is not None:
            frame = frame[frame.dims_removed <= xmax]
        x, mean = frame.dims_removed, frame.test_r2_mean
        std = frame.test_r2_std.fillna(0)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(x, mean, color=color, linewidth=1.8, label=label.capitalize(), zorder=3,
                marker=marker, markersize=3.5, markevery=max(1, len(frame) // 20))
    for name, level in (thresholds or {}).items():
        ax.axhline(level, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.text(ax.get_xlim()[1], level, name, color=MUTED, fontsize=8, ha="right", va="bottom")
    ax.axhline(0, color=AXIS, linewidth=1.2, zorder=1)
    ax.set_ylim(*ylim)
    _style(ax, "Dimensions removed", "Held-out R²  (mean ± std over folds)", title)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    return _save(fig, path)


def sawtooth_figure(summary: pd.DataFrame, path: Path, xmax: int = 40,
                    title: str = "Direction: the first rounds up close") -> Path:
    """Part 1.2, direction only (paper Fig. 4c): R² and accuracy within 15° over the
    first few rounds, where the sawtooth is visible. The paper's Figure 4c uses
    accuracy within 15°, a metric it never defines; R² is on the same axis for
    comparison with the other variables."""
    frame = summary[(summary.n_folds == summary.n_folds.max()) &
                    (summary.dims_removed <= xmax)].sort_values("dims_removed")
    fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=150)
    for column, color, label in [("test_r2", RED, "Held-out R²"),
                                 ("test_acc15", BLUE, "Accuracy within 15°")]:
        if f"{column}_mean" not in frame:
            continue
        mean, std = frame[f"{column}_mean"], frame[f"{column}_std"].fillna(0)
        ax.fill_between(frame.dims_removed, mean - std, mean + std, color=color, alpha=0.18,
                        linewidth=0, zorder=2)
        ax.plot(frame.dims_removed, mean, color=color, linewidth=2, marker="o", markersize=4,
                label=label, zorder=3)
    ax.axhline(2 * 15 / 360, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(xmax, 2 * 15 / 360, "chance @15°", color=MUTED, fontsize=8, ha="right", va="bottom")
    ax.axhline(0, color=AXIS, linewidth=1.2, zorder=1)
    ax.set_ylim(-0.05, 1.05)
    _style(ax, "Dimensions removed", "Score", title)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    return _save(fig, path)


def figure4c(summary: pd.DataFrame, path: Path, metric: str = "acc15", floor: float | None = None,
             title: str = "Direction encoding redundancy") -> Path:
    """Part 1.2, direction: our version of the paper's Figure 4c.

    Two curves from the same run, differing only in WHEN the probe is scored:

      retrained   a fresh probe fitted on the current activations -- what the
                  representation still supports
      interleaved that score alternating with the same probe re-scored after its own
                  readout subspace has been projected out. The second kind cannot vary:
                  the weighted sum is zero, so the probe emits its bias, which for
                  direction is one fixed angle. Accuracy therefore drops to an
                  arithmetic floor (2 reachable angles out of however many the dataset
                  has) rather than to anything about the encoder.

    The interleaved curve is a sawtooth; the retrained one is smooth. The paper reads
    its sawtooth as evidence of paired sin/cos features, so plotting both together is
    the clearest way to show what else can produce it.
    """
    frame = summary[summary.n_folds == summary.n_folds.max()].sort_values("round")
    fresh, stale = frame[f"test_{metric}_mean"].to_numpy(), frame[f"stale_{metric}_mean"].to_numpy()
    fresh_sd = frame[f"test_{metric}_std"].fillna(0).to_numpy()
    stale_sd = frame[f"stale_{metric}_std"].fillna(0).to_numpy()
    rounds = frame["round"].to_numpy()

    x = np.empty(2 * len(rounds));      x[0::2], x[1::2] = rounds, rounds + 0.5
    y = np.empty_like(x);               y[0::2], y[1::2] = fresh, stale
    sd = np.empty_like(x);              sd[0::2], sd[1::2] = fresh_sd, stale_sd

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=150, sharey=True)
    for i, (ax, (xs, ys, sds, color, head)) in enumerate(zip(axes, [
            (x, y, sd, BLUE, "Scored before and after removal, interleaved"),
            (rounds, fresh, fresh_sd, RED, "Scored only after retraining")])):
        ax.fill_between(xs, 100 * (ys - sds), 100 * (ys + sds), color=color, alpha=0.2, linewidth=0)
        ax.plot(xs, 100 * ys, color=color, linewidth=1.4)
        if floor is not None:
            ax.axhline(100 * floor, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
            if i:            # label once, on the panel where the data does not sit on it
                ax.text(rounds.max(), 100 * floor, f"dead-component floor {100 * floor:.1f}% ",
                        color=MUTED, fontsize=8, ha="right", va="bottom")
        ax.set_xlim(0, rounds.max())
        ax.set_ylim(0, 105)
        _style(ax, "Orthogonal probe number", "" if i else "Accuracy within 15°  (%)", head)
    fig.suptitle(title, color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    return _save(fig, path)


def steering_figure(summary: pd.DataFrame, path: Path, variable: str, unit: str,
                    target: float, baseline: float | None = None,
                    title: str | None = None) -> Path:
    """Part 1.3 (paper Fig. 24): steering error against the number of probes used.

    Two curves, and the second is the point. `mae_to_target` falls as more directions
    are steered together; `mae_to_truth` must RISE by roughly as much. If both fell,
    the target would be going into a corner of the space the evaluation probe reads
    but the representation does not use -- the intervention would be writing a note
    to the probe rather than changing the encoded variable.

    All errors come from a probe trained only on held-out clips, which saw neither the
    steering probes nor the activations that built the subspace.
    """
    frame = summary.sort_values("n_probes")
    fig, ax = plt.subplots(figsize=(6.8, 4.3), dpi=150)
    for column, color, label in [("mae_to_target", BLUE, f"error to the target ({target:g}{unit})"),
                                 ("mae_to_truth", ORANGE, "error to the clip's true value")]:
        mean, std = frame[f"{column}_mean"], frame[f"{column}_std"].fillna(0)
        ax.fill_between(frame.n_probes, mean - std, mean + std, color=color, alpha=0.18,
                        linewidth=0, zorder=2)
        ax.plot(frame.n_probes, mean, color=color, linewidth=2, label=label, zorder=3)
    if baseline is not None:
        ax.axhline(baseline, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.text(frame.n_probes.max(), baseline, "unsteered ", color=MUTED, fontsize=8,
                ha="right", va="bottom")
    ax.set_xlim(0, frame.n_probes.max())
    ax.set_ylim(bottom=0)
    _style(ax, "Probes steered together", f"Mean absolute error ({unit.strip()})",
           title or f"Steering {variable} at layer 8, judged by a held-out probe")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="center right")
    return _save(fig, path)


def steering_overlay(summaries: dict[str, pd.DataFrame], path: Path,
                     title: str = "How much of the gap does steering close?") -> Path:
    """All three variables on one axis, each normalised by its own unsteered error.

    Degrees, m/s and m/s^2 cannot share a y-axis, so each curve is divided by its own
    baseline: 1.0 means steering achieved nothing, 0 means it reached the target
    exactly. That makes "how well does this method work" comparable across variables,
    which is what Part 2 needs in order to compare against spline steering.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.3), dpi=150)
    for variable, frame in summaries.items():
        color, marker, label = POLAR_STYLE.get(variable, (BLUE, "o", variable))
        frame = frame.sort_values("n_probes")
        baseline = float(frame.mae_to_target_mean.iloc[0])
        y = frame.mae_to_target_mean / baseline
        sd = frame.mae_to_target_std.fillna(0) / baseline
        ax.fill_between(frame.n_probes, y - sd, y + sd, color=color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(frame.n_probes, y, color=color, linewidth=2, label=label.capitalize(), zorder=3,
                marker=marker, markersize=4, markevery=max(1, len(frame) // 15))
    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0, 1.0, " no improvement", color=MUTED, fontsize=8, ha="left", va="bottom")
    ax.set_ylim(0, 1.15)
    ax.set_xlim(left=0)
    _style(ax, "Probes steered together", "Error to target, relative to unsteered", title)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    return _save(fig, path)


def _manifold_frame(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")


def manifold_figure(clips, clip_values, points, point_values, curve, curve_values, path,
                    variable: str, unit: str, periodic: bool, title: str | None = None) -> Path:
    """Part 2, figure 1: the fitted manifold, in the plane of the centroids.

    `clips`, `points` (centroids) and `curve` are already projected to 3 coordinates of
    a frame built from the CENTROIDS rather than the clips -- the clips' own principal
    directions are dominated by start position and the other physical variables, which
    the manifold does not describe, and in that frame the curve is edge-on and invisible.

    The clips are drawn because they are the honest part of the picture: they scatter
    1.4-2.6x further from the curve than the curve is long, so the manifold is a
    conditional mean threading a cloud, not a surface the data lies on.
    """
    cmap = "twilight" if periodic else "viridis"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=150)
    for ax, (i, j) in zip(axes, [(0, 1), (0, 2)]):
        ax.scatter(clips[:, i], clips[:, j], c=clip_values, cmap=cmap, s=4, alpha=0.18,
                   linewidths=0, zorder=2)
        ax.plot(curve[:, i], curve[:, j], color=INK, linewidth=2.2, zorder=4, alpha=0.85)
        sc = ax.scatter(points[:, i], points[:, j], c=point_values, cmap=cmap, s=34,
                        edgecolors=SURFACE, linewidths=0.8, zorder=5)
        _manifold_frame(ax, f"PC{i+1} vs PC{j+1}", f"centroid PC{i+1}", f"centroid PC{j+1}")
    bar = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
    bar.set_label(f"{variable} ({unit.strip()})" if unit.strip() else variable,
                  color=INK_2, fontsize=9)
    bar.ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)
    fig.suptitle(title or f"Activation manifold for {variable}, layer 8",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    return _save(fig, path)


def manifold_vs_chord_figure(curve, chords, labels, path, variable: str,
                             title: str | None = None) -> Path:
    """Part 2, figure 2: the curve against the straight paths between points on it.

    A straight line between two values is what linear steering takes. Where it leaves
    the curve, it passes through activations the encoder never produces -- which for
    direction means states with no direction at all.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=150)
    colours = [ORANGE, RED, AQUA]
    for ax, (i, j) in zip(axes, [(0, 1), (0, 2)]):
        ax.plot(curve[:, i], curve[:, j], color=INK, linewidth=2.2, zorder=3,
                label="the manifold")
        for n, (path_points, name) in enumerate(zip(chords, labels)):
            ax.plot(path_points[:, i], path_points[:, j], color=colours[n % len(colours)],
                    linewidth=1.8, linestyle=(0, (5, 2)), zorder=4, label=name)
            ax.scatter(path_points[[0, -1], i], path_points[[0, -1], j],
                       color=colours[n % len(colours)], s=26, zorder=5)
        _manifold_frame(ax, f"PC{i+1} vs PC{j+1}", f"centroid PC{i+1}", f"centroid PC{j+1}")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="best")
    fig.suptitle(title or f"{variable}: the curve against straight paths across it",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, path)


def manifold_residual_figure(values, curve_error, line_error, path, variable: str,
                             unit: str, scatter: float | None = None) -> Path:
    """Part 2, figure 3: how far held-out clips sit from the curve, across the range.

    Both curves are distances in standardised activation units. `scatter` marks the
    average distance of a clip from its own value's centroid -- the floor no curve
    parameterised by this variable alone can go below, because the remaining spread is
    start position and the other physical variables.
    """
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(6.8, 4.1), dpi=150)
    for err, colour, name in [(line_error, ORANGE, "best straight line"),
                              (curve_error, BLUE, "fitted manifold")]:
        frame = pd.DataFrame({"v": np.asarray(values)[order], "e": np.asarray(err)[order]})
        grouped = frame.groupby("v").e.agg(["mean", "std"]).reset_index()
        ax.fill_between(grouped.v, grouped["mean"] - grouped["std"].fillna(0),
                        grouped["mean"] + grouped["std"].fillna(0), color=colour,
                        alpha=0.15, linewidth=0, zorder=2)
        ax.plot(grouped.v, grouped["mean"], color=colour, linewidth=2, marker="o",
                markersize=4, label=name, zorder=3)
    if scatter is not None:
        ax.axhline(scatter, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.text(grouped.v.max(), scatter, "spread within one value ", color=MUTED,
                fontsize=8, ha="right", va="bottom")
    ax.set_ylim(bottom=0)
    _style(ax, f"{variable} ({unit.strip()})" if unit.strip() else variable,
           "Distance from a held-out clip", f"{variable}: where the manifold fits")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    return _save(fig, path)


def manifold_3d_figure(clips, clip_values, points, point_values, curve, path,
                       variable: str, unit: str, periodic: bool,
                       views=((74, -64, "Top view"), (8, -64, "Side view")),
                       title: str | None = None) -> Path:
    """Part 2, figure 1b: the manifold in 3D, in the Goodfire paper's house style.

    Two viewing angles of the same three centroid components, with the curve and its
    centroids dropped onto a floor plane as a grey shadow. The shadow is not
    decoration: in a static 3D scatter there is no way to judge depth, and the paper
    uses the same device. A shape that survives both views is real; one that appears
    in a single view is an artefact of where you happened to stand.

    A thinned sample of clips is drawn very faintly. Their figure omits the raw points
    -- in their Mountain Car setting the clips sit close to the curve -- but ours do
    not, scattering 1.4-2.6x further from the curve than the curve is long, so leaving
    them out entirely would imply a tidiness this data does not have.
    """
    cmap = "twilight" if periodic else "viridis"
    floor = min(curve[:, 2].min(), points[:, 2].min()) - 0.35 * np.ptp(curve[:, 2])

    fig = plt.figure(figsize=(11.5, 5.0), dpi=150)
    for n, (elev, azim, name) in enumerate(views):
        ax = fig.add_subplot(1, len(views), n + 1, projection="3d")
        ax.view_init(elev=elev, azim=azim)

        # shadow first, so everything else sits on top of it
        ax.plot(curve[:, 0], curve[:, 1], zs=floor, zdir="z", color=AXIS,
                linewidth=1.6, alpha=0.7, zorder=1)
        ax.scatter(points[:, 0], points[:, 1], zs=floor, zdir="z", color=AXIS,
                   marker="D", s=12, alpha=0.5, zorder=1)

        if clips is not None and len(clips):
            ax.scatter(clips[:, 0], clips[:, 1], clips[:, 2], c=clip_values, cmap=cmap,
                       s=3, alpha=0.10, linewidths=0, zorder=2)
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=INK, linewidth=2.0,
                alpha=0.9, zorder=3)
        sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=point_values,
                        cmap=cmap, marker="D", s=30, edgecolors=SURFACE,
                        linewidths=0.6, zorder=4)

        ax.set_zlim(floor, max(curve[:, 2].max(), points[:, 2].max()))
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((1.0, 1.0, 1.0, 0.0))
            pane._axinfo["grid"].update(color=GRID, linewidth=0.6)
        ax.set_xlabel("PC1", color=INK_2, fontsize=8, labelpad=-6)
        ax.set_ylabel("PC2", color=INK_2, fontsize=8, labelpad=-6)
        ax.set_zlabel("PC3", color=INK_2, fontsize=8, labelpad=-6)
        ax.tick_params(colors=MUTED, labelsize=6, pad=-3)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.set_title(name, color=INK, fontsize=10)

    bar = fig.colorbar(sc, ax=fig.axes, fraction=0.022, pad=0.04)
    bar.set_label(f"{variable} ({unit.strip()})" if unit.strip() else variable,
                  color=INK_2, fontsize=9)
    bar.ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)
    fig.suptitle(title or f"Activation manifold for {variable}, layer 8",
                 color=INK, fontsize=12, fontweight="bold", y=0.98)
    return _save(fig, path)
