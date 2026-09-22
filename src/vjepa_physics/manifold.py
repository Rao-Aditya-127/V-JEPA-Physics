"""Part 2: activation manifolds for the physical variables (Goodfire / Wurgaft et al.).

Parts 1.1-1.3 treat activation space as flat: a probe reads along a direction, and
steering writes coordinates into a subspace as though any value were allowed. This
module fits the shape the activations actually occupy, so a later stage can move
*along* it instead of across it.

The construction follows the paper's App. A.3 / B.1:

    1. PCA the activations to k dimensions
    2. group clips by their label value and average each group   -> centroids
    3. fit one smoothing spline per PCA coordinate through the centroids,
       parameterised by the label and weighted by sqrt(group size)

Two things are specific to this data and are not cosmetic.

**Direction gets a periodic spline.** Its centroids lie on a circle -- measured, not
assumed: the distance between two direction centroids follows the chord formula
2R sin(dtheta/2), and opposite angles are 19.6x further apart than neighbours where a
circle predicts 20.4. A natural spline would leave a seam between 354.375 deg and
0 deg; a periodic one closes with matching slope and curvature.

**The spline is a conditional mean, not a surface the clips lie on.** Within one label
value the clips scatter 1.4-2.6x further than the whole manifold is long, because the
datasets also vary start position (and, for direction, speed). The paper's Mountain
Car setting had position as nearly the only varying factor; ours does not. Every
number this module produces should be read with that in mind -- `project` returns the
residual precisely so it cannot be forgotten.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import splev, splrep


def centroids(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group clips by label value and average each group.

    Returns the sorted distinct values, one mean activation each, and the group sizes
    (which become spline weights: a value backed by more clips has a less noisy mean).
    """
    values = np.unique(y)
    means = np.stack([X[y == v].mean(axis=0) for v in values])
    counts = np.array([int((y == v).sum()) for v in values])
    return values, means, counts


@dataclass
class Manifold:
    """A curve through activation space, parameterised by the physical variable."""

    pca_mean: np.ndarray                  # (d,) mean of the fitting clips
    components: np.ndarray                # (k, d) PCA basis, rows orthonormal
    splines: list = field(default_factory=list)   # one tck per PCA coordinate
    values: np.ndarray = None             # the label values the splines were fitted on
    periodic: bool = False
    period: float | None = None
    smoothing: float = 0.0
    counts: np.ndarray = None

    @property
    def k(self) -> int:
        return len(self.components)

    def _wrap(self, v: np.ndarray) -> np.ndarray:
        """Bring a coordinate into the fitted range. Only meaningful when periodic:
        for direction, 370 deg is 10 deg, but 4.5 m/s is simply off the end."""
        v = np.atleast_1d(np.asarray(v, dtype=np.float64))
        if self.periodic:
            start = self.values[0]
            return start + (v - start) % self.period
        return v

    def latent(self, v) -> np.ndarray:
        """Where the curve sits at value v, in PCA coordinates. Shape (n, k)."""
        v = self._wrap(v)
        return np.stack([splev(v, tck) for tck in self.splines], axis=1)

    def evaluate(self, v) -> np.ndarray:
        """Where the curve sits at value v, back in the full activation space. (n, d)

        This is what an activation at that value looks like on average -- the point a
        later steering stage would move a clip towards.
        """
        return self.latent(v) @ self.components + self.pca_mean

    def project(self, X: np.ndarray, samples: int = 2000) -> tuple[np.ndarray, np.ndarray]:
        """Nearest point on the curve for each row of X.

        Returns its coordinate and the residual X - evaluate(coordinate). Found by
        dense search rather than a solver: the curve is one-dimensional and cheap to
        sample, and a solver would have to cope with the several local minima a
        folded curve produces -- the very folding that makes straight-line steering
        unsafe.

        The residual is returned, never discarded, because it is large here: clips
        scatter further from the curve than the curve is long.
        """
        grid = self._grid(samples)
        curve = self.evaluate(grid)                                   # (samples, d)
        # (n, samples) squared distances, expanded to avoid an (n, samples, d) array
        d2 = ((X ** 2).sum(1)[:, None] - 2 * X @ curve.T + (curve ** 2).sum(1)[None, :])
        nearest = grid[np.argmin(d2, axis=1)]
        return nearest, X - self.evaluate(nearest)

    def coordinate(self, X: np.ndarray, samples: int = 2000) -> np.ndarray:
        """[OURS] Read the variable off a clip with no probe at all, by asking which
        point of the curve it sits nearest. A geometric decoder, to set beside the
        linear probes of Part 1.1."""
        return self.project(X, samples)[0]

    def _grid(self, samples: int) -> np.ndarray:
        if self.periodic:
            return self.values[0] + np.linspace(0, self.period, samples, endpoint=False)
        return np.linspace(self.values[0], self.values[-1], samples)

    def arc_length(self, samples: int = 2000) -> float:
        """Distance travelled along the curve from end to end (or once round)."""
        curve = self.evaluate(self._grid(samples))
        if self.periodic:
            curve = np.vstack([curve, curve[:1]])
        return float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())


def fit_manifold(X: np.ndarray, y: np.ndarray, *, k: int, smoothing: float,
                 periodic: bool = False, period: float | None = None) -> Manifold:
    """Fit the activation manifold: PCA, centroids, then a spline per coordinate.

    `X` must already be standardised with the fitting split's statistics, exactly as
    in Parts 1.1-1.3, so the same probes can judge the result.

    `smoothing` is scipy's `s`: the spline is allowed a total weighted squared
    residual of that size, so larger means smoother. It is not a free choice --
    `choose_smoothing` picks it on held-out clips. Interpolating the centroids
    exactly (s = 0) would be fitting noise: neighbouring centroids here sit 0.6-1.5
    apart while each carries 5.4-6.0 of sampling noise.

    For a periodic fit the first centroid is repeated at `values[0] + period` so the
    spline closes; scipy ignores that repeated y value and takes the period from the
    x range.
    """
    X = np.asarray(X, dtype=np.float64)
    if periodic and period is None:
        raise ValueError("a periodic manifold needs a period (360 for direction)")

    pca_mean = X.mean(axis=0)
    centred = X - pca_mean
    # Right singular vectors are the PCA basis; economy SVD avoids a 1024x1024 matrix.
    components = np.linalg.svd(centred, full_matrices=False)[2][:k]

    values, means, counts = centroids(centred @ components.T, y)
    weights = np.sqrt(counts)

    knots, targets, w = values, means, weights
    if periodic:
        knots = np.append(values, values[0] + period)
        targets = np.vstack([means, means[:1]])
        w = np.append(weights, weights[0])

    splines = [splrep(knots, targets[:, j], w=w, s=smoothing, per=int(periodic), k=3)
               for j in range(k)]
    return Manifold(pca_mean=pca_mean, components=components, splines=splines,
                    values=values, periodic=periodic, period=period,
                    smoothing=smoothing, counts=counts)


def choose_smoothing(X_fit, y_fit, X_val, y_val, *, k: int, grid, periodic: bool = False,
                     period: float | None = None) -> tuple[float, list]:
    """Pick the smoothing parameter by held-out reconstruction error.

    For each candidate, fit on one split and measure how far the other split's clips
    sit from the curve at their own label value. Returns the best value and the whole
    trace, so the write-up can show the choice was not made by eye.
    """
    # A manifold that predicts a held-out clip worse than the training mean does is
    # broken, not merely bad. scipy's spline fit can fail to converge for a particular
    # smoothing value ("s too small") and return a curve with an error of 1e8 -- seen
    # here for acceleration at s = 1. Such a fit must never win the comparison.
    baseline = float(np.linalg.norm(X_val - np.asarray(X_fit).mean(axis=0), axis=1).mean())

    trace = []
    for s in grid:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            manifold = fit_manifold(X_fit, y_fit, k=k, smoothing=s,
                                    periodic=periodic, period=period)
        error = float(np.linalg.norm(X_val - manifold.evaluate(y_val), axis=1).mean())
        trace.append({"smoothing": float(s), "held_out_error": error,
                      "usable": bool(np.isfinite(error) and error < baseline)})

    usable = [row for row in trace if row["usable"]]
    if not usable:
        raise RuntimeError(f"every smoothing candidate fitted worse than the mean "
                           f"({baseline:.2f}); the manifold is not describing this data")
    best = min(usable, key=lambda row: row["held_out_error"])
    return best["smoothing"], trace


def chord(manifold: Manifold, v0: float, v1: float, t) -> np.ndarray:
    """The straight line between two points of the curve -- what linear steering does.

    Comparing true intermediate activations against this, rather than against the
    curve, is the test of whether the curvature matters: if the chord passes as close
    to them as the curve does, following the curve buys nothing.
    """
    h0, h1 = manifold.evaluate(v0)[0], manifold.evaluate(v1)[0]
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))[:, None]
    return (1 - t) * h0 + t * h1


def save_manifold(manifold: Manifold, path) -> None:
    """Store a fitted manifold so the steering stage can reuse it unchanged.

    The splines are kept as their raw (knots, coefficients, degree) triples. Each PCA
    coordinate gets its own knots -- scipy places them adaptively from the smoothing
    budget, so they are not shared -- hence one indexed entry per coordinate rather
    than one array.
    """
    payload = {"pca_mean": manifold.pca_mean, "components": manifold.components,
               "values": manifold.values, "counts": manifold.counts,
               "periodic": np.array(manifold.periodic),
               "period": np.array(np.nan if manifold.period is None else manifold.period),
               "smoothing": np.array(manifold.smoothing),
               "n_splines": np.array(len(manifold.splines))}
    for j, (knots, coefficients, degree) in enumerate(manifold.splines):
        payload[f"spline{j}_t"] = knots
        payload[f"spline{j}_c"] = coefficients
        payload[f"spline{j}_k"] = np.array(degree)
    np.savez_compressed(path, **payload)


def load_manifold(path) -> Manifold:
    z = np.load(path)
    splines = [(z[f"spline{j}_t"], z[f"spline{j}_c"], int(z[f"spline{j}_k"]))
               for j in range(int(z["n_splines"]))]
    period = float(z["period"])
    return Manifold(pca_mean=z["pca_mean"], components=z["components"], splines=splines,
                    values=z["values"], periodic=bool(z["periodic"]),
                    period=None if np.isnan(period) else period,
                    smoothing=float(z["smoothing"]), counts=z["counts"])


def components_for_variance(X: np.ndarray, y: np.ndarray, fraction: float,
                            max_k: int = 60) -> int:
    """How many PCA directions are needed to hold `fraction` of the CENTROID variance.

    Chosen on the centroids rather than the clips because the manifold is what we are
    fitting: most of the clip-to-clip variance here is start position and the other
    physical variables, which the curve does not describe and should not chase.
    """
    X = np.asarray(X, dtype=np.float64)
    centred = X - X.mean(axis=0)
    basis = np.linalg.svd(centred, full_matrices=False)[2][:max_k]
    _, means, _ = centroids(centred, y)
    projected = means @ basis.T
    captured = np.cumsum((projected ** 2).sum(axis=0)) / (means ** 2).sum()
    return int(np.argmax(captured >= fraction) + 1)
