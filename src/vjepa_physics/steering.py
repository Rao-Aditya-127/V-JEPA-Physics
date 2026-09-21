"""Part 1.3: multi-probe subspace steering (paper Appendix C.12).

Parts 1.1 and 1.2 ask what is *readable*. This one asks what is *writable*: reach
into the activations, overwrite the variable, and check that an independent reader
sees the new value.

The subspace comes from Part 1.2's probe sequence. Stack those probes' weights and
orthonormalise (C.12 eq. 8):

    V, _ = QR([W_1.T, W_2.T, ..., W_K.T])                        (d, K * outputs)

Then for a clip x and a target value y*:

    1. split      c = V.T x,   x_perp = x - V c
    2. solve      every probe must predict y*
    3. rebuild    x* = V c* + x_perp

Step 2 is exact, not an optimisation. Every probe's weight rows lie inside span(V)
by construction, so W x_perp = 0 and the probes cannot see the remainder at all. The
condition W x* + b = y* therefore collapses to a square linear system in c*:

    (W V) c* = t - b            A = W V is (K*outputs, K*outputs)

K probes x `outputs` equations, K * outputs unknowns. One solve.

One correction to that picture, which the tests pin down. Probe k was trained on
activations with the first k subspaces already deleted, so for the stack to make sense
each probe must read untouched activations the way it read its own. A ridge probe
does; an Adam probe does NOT, because it keeps weight along the deleted directions
(its untouched random initialisation). `clean_weights` removes that part -- a no-op on
the data the probe was fitted and scored on, and necessary here.

What this file deliberately does NOT do is evaluate the steering. Solving for the
steering probes and then asking them what they read is circular -- they were made to
agree. The held-out protocol (C.12 p. 33) belongs in the driver script: an evaluation
probe trained on clips that built no part of the subspace.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def clean_weights(weights, basis: np.ndarray, ranks=None) -> list[np.ndarray]:
    """Drop from each probe the part that points into directions deleted before it.

    Necessary, and not a modification of the experiment. Probe k was trained on
    activations with the first k subspaces projected out, so along those directions
    its training data was exactly zero -- whatever weight Adam left there (its random
    initialisation, which no gradient ever touched: see Part 1.2's `leakage`)
    contributed nothing to any number Part 1.2 reported. Removing it is a no-op on
    that data, checked in tests.

    It matters here because steering evaluates every probe on the SAME activations.
    Left alone, an Adam probe reads untouched activations quite differently from the
    reduced ones it was fitted on -- by up to 1.6 on a target that lives in [-1, 1] --
    so "all probes predict y*" would be a condition on readouts that mean nothing.
    A ridge probe needs no cleaning: its solution lies in the data's row space, so the
    leakage is exactly zero to begin with.
    """
    basis = np.asarray(basis, dtype=np.float64)
    ranks = [w.shape[0] for w in weights] if ranks is None else list(ranks)
    cleaned, removed = [], 0
    for W, rank in zip(weights, ranks):
        W = np.asarray(W, dtype=np.float64)
        before = basis[:, :removed]
        cleaned.append(W - (W @ before) @ before.T if removed else W.copy())
        removed += int(rank)
    return cleaned


def stack_probes(weights, biases, n_probes: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The first `n_probes` of a Part 1.2 sequence, as one (K*outputs, d) matrix.

    `weights` is (K, outputs, d) and `biases` (K, outputs), as `bases.npz` stores
    them. Row order is probe 0's outputs, then probe 1's, and so on, so a stacked
    prediction reshapes to (n, K, outputs) without reordering.
    """
    weights, biases = np.asarray(weights, dtype=np.float64), np.asarray(biases, dtype=np.float64)
    k = len(weights) if n_probes is None else min(n_probes, len(weights))
    return weights[:k].reshape(-1, weights.shape[-1]), biases[:k].reshape(-1)


def steering_basis(W: np.ndarray) -> np.ndarray:
    """C.12 eq. 8: an orthonormal basis for everything the stacked probes can read.

    Columns are dropped when the stacked rows are linearly dependent -- later probes
    in a sequence can contribute directions that earlier ones already span, and QR
    fills such columns with arbitrary vectors that are not in the row space.
    """
    V, R = np.linalg.qr(np.asarray(W, dtype=np.float64).T)
    keep = np.abs(np.diag(R)) > 1e-8 * np.abs(np.diag(R)).max()
    return V[:, keep]


def target_coordinates(W: np.ndarray, b: np.ndarray, V: np.ndarray,
                       y_star: np.ndarray) -> np.ndarray:
    """Solve for the coordinates inside the subspace that make every probe read y*.

    Returns c* of length V.shape[1] -- one vector, shared by every clip, because the
    condition does not involve x at all. `lstsq` rather than `solve` so a rank-
    deficient stack degrades to the minimum-norm answer instead of raising.
    """
    W, b, V = (np.asarray(a, dtype=np.float64) for a in (W, b, V))
    target = np.tile(np.asarray(y_star, dtype=np.float64).ravel(), len(W) // len(np.ravel(y_star)))
    return np.linalg.lstsq(W @ V, target - b, rcond=None)[0]


def steer(X: np.ndarray, V: np.ndarray, c_star: np.ndarray) -> np.ndarray:
    """x* = V c* + x_perp, for every row of X.

    Every clip's readable coordinates are overwritten with the same c*; only the
    part no probe can see keeps the clips distinct.
    """
    X, V = np.asarray(X, dtype=np.float64), np.asarray(V, dtype=np.float64)
    x_perp = X - (X @ V) @ V.T
    return x_perp + np.broadcast_to(c_star, (len(X), len(c_star))) @ V.T


@dataclass
class Intervention:
    """One steering setup: the subspace, the solved coordinates, and its size."""
    V: np.ndarray                 # (d, m) orthonormal basis of the steering subspace
    c_star: np.ndarray            # (m,) coordinates every clip is moved to
    n_probes: int                 # how many probes of the sequence were used
    dims: int                     # m, the subspace dimension actually spanned

    def apply(self, X: np.ndarray) -> np.ndarray:
        return steer(X, self.V, self.c_star)


def build(weights, biases, y_star: np.ndarray, n_probes: int,
          basis: np.ndarray | None = None, ranks=None) -> Intervention:
    """The whole C.12 construction for one probe count.

    Pass `basis` (the accumulated orthonormal basis from the same Part 1.2 sequence)
    to clean the probe weights first -- required for Adam probes, harmless for ridge.
    """
    if basis is not None:
        weights = clean_weights(weights, basis, ranks)
    W, b = stack_probes(weights, biases, n_probes)
    V = steering_basis(W)
    return Intervention(V=V, c_star=target_coordinates(W, b, V, y_star),
                        n_probes=min(n_probes, len(weights)), dims=V.shape[1])


def displacement(X: np.ndarray, X_steered: np.ndarray) -> float:
    """How far steering moves a clip, as a fraction of its length.

    Not in the paper, and worth watching: the intervention can be large enough to
    put activations well outside the distribution the encoder ever produces, in
    which case a probe reading the target says less than it appears to.
    """
    moved = np.linalg.norm(np.asarray(X_steered) - np.asarray(X), axis=1)
    return float(moved.mean() / (np.linalg.norm(X, axis=1).mean() + 1e-12))
