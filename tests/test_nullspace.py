"""The projection must do exactly what C.11 claims: delete a subspace, nothing more.

These check the algebra rather than the experiment -- if any of them fails, the
dimensionality numbers in Part 1.2 mean nothing.
"""

import numpy as np
import pandas as pd
import pytest

from vjepa_physics.nullspace import (
    Sequence, dimensionality, leakage, orthonormal_basis, project, run_sequence,
)
from vjepa_physics.probes import sincos


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_basis_is_orthonormal_and_spans_the_probe_weights(rng):
    W = rng.normal(size=(2, 50))                       # a direction probe: sin and cos rows
    Q = orthonormal_basis(W)
    assert Q.shape == (50, 2)
    assert np.allclose(Q.T @ Q, np.eye(2), atol=1e-12)
    # Same subspace: each weight vector is reproduced exactly from the Q basis.
    assert np.allclose(Q @ (Q.T @ W.T), W.T, atol=1e-12)


def test_projection_makes_the_probe_output_constant(rng):
    X = rng.normal(size=(200, 50))
    W = rng.normal(size=(1, 50))
    Xp = project(X, orthonormal_basis(W))
    readout = Xp @ W[0]
    assert np.abs(readout).max() < 1e-10               # w.x = 0 for every row, so only the bias is left


def test_rank_drops_by_exactly_the_number_of_outputs(rng):
    X = rng.normal(size=(200, 50))
    assert np.linalg.matrix_rank(X) == 50
    X1 = project(X, orthonormal_basis(rng.normal(size=(1, 50))))
    assert np.linalg.matrix_rank(X1) == 49             # speed: one direction per round
    X2 = project(X, orthonormal_basis(rng.normal(size=(2, 50))))
    assert np.linalg.matrix_rank(X2) == 48             # direction: a plane per round


def test_projection_is_idempotent_and_symmetric(rng):
    Q = orthonormal_basis(rng.normal(size=(3, 40)))
    P = np.eye(40) - Q @ Q.T
    assert np.allclose(P, P.T)
    assert np.allclose(P @ P, P)                       # removing twice changes nothing more
    assert np.linalg.matrix_rank(P) == 37


def test_orthogonalising_against_earlier_rounds_removes_nothing_twice(rng):
    """A probe whose weights point partly along an already-deleted direction must
    contribute only the genuinely new part (N9)."""
    first = orthonormal_basis(rng.normal(size=(1, 30)))
    fresh = rng.normal(size=30)
    W = (3.0 * first[:, 0] + fresh).reshape(1, 30)     # mostly the old direction
    Q = orthonormal_basis(W, against=first)
    assert Q.shape[1] == 1
    assert abs(float(first[:, 0] @ Q[:, 0])) < 1e-12   # perpendicular to what is gone


def test_dependent_weight_rows_do_not_inflate_the_rank(rng):
    w = rng.normal(size=30)
    W = np.stack([w, 2.0 * w])                         # one direction written twice
    assert orthonormal_basis(W).shape[1] == 1          # not 2


def test_leakage_is_zero_for_a_fresh_direction_and_one_for_a_removed_one(rng):
    basis = orthonormal_basis(rng.normal(size=(2, 30)))
    assert leakage(basis[:, 0].reshape(1, 30), basis) == pytest.approx(1.0)
    orthogonal = project(rng.normal(size=(1, 30)), basis)
    assert leakage(orthogonal, basis) < 1e-12
    assert leakage(rng.normal(size=(1, 30)), np.zeros((30, 0))) == 0.0


def test_sequence_drives_a_readable_signal_to_chance_and_then_stops(rng):
    """End to end: the probe reads the target, the loop removes directions until it
    cannot, and the patience rule ends the run before the cap."""
    d, n = 40, 400
    plane = orthonormal_basis(rng.normal(size=(2, d)))
    coords = rng.normal(size=(n, 2))
    X = coords @ plane.T + 0.01 * rng.normal(size=(n, d))
    y = coords @ np.array([2.0, -1.0])                 # a linear function of that plane only
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    seq = run_sequence(X[:300], y[:300], X[300:], y[300:], lr=1e-2, wd=1e-4, epochs=40,
                       max_rounds=15, stop_r2=0.1, patience=2, verbose=False)
    assert seq.rounds.test_r2.iloc[0] > 0.9            # readable at the start
    assert seq.rounds.test_r2.iloc[-1] < 0.1           # at chance by the end
    assert len(seq.rounds) < 15                        # stopped on the rule, not the cap
    assert seq.dims_removed == len(seq.rounds)         # one dimension per round, scalar target
    assert (seq.rounds.rank_this_round == 1).all()


def test_the_removed_direction_stays_inside_the_data_span(rng):
    """N9's second job. Adam's weights include random-initialisation components that
    no gradient touches. Orthogonalising against everything already removed forces
    each new direction into the span the data still occupies, so the rank really does
    fall by one per round -- projecting out a direction the data does not occupy
    would change nothing at all."""
    d = 30
    X = rng.normal(size=(200, d))
    X = (X - X.mean(0)) / X.std(0)
    y = X @ rng.normal(size=d)
    seq = run_sequence(X[:150], y[:150], X[150:], y[150:], lr=1e-2, wd=1e-4, epochs=20,
                       max_rounds=5, stop_r2=-np.inf, verbose=False)
    ranks = [np.linalg.matrix_rank(project(X[:150], seq.basis[:, :k])) for k in range(6)]
    assert ranks == [d - k for k in range(6)]


def test_the_test_split_is_projected_with_the_training_basis_only(rng):
    """The basis must come from training clips; the held-out clips are only projected
    with it. If it were refitted on the test set, deleting a direction would always
    kill the test score exactly, and the evaluation would be meaningless."""
    d = 30
    X = rng.normal(size=(200, d))
    y = X @ rng.normal(size=d)
    X = (X - X.mean(0)) / X.std(0)
    train, test = X[:150], X[150:]
    seq = run_sequence(train, y[:150], test, y[150:], lr=1e-2, wd=1e-4, epochs=20,
                       max_rounds=3, stop_r2=-np.inf, verbose=False)
    # Every round's basis is orthonormal as a whole, across rounds.
    assert np.allclose(seq.basis.T @ seq.basis, np.eye(seq.basis.shape[1]), atol=1e-10)
    # The held-out rows keep a non-zero component outside the removed subspace.
    assert np.linalg.norm(project(test, seq.basis)) > 0


def test_direction_removes_two_dimensions_per_round(rng):
    d, n = 40, 400
    theta = rng.uniform(0, 360, size=n)
    plane = orthonormal_basis(rng.normal(size=(2, d)))
    X = sincos(theta) @ plane.T + 0.01 * rng.normal(size=(n, d))
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    seq = run_sequence(X[:300], sincos(theta[:300]), X[300:], sincos(theta[300:]),
                       lr=1e-2, wd=1e-4, epochs=40, max_rounds=4, stop_r2=-np.inf,
                       angles_test=theta[300:], verbose=False)
    assert (seq.rounds.rank_this_round == 2).all()
    assert seq.dims_removed == 2 * len(seq.rounds)
    assert {"test_circ_mae", "test_acc15"} <= set(seq.rounds.columns)
    assert seq.rounds.test_circ_mae.iloc[0] < 10.0     # readable before anything is removed


def test_dimensionality_needs_a_sustained_drop_not_a_single_dip():
    """Direction's curve is a sawtooth, so one round below the threshold is noise."""
    rounds = pd.DataFrame({"test_r2": [0.9, 0.8, 0.02, 0.7, 0.6, 0.01, 0.01, 0.01, 0.01],
                           "dims_removed": [0, 2, 4, 6, 8, 10, 12, 14, 16],
                           "rank_this_round": [2] * 9})
    got = dimensionality(rounds, threshold=0.1, patience=3)
    assert got["K"] == 5 and got["dims"] == 10 and got["reached"]     # not K=2, the dip

    never = dimensionality(pd.DataFrame({"test_r2": [0.9, 0.8], "dims_removed": [0, 1],
                                         "rank_this_round": [1, 1]}), threshold=0.1)
    assert not never["reached"] and never["K"] == 2 and never["dims"] == 2


def test_a_probe_scored_after_its_own_subspace_is_removed_collapses(rng):
    """The `stale_*` columns: the same probe, re-scored once the plane it reads from is
    gone. Its weighted sum is zero by construction, so it emits only its bias and the
    score collapses to a floor. Interleaving these with the retrained scores is what
    reproduces the paper's Fig. 4c sawtooth -- so the columns must mean exactly this."""
    d, n = 60, 240
    theta = np.repeat(np.linspace(0, 360, 16, endpoint=False), n // 16)
    plane = orthonormal_basis(rng.normal(size=(2, d)))
    X = sincos(theta) @ plane.T + 0.05 * rng.normal(size=(len(theta), d))
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    tr = np.arange(len(theta)) % 5 != 0
    seq = run_sequence(X[tr], sincos(theta[tr]), X[~tr], sincos(theta[~tr]), lr=1e-2, wd=1e-4,
                       epochs=40, max_rounds=4, stop_r2=-np.inf, dims_per_round=1,
                       angles_test=theta[~tr], verbose=False)
    assert {"stale_r2", "stale_acc15"} <= set(seq.rounds.columns)
    assert (seq.rounds.stale_r2 <= seq.rounds.test_r2 + 1e-9).all()      # never better
    assert seq.rounds.test_acc15.iloc[0] > 0.9                           # readable when retrained
    assert seq.rounds.stale_acc15.iloc[0] < 0.3                          # collapsed when not
