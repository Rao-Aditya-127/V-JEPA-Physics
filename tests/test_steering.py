"""Steering must overwrite exactly the part the probes read, and nothing else.

If any of these fail, Part 1.3's held-out numbers are measuring something other than
the intervention they claim to measure.
"""

import numpy as np
import pytest

from vjepa_physics.nullspace import orthonormal_basis, project, run_sequence
from vjepa_physics.probes import sincos
from vjepa_physics.steering import (
    build, clean_weights, displacement, stack_probes, steer, steering_basis,
    target_coordinates,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _sequence(rng, d=40, n=400, outputs=2):
    """A small Part 1.2 sequence to steer with."""
    theta = np.repeat(np.linspace(0, 360, 20, endpoint=False), n // 20)
    plane = orthonormal_basis(rng.normal(size=(2, d)))
    X = sincos(theta) @ plane.T + 0.05 * rng.normal(size=(len(theta), d))
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    tr = np.arange(len(theta)) % 5 != 0
    seq = run_sequence(X[tr], sincos(theta[tr]), X[~tr], sincos(theta[~tr]), lr=1e-2, wd=1e-4,
                       epochs=40, max_rounds=6, stop_r2=-np.inf, verbose=False)
    return seq, X[~tr], theta[~tr]


def test_stacking_keeps_probe_order(rng):
    weights = rng.normal(size=(4, 2, 30))
    biases = rng.normal(size=(4, 2))
    W, b = stack_probes(weights, biases, n_probes=3)
    assert W.shape == (6, 30) and b.shape == (6,)
    assert np.allclose(W[0], weights[0, 0]) and np.allclose(W[1], weights[0, 1])
    assert np.allclose(W[4], weights[2, 0])          # probe 2's first output
    assert np.allclose(b[4], biases[2, 0])


def test_asking_for_more_probes_than_exist_is_not_an_error(rng):
    W, b = stack_probes(rng.normal(size=(3, 2, 30)), rng.normal(size=(3, 2)), n_probes=99)
    assert W.shape == (6, 30)


def test_basis_is_orthonormal_and_spans_the_stacked_probes(rng):
    W = rng.normal(size=(6, 40))
    V = steering_basis(W)
    assert V.shape == (40, 6)
    assert np.allclose(V.T @ V, np.eye(6), atol=1e-12)
    assert np.allclose(V @ (V.T @ W.T), W.T, atol=1e-12)     # nothing of W lies outside V


def test_dependent_probe_rows_do_not_inflate_the_subspace(rng):
    w = rng.normal(size=40)
    W = np.stack([w, 2 * w, rng.normal(size=40)])
    assert steering_basis(W).shape[1] == 2                   # not 3


def test_probes_cannot_see_the_remainder(rng):
    """The reason step 2 is a square solve rather than an optimisation."""
    W = rng.normal(size=(6, 40))
    V = steering_basis(W)
    X = rng.normal(size=(50, 40))
    x_perp = X - (X @ V) @ V.T
    assert np.abs(x_perp @ W.T).max() < 1e-10


def test_every_probe_reads_the_target_after_steering(rng):
    """The defining property: W x* + b = y* for all probes, exactly."""
    W, b = rng.normal(size=(6, 40)), rng.normal(size=6)
    V = steering_basis(W)
    y_star = np.array([0.5, -0.3])                           # a 2-output target, 3 probes
    c_star = target_coordinates(W, b, V, y_star)
    X_star = steer(rng.normal(size=(50, 40)), V, c_star)
    reads = X_star @ W.T + b
    assert np.abs(reads - np.tile(y_star, 3)).max() < 1e-9


def test_steering_leaves_the_invisible_part_alone(rng):
    W = rng.normal(size=(4, 40))
    V = steering_basis(W)
    X = rng.normal(size=(50, 40))
    X_star = steer(X, V, target_coordinates(W, rng.normal(size=4), V, np.array([1.0])))
    before, after = X - (X @ V) @ V.T, X_star - (X_star @ V) @ V.T
    assert np.allclose(before, after, atol=1e-12)            # x_perp untouched
    assert not np.allclose(X, X_star)                        # but something did change


def test_all_clips_land_on_the_same_coordinates(rng):
    """c* does not depend on x, so steered clips differ only in what no probe reads."""
    W = rng.normal(size=(4, 40))
    V = steering_basis(W)
    X_star = steer(rng.normal(size=(20, 40)), V,
                   target_coordinates(W, np.zeros(4), V, np.array([1.0])))
    coords = X_star @ V
    assert np.abs(coords - coords[0]).max() < 1e-10


def test_raw_adam_probes_do_not_transfer_to_untouched_activations(rng):
    """Why `clean_weights` exists. An Adam probe keeps weight along directions that
    were already deleted when it was trained -- its random initialisation, which no
    gradient touched because the data was flat there. On its own reduced activations
    that weight contributes nothing; on untouched activations it corrupts the readout."""
    seq, X_test, _ = _sequence(rng)
    ranks = seq.rounds.rank_this_round.tolist()
    worst = 0.0
    for k, W in enumerate(seq.weights):
        reduced = project(X_test, seq.basis[:, :sum(ranks[:k])])
        worst = max(worst, np.abs(reduced @ W.T - X_test @ W.T).max())
    assert worst > 0.1                                        # the problem is real


def test_cleaning_makes_probes_transfer_and_changes_nothing_they_measured(rng):
    """After cleaning, a probe reads untouched activations exactly as it reads its own
    reduced ones -- and on those reduced activations the cleaning changed nothing, so
    no number Part 1.2 reported moves."""
    seq, X_test, _ = _sequence(rng)
    ranks = seq.rounds.rank_this_round.tolist()
    cleaned = clean_weights(seq.weights, seq.basis, ranks)
    for k, (raw, fixed) in enumerate(zip(seq.weights, cleaned)):
        reduced = project(X_test, seq.basis[:, :sum(ranks[:k])])
        assert np.abs(reduced @ fixed.T - X_test @ fixed.T).max() < 1e-9      # transfers
        assert np.abs(reduced @ fixed.T - reduced @ raw.T).max() < 1e-9       # no-op on its own data


def test_build_end_to_end_hits_the_target_on_real_probes(rng):
    seq, X_test, _ = _sequence(rng)
    y_star = sincos(np.array([90.0]))[0]
    plan = build(seq.weights, seq.biases, y_star, n_probes=4,
                 basis=seq.basis, ranks=seq.rounds.rank_this_round.tolist())
    assert plan.n_probes == 4 and plan.dims == 8
    # checked against the CLEANED probes -- those are what the plan solved for, and
    # the only form in which a probe's readout on untouched activations is meaningful
    W, b = stack_probes(clean_weights(seq.weights, seq.basis,
                                      seq.rounds.rank_this_round.tolist()), seq.biases, 4)
    reads = plan.apply(X_test) @ W.T + b
    assert np.abs(reads - np.tile(y_star, 4)).max() < 1e-8


def test_more_probes_means_a_bigger_edit(rng):
    """Steering more directions moves the activations further -- the cost the paper
    does not report, and the reason `displacement` is recorded."""
    seq, X_test, _ = _sequence(rng)
    y_star = sincos(np.array([90.0]))[0]
    ranks = seq.rounds.rank_this_round.tolist()
    moves = [displacement(X_test, build(seq.weights, seq.biases, y_star, n,
                                        basis=seq.basis, ranks=ranks).apply(X_test))
             for n in (1, 3, 6)]
    assert moves[0] < moves[1] < moves[2]
    assert moves[0] > 0
