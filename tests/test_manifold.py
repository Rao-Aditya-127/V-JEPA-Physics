"""The manifold must be the curve the data actually lies on, and must close for direction.

If these fail, Part 2's comparison against Part 1.3 is comparing against a curve that
does not describe the activations.
"""

import numpy as np
import pytest

from vjepa_physics.manifold import (
    Manifold, centroids, choose_smoothing, chord, fit_manifold,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _straight(rng, d=20, n_values=32, per_value=12, noise=0.0):
    """Activations lying exactly on a straight line in the label."""
    values = np.repeat(np.linspace(1.0, 5.0, n_values), per_value)
    direction, offset = rng.normal(size=d), rng.normal(size=d)
    X = offset + values[:, None] * direction + noise * rng.normal(size=(len(values), d))
    return X, values


def _circle(rng, d=20, n_values=32, per_value=12, noise=0.0):
    """Activations on a circle, as direction's centroids are.

    Also returns the exact point at any angle, so tests can ask where the truth is
    at angles that were never fitted -- which is where a non-periodic fit fails.
    """
    theta = np.repeat(np.linspace(0, 360, n_values, endpoint=False), per_value)
    a, b = rng.normal(size=d), rng.normal(size=d)
    truth = lambda deg: (np.cos(np.radians(np.atleast_1d(deg)))[:, None] * a
                         + np.sin(np.radians(np.atleast_1d(deg)))[:, None] * b)
    X = truth(theta) + noise * rng.normal(size=(len(theta), d))
    return X, theta, truth


def test_centroids_group_and_count(rng):
    X, y = _straight(rng)
    values, means, counts = centroids(X, y)
    assert len(values) == 32 and means.shape == (32, 20)
    assert (counts == 12).all()
    assert np.allclose(means[0], X[y == values[0]].mean(0))


def test_a_straight_line_is_recovered_exactly(rng):
    """The floor: if the data is a line, the fitted curve must be that line."""
    X, y = _straight(rng)
    m = fit_manifold(X, y, k=3, smoothing=0.0)
    assert np.abs(m.evaluate(y) - X).max() < 1e-8
    mid = m.evaluate([3.0])[0]                       # a value never used as a knot centre
    exact = X[np.isclose(y, 1.0)][0] + (3.0 - 1.0) / (5.0 - 1.0) * (
        X[np.isclose(y, 5.0)][0] - X[np.isclose(y, 1.0)][0])
    assert np.abs(mid - exact).max() < 1e-8


def test_a_periodic_manifold_closes(rng):
    """354.375 deg -> 0 deg must be an ordinary step, not a seam."""
    X, y, _ = _circle(rng)
    m = fit_manifold(X, y, k=4, smoothing=0.0, periodic=True, period=360.0)
    assert np.abs(m.evaluate([0.0]) - m.evaluate([360.0])).max() < 1e-8
    step = np.linalg.norm(np.diff(m.evaluate(np.linspace(0, 350, 36)), axis=0), axis=1)
    seam = np.linalg.norm(m.evaluate([0.0]) - m.evaluate([350.0]))
    assert seam < 2.5 * step.mean()                  # the wrap is a normal-sized step


def test_a_non_periodic_fit_breaks_at_the_wrap(rng):
    """Why periodic matters. The last fitted angle (348.75) and the first (0) are
    neighbours on the circle, so there is no positional gap between them -- the break
    is in what happens BETWEEN them, where a non-periodic spline has to extrapolate."""
    X, y, truth = _circle(rng)
    per = fit_manifold(X, y, k=4, smoothing=0.0, periodic=True, period=360.0)
    nat = fit_manifold(X, y, k=4, smoothing=0.0, periodic=False)
    gap = 355.0                                       # between the last knot and the wrap
    target = truth(gap)[0]
    radius = np.linalg.norm(truth(np.arange(0, 360, 10)), axis=1).mean()
    err_per = np.linalg.norm(per.evaluate([gap])[0] - target)
    err_nat = np.linalg.norm(nat.evaluate([gap])[0] - target)
    assert err_per < 0.15 * radius                    # periodic interpolates across the wrap
    assert err_nat > 3 * err_per                      # natural extrapolates and misses


def test_wrapping_only_happens_when_periodic(rng):
    X, y, _ = _circle(rng)
    m = fit_manifold(X, y, k=4, smoothing=0.0, periodic=True, period=360.0)
    assert np.allclose(m.evaluate([370.0]), m.evaluate([10.0]), atol=1e-8)
    Xs, ys = _straight(rng)
    line = fit_manifold(Xs, ys, k=3, smoothing=0.0)
    assert not np.allclose(line.evaluate([6.0]), line.evaluate([2.0]))   # no wrap


def test_projection_finds_the_right_coordinate(rng):
    X, y, _ = _circle(rng, noise=0.01)
    m = fit_manifold(X, y, k=4, smoothing=0.0, periodic=True, period=360.0)
    coord, residual = m.project(X)
    err = np.abs((coord - y + 180) % 360 - 180)
    assert err.mean() < 5.0                                   # recovers the angle
    assert np.allclose(X - m.evaluate(coord), residual, atol=1e-10)


def test_projection_is_idempotent(rng):
    """A point already on the curve must project to itself."""
    X, y = _straight(rng)
    m = fit_manifold(X, y, k=3, smoothing=0.0)
    on_curve = m.evaluate(np.array([2.0, 3.0, 4.0]))
    coord, residual = m.project(on_curve)
    assert np.abs(coord - np.array([2.0, 3.0, 4.0])).max() < 0.02
    assert np.abs(residual).max() < 0.05


def test_smoothing_is_chosen_not_guessed(rng):
    """With noisy centroids the best smoothing must be non-zero -- interpolating
    every centroid would be fitting the noise."""
    X, y = _straight(rng, noise=2.0)
    half = len(X) // 2
    order = rng.permutation(len(X))
    fit, val = order[:half], order[half:]
    best, trace = choose_smoothing(X[fit], y[fit], X[val], y[val], k=3,
                                   grid=[0.0, 1.0, 10.0, 100.0, 1000.0])
    assert len(trace) == 5
    assert best > 0.0
    assert min(r["held_out_error"] for r in trace) < trace[0]["held_out_error"]


def test_chord_is_the_straight_path(rng):
    X, y, _ = _circle(rng)
    m = fit_manifold(X, y, k=4, smoothing=0.0, periodic=True, period=360.0)
    line = chord(m, 0.0, 180.0, [0.0, 0.5, 1.0])
    assert np.allclose(line[0], m.evaluate([0.0])[0], atol=1e-8)
    assert np.allclose(line[2], m.evaluate([180.0])[0], atol=1e-8)
    # the midpoint of a chord across a circle sits near the centre, far off the curve
    to_curve = np.linalg.norm(line[1] - m.evaluate(np.linspace(0, 360, 361)), axis=1).min()
    radius = np.linalg.norm(m.evaluate(np.linspace(0, 360, 361)) - m.evaluate(
        np.linspace(0, 360, 361)).mean(0), axis=1).mean()
    assert to_curve > 0.5 * radius


def test_going_round_is_much_further_than_cutting_across(rng):
    """The quantity that makes manifold steering differ from linear steering at all."""
    X, y, _ = _circle(rng)
    m = fit_manifold(X, y, k=4, smoothing=0.0, periodic=True, period=360.0)
    across = np.linalg.norm(m.evaluate([0.0]) - m.evaluate([180.0]))
    assert m.arc_length() > 2.5 * across


def test_a_periodic_fit_needs_a_period(rng):
    X, y, _ = _circle(rng)
    with pytest.raises(ValueError, match="period"):
        fit_manifold(X, y, k=3, smoothing=0.0, periodic=True)


def test_a_diverged_spline_can_never_be_chosen(rng):
    """scipy's fit does not always converge for a given smoothing, and when it fails it
    returns a curve with an enormous error rather than raising. Seen for real: at
    s = 1 the acceleration manifold fitted with a held-out error of 4e8. Anything
    worse than predicting the training mean is rejected outright."""
    X, y = _straight(rng, noise=1.0)
    half = len(X) // 2
    order = rng.permutation(len(X))
    fit, val = order[:half], order[half:]
    best, trace = choose_smoothing(X[fit], y[fit], X[val], y[val], k=3,
                                   grid=[0.0, 10.0, 1000.0])
    assert all("usable" in row for row in trace)
    baseline = np.linalg.norm(X[val] - X[fit].mean(0), axis=1).mean()
    chosen = next(r for r in trace if r["smoothing"] == best)
    assert chosen["held_out_error"] < baseline          # never worse than no manifold at all
