import numpy as np
import torch

from vjepa_physics.probes import circular_mae, fit_probe, mae, r2_score, sincos, standardize, sweep


def planted(n=400, d=64, noise=0.1, seed=0):
    """Features with a known linear signal: y = X w + 2 + noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = X @ rng.normal(size=d) * 0.3 + 2.0 + noise * rng.normal(size=n)
    return X, y


def split(X, y):
    Xf, Xv, Xt = standardize(X[:250], X[250:320], X[320:])
    return Xf, y[:250], Xv, y[250:320], Xt, y[320:]


def test_r2_reference_points():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(r2_score(y, y), 1.0)
    assert np.isclose(r2_score(y, np.full(4, y.mean())), 0.0)
    assert r2_score(y, y[::-1]) < 0                      # worse than the mean
    assert np.isclose(mae(y, y + 0.5), 0.5)


def test_bank_follows_the_reference_probe_exactly():
    """The fast lockstep sweep must match training each config on its own."""
    Xf, yf, Xv, yv, Xt, yt = split(*planted())
    result = sweep(Xf, yf, Xv, yv, Xt, yt, learning_rates=[1e-3, 3e-3],
                   weight_decays=[0.01, 0.4], epochs=5, seed=0)
    for p, (lr, wd) in enumerate(result.configs):
        probe = fit_probe(Xf, yf, lr, wd, epochs=5, seed=0)
        with torch.no_grad():
            reference = probe(torch.tensor(Xt)).numpy()
        assert np.allclose(result.test_pred[p], reference, atol=1e-4), (lr, wd)


def test_recovers_a_planted_signal():
    Xf, yf, Xv, yv, Xt, yt = split(*planted())
    result = sweep(Xf, yf, Xv, yv, Xt, yt, learning_rates=[1e-3, 3e-3, 5e-3],
                   weight_decays=[0.01], epochs=50)
    assert result.test_r2[result.best] > 0.95


def test_shuffled_labels_carry_no_signal():
    X, y = planted()
    y = np.random.default_rng(1).permutation(y)
    Xf, yf, Xv, yv, Xt, yt = split(X, y)
    result = sweep(Xf, yf, Xv, yv, Xt, yt, learning_rates=[1e-3, 3e-3],
                   weight_decays=[0.01, 0.8], epochs=50)
    assert result.test_r2[result.best] < 0.1


def test_shared_scale_does_not_explode_on_unseen_pixels():
    """A pixel that is flat background in every training clip but holds the disk in a
    test clip: per-pixel z-scoring divides by ~1e-6; the shared scale must not."""
    rng = np.random.default_rng(0)
    fit = rng.uniform(0.3, 0.6, size=(50, 8))
    fit[:, 0] = 0.12                                  # background in every training clip
    test = fit[:5].copy()
    test[:, 0] = 0.6                                  # the disk arrives in this cell
    _, per_pixel = standardize(fit, test)
    _, shared = standardize(fit, test, per_feature=False)
    assert np.abs(per_pixel[:, 0]).max() > 1e4        # the failure mode
    assert np.abs(shared).max() < 10                  # the fix


def test_selection_uses_validation_not_test():
    Xf, yf, Xv, yv, Xt, yt = split(*planted())
    result = sweep(Xf, yf, Xv, yv, Xt, yt, learning_rates=[1e-4, 3e-3],
                   weight_decays=[0.01], epochs=5)
    assert result.best == int(np.argmax(result.val_r2))


def test_sincos_roundtrip():
    theta = np.array([0.0, 90.0, 180.0, 270.0, 359.0])
    s = sincos(theta)
    assert s.shape == (5, 2)
    assert np.allclose(s[1], [1.0, 0.0]) and np.allclose(s[2], [0.0, -1.0])   # (sin, cos) order
    assert circular_mae(theta, s) < 1e-9                                       # exact prediction


def test_circular_mae_measures_the_short_way_round():
    assert np.isclose(circular_mae(np.array([359.0]), sincos(np.array([1.0]))), 2.0)
    assert np.isclose(circular_mae(np.array([10.0]), sincos(np.array([190.0]))), 180.0)
    assert np.isclose(circular_mae(np.array([0.0]), 5 * sincos(np.array([30.0]))), 30.0)  # length ignored


def test_circular_mae_chance_is_90_degrees():
    rng = np.random.default_rng(0)
    truth, guess = rng.uniform(0, 360, 20000), rng.uniform(0, 360, 20000)
    assert abs(circular_mae(truth, sincos(guess)) - 90.0) < 1.5
