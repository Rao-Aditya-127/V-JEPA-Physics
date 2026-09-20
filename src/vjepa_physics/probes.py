"""Linear probes and the Appendix B hyperparameter sweep.

A probe is f(h) = W h + b -- for speed, 1024 weights and one bias -- trained with
Adam on MSE. Appendix B sweeps 5 learning rates x 4 weight decays = 20 configs and
keeps the one with the best validation score.

Two implementations, deliberately:

  fit_probe()   one probe, plain nn.Linear + torch.optim.Adam. The readable reference.
  sweep()       all 20 configs trained in lockstep: weights stacked as (20, d_in, d_out)
                and Adam's update vectorised over the config axis. Same init, same
                minibatch order, same per-config lr/wd, so each probe follows exactly
                the trajectory fit_probe() gives it -- tests/test_probes.py checks this.
                It exists purely for speed.

Choices the paper leaves open (see the audit in think/01-SPEED-TASK.md):
  U2  features standardised with statistics of the fit split only
  U6  minibatch size 32 -- "50 epochs" means ~1,900 updates, not 50
  U7  the bias is excluded from weight decay and initialised at the fit-split mean
      of y; mean speed is ~2.1 m/s and wd up to 0.8 would otherwise drag it to 0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def standardize(fit: np.ndarray, *others: np.ndarray, per_feature: bool = True) -> list[np.ndarray]:
    """Centre and scale every array using fit-split statistics only (U2).

    per_feature=True   z-score each dimension -- for encoder features, whose scale
                       varies a lot across dimensions and depth.
    per_feature=False  centre each dimension, then divide by one shared scale -- for
                       raw pixels, which already share units. Per-pixel z-scoring
                       divides background cells by their codec-noise std (~0.002), so
                       a test clip with the disk in a cell no training clip covered
                       blows up (R^2 ~ -1e5 on a small subset).
    """
    mean = fit.mean(axis=0)
    std = fit.std(axis=0) + 1e-6 if per_feature else (fit - mean).std() + 1e-6
    return [(x - mean) / std for x in (fit, *others)]


def _align(y: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """y -> (n, k); pred -> (n, k) or a stack (P, n, k)."""
    y = np.asarray(y).reshape(len(y), -1)
    pred = np.asarray(pred)
    return y, (pred.reshape(-1, 1) if pred.ndim == 1 else pred)


def r2_score(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """1 - SS_res / SS_tot, averaged over outputs; a stack (P, n, k) gives (P,).
    Negative means worse than always predicting the mean."""
    y, pred = _align(y, pred)
    ss_res = ((pred - y) ** 2).sum(axis=-2)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    return (1.0 - ss_res / ss_tot).mean(axis=-1)


def mae(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    y, pred = _align(y, pred)
    return np.abs(pred - y).mean(axis=(-2, -1))


def sincos(theta_degrees: np.ndarray) -> np.ndarray:
    """Direction as a probe target, (sin θ, cos θ) -> shape (n, 2).

    The paper's circular regression (C.11). Regressing the angle itself would be
    wrong: 359° and 1° are 2° apart but 358 apart as numbers.
    """
    radians = np.radians(np.asarray(theta_degrees, dtype=np.float64))
    return np.stack([np.sin(radians), np.cos(radians)], axis=1)


def circular_errors(theta_degrees: np.ndarray, pred_sincos: np.ndarray) -> np.ndarray:
    """Signed angular error in degrees, wrapped to [-180, 180).

    The predicted (sin, cos) pair is turned back into an angle with atan2, so only
    its direction matters, not its length.
    """
    pred = np.degrees(np.arctan2(pred_sincos[:, 0], pred_sincos[:, 1]))
    return (pred - np.asarray(theta_degrees) + 180.0) % 360.0 - 180.0


def circular_mae(theta_degrees: np.ndarray, pred_sincos: np.ndarray) -> float:
    """Mean angular error in degrees, the short way round. Perfect is 0°; chance is 90°."""
    return float(np.abs(circular_errors(theta_degrees, pred_sincos)).mean())


def accuracy_within(theta_degrees: np.ndarray, pred_sincos: np.ndarray, tol: float = 15.0) -> float:
    """Fraction of clips predicted within `tol` degrees -- the y-axis of the paper's
    Figure 4c ("accuracy within 15°"), which the text never defines. Chance at 15° is
    2*15/360 = 0.083."""
    return float((np.abs(circular_errors(theta_degrees, pred_sincos)) <= tol).mean())


def _as_2d(y: np.ndarray) -> np.ndarray:
    return y.reshape(len(y), -1).astype(np.float32)


def _initial_linear(d_in: int, d_out: int, seed: int) -> nn.Linear:
    """PyTorch's default nn.Linear init, seeded, so every config starts identically."""
    torch.manual_seed(seed)
    return nn.Linear(d_in, d_out)


def _batches(n: int, batch_size: int, epochs: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            yield order[start:start + batch_size]


# ---------------------------------------------------------------------------------
# Reference implementation: one probe.
# ---------------------------------------------------------------------------------

def fit_probe(X_fit: np.ndarray, y_fit: np.ndarray, lr: float, wd: float, *, epochs: int = 50,
              batch_size: int = 32, seed: int = 0, device: str = "cpu") -> nn.Linear:
    """Train one linear probe on already-standardised features."""
    X = torch.tensor(X_fit, dtype=torch.float32, device=device)
    y = torch.tensor(_as_2d(y_fit), device=device)
    probe = _initial_linear(X.shape[1], y.shape[1], seed).to(device)
    with torch.no_grad():
        probe.bias.copy_(y.mean(dim=0))                                        # U7
    optimizer = torch.optim.Adam([
        {"params": [probe.weight], "lr": lr, "weight_decay": wd},
        {"params": [probe.bias], "lr": lr, "weight_decay": 0.0},               # U7
    ])
    for idx in _batches(len(X), batch_size, epochs, seed):
        idx = idx.to(device)
        optimizer.zero_grad()
        loss = ((probe(X[idx]) - y[idx]) ** 2).mean()                          # MSE
        loss.backward()
        optimizer.step()
    return probe


# ---------------------------------------------------------------------------------
# Fast implementation: all configs at once.
# ---------------------------------------------------------------------------------

def _adam_step(param, grad, m, v, t, lr, wd, beta1=0.9, beta2=0.999, eps=1e-8):
    """torch.optim.Adam's update, vectorised over a leading probe axis.

    Same maths as the library (coupled L2 weight decay, bias-corrected moments), but
    `lr` and `wd` are tensors, so all P configs update in one operation instead of a
    Python loop over 2P parameter groups on every step.
    """
    grad = grad + wd * param
    m.mul_(beta1).add_(grad, alpha=1 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    denom = (v.sqrt() / (1 - beta2 ** t) ** 0.5).add_(eps)
    param.sub_(lr / (1 - beta1 ** t) * m / denom)


@dataclass
class SweepResult:
    configs: list[tuple[float, float]]    # (lr, wd) per probe
    val_r2: np.ndarray                    # (P,)
    test_r2: np.ndarray                   # (P,)
    test_mae: np.ndarray                  # (P,)
    test_pred: np.ndarray                 # (P, n_test, d_out)

    @property
    def best(self) -> int:
        """Selection on validation only -- never on test (U3)."""
        return int(np.argmax(self.val_r2))


def sweep(X_fit, y_fit, X_val, y_val, X_test, y_test, *, learning_rates, weight_decays,
          epochs: int = 50, batch_size: int = 32, seed: int = 0, device: str = "cpu") -> SweepResult:
    """Train every (lr, wd) config on the fit split; score each on val and test.
    Features must already be standardised."""
    configs = [(lr, wd) for lr in learning_rates for wd in weight_decays]
    P = len(configs)
    X = torch.tensor(X_fit, dtype=torch.float32, device=device)
    y = torch.tensor(_as_2d(y_fit), device=device)
    init = _initial_linear(X.shape[1], y.shape[1], seed)

    # Every probe starts from the same init: W (P, d_in, d_out), b (P, d_out).
    W = init.weight.detach().T.to(device).repeat(P, 1, 1).requires_grad_(True)
    b = y.mean(dim=0).repeat(P, 1).requires_grad_(True)                        # U7
    lr = torch.tensor([c[0] for c in configs], device=device).view(P, 1, 1)
    wd = torch.tensor([c[1] for c in configs], device=device).view(P, 1, 1)
    mW, vW, mb, vb = (torch.zeros_like(W), torch.zeros_like(W),
                      torch.zeros_like(b), torch.zeros_like(b))

    def predict(x):                                                            # (B, d_in) -> (P, B, d_out)
        return torch.einsum("bi,pio->pbo", x, W) + b[:, None, :]

    for t, idx in enumerate(_batches(len(X), batch_size, epochs, seed), start=1):
        idx = idx.to(device)
        # Sum of per-probe MSEs: each probe's gradient is exactly its own MSE gradient.
        loss = ((predict(X[idx]) - y[idx]) ** 2).mean(dim=(1, 2)).sum()
        gW, gb = torch.autograd.grad(loss, (W, b))
        with torch.no_grad():
            _adam_step(W, gW, mW, vW, t, lr, wd)
            _adam_step(b, gb, mb, vb, t, lr[:, :, 0], 0.0)                     # U7: no decay on bias

    with torch.no_grad():
        val_pred = predict(torch.tensor(X_val, dtype=torch.float32, device=device)).cpu().numpy()
        test_pred = predict(torch.tensor(X_test, dtype=torch.float32, device=device)).cpu().numpy()
    return SweepResult(configs=configs, val_r2=r2_score(_as_2d(y_val), val_pred),
                       test_r2=r2_score(_as_2d(y_test), test_pred),
                       test_mae=mae(_as_2d(y_test), test_pred), test_pred=test_pred)
