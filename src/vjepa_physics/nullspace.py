"""Part 1.2: iterative nullspace projection (paper Appendix C.11).

The question: how many independent directions of the residual stream carry a
physical variable? C.11's answer is to keep deleting the direction a probe uses
until no probe can read the variable any more:

    1. train a linear probe on the current activations X
    2. take its weights W and orthonormalise them:  Q, _ = QR(W.T)
    3. delete that subspace:  X <- X - X Q Q.T
    4. repeat until the probe's score reaches chance

K, the number of rounds survived, is the variable's effective dimensionality: one
dimension per round for speed and acceleration, two for direction (sin, cos). A
curve that falls off a cliff means the variable lives in a few directions; one
that decays slowly means it is written redundantly across many.

Where the paper leaves a choice open, think/PLAN.md records the decision. The
three that shape this module:

  N4  features are standardised ONCE, before round 0, and never again. Re-scaling
      after a projection would partially undo it: if x.q = 0 and each coordinate is
      then rescaled by a diagonal D, (Dx).q = x.(Dq), which is not zero.
  N9  each round's basis is orthogonalised against everything removed so far, and
      the rank it actually adds is recorded rather than assumed. Adam's weights can
      keep a component along an already-deleted direction -- the gradient there is
      zero, so only weight decay shrinks it -- which would inflate the dimension count.
  --  Q is built from probes trained on the training clips only and applied unchanged
      to the test clips. Refitting it on the test set would leak.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from .probes import accuracy_within, circular_mae, fit_probe, mae, r2_score


def orthonormal_basis(W: np.ndarray, against: np.ndarray | None = None,
                      tol: float = 1e-8) -> np.ndarray:
    """An orthonormal basis for the subspace the probe reads from.

    W is (k_out, d) -- one row per probe output -- so the vectors become columns of
    W.T before QR, which expects them that way. Returns Q of shape (d, r) with
    Q.T @ Q = I, spanning the same subspace as W's rows.

    `against` (d, m), when given, is the basis of everything already removed: its
    span is subtracted from W first (twice -- the standard re-orthogonalisation,
    since one pass leaves rounding-level residue), so the returned columns are
    perpendicular to it and r counts only genuinely new dimensions.

    Columns whose R diagonal is negligible are dropped: they mean W's rows were
    linearly dependent, and QR fills such columns with arbitrary vectors that do
    not lie in W's span. r < k_out is therefore possible and is not an error.
    """
    V = np.asarray(W, dtype=np.float64).T                       # (d, k_out)
    scale = float(np.linalg.norm(V, axis=0).max())
    if against is not None and against.shape[1]:
        for _ in range(2):
            V = V - against @ (against.T @ V)
    Q, R = np.linalg.qr(V)
    keep = np.abs(np.diag(R)) > tol * max(scale, 1e-300)
    return Q[:, keep]


def project(X: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """X - X Q Q.T: delete the subspace spanned by Q from every row of X.

    Grouped this way it costs two (N, d) x (d, r) products; forming Q Q.T first
    would build a d x d matrix of rank r for no reason.
    """
    return X - (X @ Q) @ Q.T


def leakage(W: np.ndarray, basis: np.ndarray) -> float:
    """How much of the probe's weight points into already-removed directions, as a
    fraction of its length. Zero for a least-squares fit; small but non-zero for
    Adam, whose random initialisation lands partly in the deleted subspace and is
    never pushed out of it by a gradient (N9). A diagnostic, not a correction."""
    if basis.shape[1] == 0:
        return 0.0
    W = np.asarray(W, dtype=np.float64)
    return float(np.linalg.norm(basis.T @ W.T) / (np.linalg.norm(W) + 1e-12))


def _variance_removed(X: np.ndarray, Q: np.ndarray) -> float:
    """Fraction of the activations' total variance that this round deletes.

    A direction found by an Adam probe is part signal and part leftover random
    initialisation -- the trained weight vector is barely longer than the vector it
    started from, and no gradient ever removes the unused part, because it lies in
    directions the data hardly occupies. Those directions carry little variance, so
    a small number here means the round removed little of the representation, and K
    partly measures the probe rather than the encoder.
    """
    total = float((X ** 2).sum())
    return float(((X @ Q) ** 2).sum() / total)


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 100.0) -> torch.nn.Linear:
    """[OURS] A closed-form probe, for the diagnostic C.11 cannot run.

    Ridge's solution lies in the row space of X, so its weights contain no leftover
    random initialisation: the direction it hands over is entirely signal. Adam's is
    roughly half initialisation (see `leakage`), which makes each removal less
    effective and inflates K. Running the same loop both ways separates what the
    encoder does from what the optimiser does. Returned as an nn.Linear, in float64,
    so the rest of the loop treats it exactly like a trained probe.
    """
    from sklearn.linear_model import Ridge

    y2 = y.reshape(len(y), -1)
    model = Ridge(alpha=alpha).fit(X, y2)
    linear = torch.nn.Linear(X.shape[1], y2.shape[1]).to(torch.float64)
    with torch.no_grad():
        linear.weight.copy_(torch.as_tensor(np.atleast_2d(model.coef_), dtype=torch.float64))
        linear.bias.copy_(torch.as_tensor(np.atleast_1d(model.intercept_), dtype=torch.float64))
    return linear


@torch.inference_mode()
def predict(probe: torch.nn.Linear, X: np.ndarray) -> np.ndarray:
    return probe(torch.tensor(X, dtype=probe.weight.dtype,
                              device=probe.weight.device)).cpu().numpy()


@dataclass
class Sequence:
    """One run of the loop on one split."""
    rounds: pd.DataFrame                  # one row per round; see the columns built below
    basis: np.ndarray                     # (d, dims_removed) everything deleted, orthonormal
    weights: list[np.ndarray] = field(default_factory=list)   # each round's raw W, for Part 1.3

    @property
    def dims_removed(self) -> int:
        return self.basis.shape[1]


def _score(y_true: np.ndarray, pred: np.ndarray, angles: np.ndarray | None) -> dict:
    y = y_true.reshape(len(y_true), -1)
    out = {"r2": float(r2_score(y, pred)), "mae": float(mae(y, pred))}
    if angles is not None:
        out["circ_mae"] = circular_mae(angles, pred)
        out["acc15"] = accuracy_within(angles, pred, tol=15.0)
    return out


def run_sequence(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                 *, lr: float, wd: float, epochs: int, batch_size: int = 32, seed: int = 0,
                 max_rounds: int = 60, stop_r2: float = 0.05, patience: int = 3,
                 angles_test: np.ndarray | None = None, device: str = "cpu",
                 dims_per_round: int | None = None, probe_fit: str = "adam",
                 ridge_alpha: float = 100.0, verbose: bool = True) -> Sequence:
    """Run steps 1-4 until the probe reaches chance, or `max_rounds` rounds.

    X_train and X_test must already be standardised (N4). y is (n,) for a scalar
    target or (n, 2) = (sin, cos) for direction; `angles_test` carries the true
    angles in degrees so circular error can be reported alongside R2.

    Stopping (N5): the loop ends when test R2 stays below `stop_r2` for `patience`
    consecutive rounds. A single dip is not enough, because direction's curve is a
    sawtooth -- it drops and recovers as sin/cos pairs are removed one half at a
    time. `stop_r2` is the *strictest* threshold the paper gives, so K under looser
    ones (Fig. 22's) can be read off the recorded curve afterwards.

    Two options exist only for diagnostics and are not part of C.11:

    `dims_per_round` caps how many of the basis columns are actually deleted. The
    paper attributes direction's sawtooth to sin/cos feature pairs; setting this to 1
    deletes one half of the pair per round, alternating between them, which is the
    only way an alternating curve can arise. The default deletes the whole basis, as
    C.11 specifies.

    `probe_fit="ridge"` swaps Adam for the closed-form solution -- see `fit_ridge`.
    """
    X_train = np.asarray(X_train, dtype=np.float64)              # float64: the projection is
    X_test = np.asarray(X_test, dtype=np.float64)                # applied hundreds of times
    basis = np.zeros((X_train.shape[1], 0))
    rows, weights, below = [], [], 0

    # Always predicting the training mean -- the "random baseline" C.11's speed rule
    # compares against, and the definition of R2 = 0.
    train_2d = y_train.reshape(len(y_train), -1)
    baseline = np.broadcast_to(train_2d.mean(axis=0), (len(y_test), train_2d.shape[1]))
    baseline_mae = float(mae(y_test.reshape(len(y_test), -1), baseline))

    for k in range(max_rounds):
        probe = (fit_ridge(X_train, y_train, ridge_alpha) if probe_fit == "ridge" else
                 fit_probe(X_train, y_train, lr=lr, wd=wd, epochs=epochs,
                           batch_size=batch_size, seed=seed, device=device))
        W = probe.weight.detach().cpu().numpy().astype(np.float64)          # (k_out, d)
        test = _score(y_test, predict(probe, X_test), angles_test)
        train = _score(y_train, predict(probe, X_train), None)

        Q = orthonormal_basis(W, against=basis)
        if dims_per_round is not None and Q.shape[1] > dims_per_round:
            # Rotate which columns are taken, so a 2-output probe loses its sin half in
            # one round and its cos half in the next. Always taking the first column
            # would chase the same output forever and leave the other one readable.
            pick = [(k * dims_per_round + j) % Q.shape[1] for j in range(dims_per_round)]
            Q = Q[:, pick]
        leak = leakage(W, basis)
        variance = _variance_removed(X_train, Q)
        X_train, X_test = project(X_train, Q), project(X_test, Q)

        # The same probe, scored again now that the subspace it reads from is gone. Its
        # weighted sum is zero by construction, so it emits only its bias -- for direction
        # that is a single fixed angle, and accuracy collapses to an arithmetic floor. This
        # is not a measure of the representation; it is recorded because interleaving it
        # with the retrained score reproduces the paper's Fig. 4c sawtooth exactly, which
        # the paper attributes to sin/cos feature pairs. See results/nullspace-probing.
        stale = _score(y_test, predict(probe, X_test), angles_test)

        row = {"round": k,
               "dims_removed": basis.shape[1],        # x-axis: dimensions gone BEFORE this probe
               "rank_this_round": int(Q.shape[1]),
               "probe_outputs": int(W.shape[0]),
               "leakage": leak,
               "variance_removed": variance,
               "weight_norm": float(np.linalg.norm(W)),
               "train_r2": train["r2"], "test_r2": test["r2"], "test_mae": test["mae"],
               "mae_vs_baseline": test["mae"] / (baseline_mae + 1e-12)}
        row.update({f"stale_{key}": value for key, value in stale.items()})
        if angles_test is not None:
            row["test_circ_mae"], row["test_acc15"] = test["circ_mae"], test["acc15"]
        rows.append(row)
        weights.append(W)

        if verbose:
            extra = (f"  circMAE {row['test_circ_mae']:5.1f} deg  acc@15 {row['test_acc15']:.3f}"
                     if angles_test is not None else f"  MAE/base {row['mae_vs_baseline']:.2f}")
            print(f"    round {k:3d}  dims removed {row['dims_removed']:4d}  "
                  f"test R2 {row['test_r2']:7.3f}  (train {row['train_r2']:6.3f}){extra}"
                  f"  leak {row['leakage']:.3f}", flush=True)

        basis = np.concatenate([basis, Q], axis=1)

        below = below + 1 if test["r2"] < stop_r2 else 0
        if below >= patience:
            if verbose:
                print(f"    stopped: test R2 < {stop_r2} for {patience} consecutive rounds")
            break

    return Sequence(rounds=pd.DataFrame(rows), basis=basis, weights=weights)


def dimensionality(rounds: pd.DataFrame, threshold: float, patience: int = 3,
                   column: str = "test_r2") -> dict:
    """K and the effective dimensionality at a given stopping threshold.

    K is the first round after which the score stays below `threshold` for
    `patience` consecutive rounds -- i.e. the number of probes that were still
    readable. `dims` is the number of activation dimensions those probes spanned,
    counted from the measured rank, which is 2K for direction and K for the scalars.
    Returns reached=False when the curve never settles below the threshold, in
    which case K is a lower bound set by the round cap.
    """
    score = rounds[column].to_numpy()
    below = score < threshold
    for k in range(len(score) - patience + 1):
        if below[k:k + patience].all():
            return {"threshold": threshold, "K": int(k), "reached": True,
                    "dims": int(rounds.dims_removed.iloc[k])}
    last = int(rounds.dims_removed.iloc[-1] + rounds.rank_this_round.iloc[-1])
    return {"threshold": threshold, "K": int(len(score)), "reached": False, "dims": last}
