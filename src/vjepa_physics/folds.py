"""Cross-validation splits.

Appendix B: "5-fold grouped cross-validation", with the grouping variable unstated.
U1: we group by the label value, so every test fold holds out whole speed values and
the probe must predict speeds it never trained on.

Each fold has three disjoint parts:

    fit   -- the probe trains on these
    val   -- inner validation, picks the (lr, wd) config          (U3)
    test  -- the held-out fold, touched only to report a number

Groups are assigned to folds *interleaved in sorted order* (value i -> fold i % k)
rather than at random. R^2 is computed relative to the variance of the test fold's
own labels, so a fold that happened to receive a clump of similar speeds would get a
deflated R^2 for the same error. Interleaving gives every fold the full label range.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Fold:
    fit: np.ndarray
    val: np.ndarray
    test: np.ndarray


def grouped_folds(groups: np.ndarray, n_folds: int = 5, val_every: int = 5) -> list[Fold]:
    """Folds where no group value appears in more than one of fit / val / test."""
    groups = np.asarray(groups)
    values = np.unique(groups)                       # sorted
    if len(values) < n_folds * 2:
        raise ValueError(f"only {len(values)} distinct groups for {n_folds} folds")
    folds = []
    for k in range(n_folds):
        test_values = values[np.arange(len(values)) % n_folds == k]
        train_values = values[np.arange(len(values)) % n_folds != k]
        val_values = train_values[np.arange(len(train_values)) % val_every == val_every // 2]
        fit_values = np.setdiff1d(train_values, val_values)
        folds.append(Fold(fit=np.flatnonzero(np.isin(groups, fit_values)),
                          val=np.flatnonzero(np.isin(groups, val_values)),
                          test=np.flatnonzero(np.isin(groups, test_values))))
    return folds


def random_folds(n: int, n_folds: int = 5, val_fraction: float = 0.2, seed: int = 0) -> list[Fold]:
    """[OURS] Ungrouped comparison: clips assigned at random, so every label value
    appears in training. Measures how much the grouped score is helped by having
    seen the exact test values."""
    rng = np.random.default_rng(seed)
    chunks = np.array_split(rng.permutation(n), n_folds)
    folds = []
    for k in range(n_folds):
        train = rng.permutation(np.concatenate([c for j, c in enumerate(chunks) if j != k]))
        n_val = int(round(len(train) * val_fraction))
        folds.append(Fold(fit=np.sort(train[n_val:]), val=np.sort(train[:n_val]),
                          test=np.sort(chunks[k])))
    return folds


def check_folds(folds: list[Fold], n: int, groups: np.ndarray | None = None) -> None:
    """Every row is tested exactly once; parts are disjoint; with groups, no value leaks."""
    tested = np.concatenate([f.test for f in folds])
    if sorted(tested.tolist()) != list(range(n)):
        raise AssertionError("test folds do not partition the rows")
    for f in folds:
        parts = [set(f.fit.tolist()), set(f.val.tolist()), set(f.test.tolist())]
        if parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2]:
            raise AssertionError("fit / val / test overlap")
        if groups is not None:
            g = [set(np.asarray(groups)[list(p)].tolist()) for p in parts]
            if g[0] & g[1] or g[0] & g[2] or g[1] & g[2]:
                raise AssertionError("a group value appears in more than one part")


def save_folds(folds: list[Fold], clip_ids: np.ndarray, path: str | Path, scheme: str) -> None:
    """Store clip ids (not row positions) so the file survives re-ordering."""
    clip_ids = np.asarray(clip_ids)
    payload = {"scheme": scheme, "folds": [
        {part: clip_ids[getattr(f, part)].tolist() for part in ("fit", "val", "test")} for f in folds]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload), encoding="utf-8")
