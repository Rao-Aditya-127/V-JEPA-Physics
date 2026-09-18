"""Load cached features, and the one place that knows how layers are indexed.

The cached array has num_layers + 1 entries along axis 1. The paper probes
l in {0, ..., n-1} (24 layers for ViT-L) and calls one-third depth "layer 8".
HuggingFace's hidden_states puts the patch embedding at index 0, so:

    paper layer l   ==   pooled[:, l + 1, :]
    paper layer 8   ==   pooled[:, 9, :]

Getting this off by one shifts the whole curve against the paper's.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EMBEDDING_INDEX = 0


def hidden_index(paper_layer: int) -> int:
    return paper_layer + 1


def layer_fraction(paper_layer: int, num_layers: int = 24) -> float:
    """x-axis of the paper's Figure 2: layer / (n - 1), so layer 8 of 24 sits at 0.348."""
    return paper_layer / (num_layers - 1)


@dataclass
class Features:
    pooled: np.ndarray       # (N, num_layers + 1, hidden) float32
    pixels: np.ndarray       # (N, pixel_dim) float32
    labels: pd.DataFrame
    meta: dict

    @property
    def num_layers(self) -> int:
        return self.pooled.shape[1] - 1

    def layer(self, paper_layer: int) -> np.ndarray:
        return self.pooled[:, hidden_index(paper_layer), :]

    def embeddings(self) -> np.ndarray:
        return self.pooled[:, EMBEDDING_INDEX, :]


def load_features(feat_dir: str | Path, allow_partial: bool = False) -> Features:
    feat_dir = Path(feat_dir)
    done = np.load(feat_dir / "done.npy")
    if not done.all() and not allow_partial:
        raise RuntimeError(f"{feat_dir}: only {done.sum()}/{len(done)} clips extracted. "
                           "Finish the extraction (it resumes) or pass allow_partial=True.")
    labels = pd.read_csv(feat_dir / "labels.csv")
    pooled = np.load(feat_dir / "pooled.npy").astype(np.float32)[done]
    pixels = np.load(feat_dir / "pixels.npy").astype(np.float32)[done]
    labels = labels[done].reset_index(drop=True)
    if len(labels) != len(pooled):
        raise RuntimeError("labels.csv and pooled.npy are misaligned")
    meta = json.loads((feat_dir / "meta.json").read_text(encoding="utf-8")) \
        if (feat_dir / "meta.json").exists() else {}
    return Features(pooled=pooled, pixels=pixels, labels=labels, meta=meta)
