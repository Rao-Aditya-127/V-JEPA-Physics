# Interpreting Physics in V-JEPA 2

How does a video world model represent physical variables in its latent space?

This project investigates that question on a **frozen V-JEPA 2 encoder**, for three
properties of a simple moving object: **direction**, **speed**, and
**acceleration**.

---

## The task

### Part 1 — Reproduction

Reproduce the main experimental progression from Sonia Joseph et al.,
[*Interpreting Physics in Video World Models*](https://arxiv.org/abs/2602.07050):

1. **Layer-wise probing.** Train and evaluate probes at every V-JEPA transformer
   layer for direction, speed, and acceleration. Plot probe performance across
   layers and use the curves to identify where each variable becomes available.
2. **Iterative nullspace probing.** Select a layer, then repeatedly fit a probe,
   remove its readout subspace, and fit another. Use the resulting performance
   curve to investigate the dimensionality and redundancy with which each variable
   is represented.
3. **Multi-probe subspace steering.** Intervene in the subspace defined by multiple
   probes, and evaluate the intervention on held-out data that was not used to
   construct the steering subspace.

The supplied data is smaller and simpler than the paper's. The aim is to reproduce
the **methodology and qualitative findings**, not the exact numbers.

### Part 2 — Extension

Apply the manifold-steering ideas from Goodfire / Wurgaft et al., [*Manifold
Steering Reveals the Shared Geometry of Neural Network Representation and
Behavior*](https://arxiv.org/abs/2605.05115) to V-JEPA's representations of
physical variables. At minimum, learn manifolds or splines for speed,
acceleration, and direction; decide how they should be constructed, visualised,
and evaluated; and compare spline steering against the multi-probe subspace method
of Part 1 — including strengths, limitations, and failure cases.

Two things need particular care: the **circular structure of direction**, and what
counts as a **meaningful held-out steering evaluation**.

---

## Setup

**Model.** [`facebook/vjepa2-vitl-fpc64-256`](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256)
— V-JEPA 2 ViT-L/16 at 256 px resolution, **kept frozen throughout**.
24 layers, hidden size 1024, patch size 16, tubelet size 2. A 16-frame 256×256
clip becomes 8 × 16 × 16 = 2048 tokens.

**Data.** Three synthetic datasets, each a single disk moving on a dark
background. Every clip is 16 frames, 256×256, at 24 fps.

| Dataset | Target | Clips | Range | Metrics |
|---|---|---:|---|---|
| `direction` | `theta_degrees` | 1,500 | 64 equally spaced directions in [0°, 360°) | circular MAE; R² on sin/cos |
| `speed` | `speed_mps` | 1,536 | 64 speeds, 0.25 – 4.0 m/s | MAE; R² |
| `acceleration` | `acceleration_mps2` | 1,536 | 64 accelerations, 0.25 – 10.0 m/s² | MAE; R² |

Direction is circular, so it is probed as `(sin θ, cos θ)` and the predicted pair
is converted back to an angle before computing circular error.

The datasets are **not tracked in this repository** — they are supplied material.
Place them at `data/<variable>/` with the original layout
(`manifest.jsonl` plus `videos/scene_XXXX/{video.mp4,metadata.json}`).

**Ground rules.** The encoder stays frozen. How intermediate representations are
extracted and pooled is documented. Data used to fit probes or manifolds is kept
strictly separate from data used for evaluation. The supplied videos and metadata
are never modified, and all derived artifacts are written outside `data/`.

---

## Repository

```
src/          extraction, probing, nullspace, steering, manifolds
artifacts/    cached activations, probes, figures        (not tracked)
data/         supplied videos and metadata               (not tracked)
```

---

## Deliverable

A ~15 minute presentation covering methods, results, interpretations,
comparisons, and limitations, serving as the basis for an open discussion.

---

## Status

Planning complete; implementation in progress. Results and usage instructions
will be added here as experiments land.
