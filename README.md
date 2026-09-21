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
config.yaml              every path and hyperparameter, in one place
src/vjepa_physics/
  video.py               mp4 -> RGB uint8 frames (torchcodec or OpenCV)
  data.py                manifest + metadata -> one records table
  encoder.py             frozen V-JEPA 2: preprocess, forward, mean-pool
  extract.py             cache pooled features for a whole dataset (resumable)
  features.py            load cached features; paper layer l == hidden_states[l+1]
  folds.py               5-fold cross-validation grouped by label value
  probes.py              linear probes and the 20-config Adam sweep
  nullspace.py           Part 1.2: QR, projection, the probe sequence
  steering.py            Part 1.3: the steering subspace and the target solve
  plots.py               figures
scripts/                 pipeline: each step writes artifacts
  00_sanity.py           smoke test before a full run
  01_extract.py          extraction (needs the encoder; GPU recommended)
  02_layerwise_probe.py  Part 1.1 layer-wise probing (CPU is fine)
  03_nullspace.py        Part 1.2 iterative nullspace probing (CPU is fine)
  04_steering.py         Part 1.3 multi-probe subspace steering (CPU is fine)
  figures/               reporting: read cached results, write figures
    layerwise.py         Part 1.1, all three variables, like the paper's Fig. 2c
    nullspace.py         Part 1.2 curves + the dimensionality table
    figure4c.py          Part 1.2 direction, in the form of the paper's Fig. 4c
    steering.py          Part 1.3 steering curves, like the paper's Fig. 24
tests/                   decode, data, folds, probe and nullspace correctness
artifacts/               features, splits, results, figures   (not tracked)
data/                    supplied videos and metadata          (not tracked)
```

---

## Running it

```bash
pip install -r requirements.txt     # or: pip install -e .
python -m pytest -q                 # decode, data, fold and probe checks

python scripts/00_sanity.py         # 4 checks incl. a viability probe; exits non-zero on failure
python scripts/01_extract.py --dataset speed               # -> artifacts/features/speed
python scripts/02_layerwise_probe.py --variable speed      # Part 1.1, -> artifacts/results/speed
python scripts/03_nullspace.py --variable speed           # Part 1.2, ~6 min on CPU
python scripts/04_steering.py --variable speed            # Part 1.3, ~1 min on CPU

python scripts/figures/layerwise.py                       # once all three variables are probed
python scripts/figures/nullspace.py
python scripts/figures/figure4c.py
python scripts/figures/steering.py
```

Extraction is the only step that needs the encoder: roughly 10 s per clip on a
4-core CPU, a few minutes for the whole speed dataset on a T4. It resumes if
interrupted. Probing reads the cached features and runs in minutes on CPU; add
`--device cuda` to run it on a GPU instead.

By default, probing runs exactly the experiment the brief asks for. Additional
checks beyond it (controls, the patch embedding) are opt-in, with
`--conditions embedding random_cv shuffled pixels`; they are added to an existing
run's results rather than replacing them.

`01_extract.py` and `02_layerwise_probe.py` accept `--limit N` to work on a
seeded random subset of N clips, written to its own directory (`speed_nN`) so a
debug run is never mistaken for the full one.

---

## Deliverable

A ~15 minute presentation covering methods, results, interpretations,
comparisons, and limitations, serving as the basis for an open discussion.

---

## Results

| Part | Write-up |
|---|---|
| 1.1 Layer-wise probing | [results/layer-wise-probing](results/layer-wise-probing/README.md) |
| 1.2 Iterative nullspace probing | [results/nullspace-probing](results/nullspace-probing/README.md) |
| 1.3 Multi-probe subspace steering | [results/subspace-steering](results/subspace-steering/README.md) |

## Status

Part 1 is complete for all three variables, with write-ups and figures. Part 2
(spline / manifold steering, and its comparison against Part 1.3) is next.
