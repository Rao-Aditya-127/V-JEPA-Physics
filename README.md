# Interpreting Physics in V-JEPA 2

How does a video world model represent physical variables in its latent space?

This project investigates that question on a **frozen V-JEPA 2 encoder**, for three
properties of a simple moving object: **direction**, **speed**, and
**acceleration**.

---

Part 1 reproduces Sonia Joseph et al., [*Interpreting Physics in Video World
Models*](https://arxiv.org/abs/2602.07050). Part 2 extends it with the
manifold-steering method of Goodfire / Wurgaft et al., [*Manifold Steering Reveals
the Shared Geometry of Neural Network Representation and
Behavior*](https://arxiv.org/abs/2605.05115).

---

## Findings

Every score below is measured on held-out label values: a probe, curve or steering
method is always tested on values it never saw during fitting. Parts 1.2, 1.3 and 2
use layer 8, the paper's one-third-depth "Physics Emergence Zone".

### Part 1.1 — Layer-wise probing: where each variable becomes available

- **Speed and acceleration are available from the first block** (held-out R² 0.977
  and 0.967) and stay high at every layer.
- **Direction is the variable that needs depth.** It is the weakest early (R² 0.795,
  a 17° error) and improves the most, reaching 0.977 (4.4°) by layer 8. This is the
  paper's central asymmetry, and it reproduces.
- **The paper's sharp jump does not.** Its direction curve leaps from about 0.2 to
  0.9 at layer 8; ours rises gradually and is mostly complete by layer 5. The
  write-up lists the candidate reasons, chief among them our simpler 2-D scene.
- From layer 12 onward all three variables sit between R² 0.983 and 0.990.

→ [results/layer-wise-probing](results/layer-wise-probing/README.md)

### Part 1.2 — Iterative nullspace probing: dimensionality and redundancy

- **Every variable is spread over many directions.** Deleting the single best probe
  direction changes nothing measurable (speed R² 0.981 → 0.982); tens of directions
  must go before any variable becomes unreadable.
- **Direction is by far the most redundant.** A freshly trained probe can still read
  it until **116** dimensions have been deleted, against **38–47** for speed and
  **24–31** for acceleration; the range depends on whether the paper's stopping rule
  is applied through its R² clause alone or through R² or error. The paper reports
  136 for direction and 28 for speed. All three are equally readable at layer 8, yet
  stored very differently.
- **The exact count depends on how the probe is fitted.** With an exactly solved
  (ridge) probe, direction falls to 48 dimensions. The ordering, direction far above
  the scalars, holds either way; the number itself does not.
- **The paper's sawtooth (Fig. 4c) does not appear with freshly trained probes**,
  under any of five setups. It appears when each probe is also scored after its own
  directions have been deleted: the probe can then only answer 0° or 180°, which pins
  the dips at a floor of 10 of 64 directions, 15.6% (measured: 15.0%).

→ [results/nullspace-probing](results/nullspace-probing/README.md) ·
[verification appendix](results/nullspace-probing/sawtooth.md)

### Part 1.3 — Multi-probe subspace steering

- **Direction can be overwritten.** Steering every held-out clip to 90° takes a
  held-out probe's error from 89.1° to 7.5° with 20 probes (paper: 82.9° → 11.9°),
  and to 5.5° at best, with 31.
- **It takes many probes moving together.** One alone closes only half the gap
  (44.9°); about ten are needed. There is no single dial for direction, as the paper
  claims.
- **The old value is really gone.** As the error to the target falls, the error to
  each clip's true direction rises to 88°. Speed and acceleration, which the paper
  does not steer, behave the same way (88–89% of the gap closed), and the result holds
  across 8–9 different targets.
- **The evaluation is stricter than the paper's.** The judging probe is trained only
  on clips whose values the steering probes never saw; the paper's single random
  split shares all 8 of its directions between the two.
- **The edit is large.** The best setting moves each clip 60% of its own length, far
  outside anything the encoder produces from a real video.

→ [results/subspace-steering](results/subspace-steering/README.md)

### Part 2 — Activation manifolds: construction, visualisation and evaluation

- **Direction is a closed ring.** This was tested, not assumed: distances between
  direction centroids follow a circle's chord formula (opposite directions sit 19.6×
  further apart than neighbours; a circle predicts 20.4). Its spline is periodic, so
  354.375° joins back to 0°.
- **Speed and acceleration are gently bowed, open arcs**, cleanly ordered (the first
  principal direction correlates 0.93–0.96 with the value).
- **Construction follows Goodfire's recipe**, with three changes. PCA keeps 7–14
  dimensions, chosen to hold 90% of the variance between value averages rather than a
  fixed 64; clips are averaged at each value; one cubic smoothing spline is fitted per
  coordinate, with the smoothing chosen on held-out values. Figures are drawn in the
  centroids' own top three directions, because in the clips' directions the ring is
  seen edge-on.
- **The curvature is real only for direction.** At reconstructing held-out clips,
  the curve beats a straight line by 12.7% for direction but by only 1.6–1.8% for
  speed and acceleration.
- **The curve is an average, not where clips sit.** Clips scatter 1.4–2.6× further
  from it than it is long, because start position and the other variables move them.

→ [results/manifold-steering/activation-manifolds](results/manifold-steering/activation-manifolds/README.md)

### Part 2 — Spline steering against multi-probe subspace steering

- **The two methods differ in one respect.** Subspace steering *replaces* the clip's
  coordinates in its 25–62 probe dimensions with a template that is the same for every
  clip. Manifold steering *shifts* the clip along the curve, from its own value to the
  target, and keeps everything else.
- **Head to head**, on the same held-out clips, over 8–9 targets, judged both by a
  probe trained on held-out clips and by the nearest real clips:
  - *Reaching the target:* manifold is better for direction (4.8° vs 6.1°), subspace
    for acceleration (0.28 vs 0.36 m/s²), and speed is a tie. Both judges agree.
  - *Leaving the rest of the clip alone:* manifold, for every variable. Steering
    acceleration with the subspace method also turns the clip's direction by 7.4°
    (manifold: 1.6°), and start position moves up to 7.5× more.
  - *Changing nothing when asked to:* manifold leaves the clip exactly as it was;
    subspace still rewrites 16–41% of it.
- **In short, subspace steering hits the number and manifold steering keeps the
  clip.** Manifold steering is the better choice where the shape is genuinely curved;
  subspace steering reaches the target more reliably where it is not, at a cost to
  everything else in the clip.
- **Against a straight line** (Goodfire's own comparison), following the ring cuts
  direction's steering error by 59% over the journey and by 85% on half-circle trips,
  where the straight path crosses the empty centre of the ring. For speed and
  acceleration it is 19–38% worse, as the curve-vs-line test predicted.

→ [results/manifold-steering/steering-comparison](results/manifold-steering/steering-comparison/README.md) ·
[`scripts/07_steering_vs_subspace.py`](scripts/07_steering_vs_subspace.py)

### What these findings do not show

- **Steering is judged by readouts.** Every steering result is what a probe, or the
  nearest real clips, reads. The encoder's later layers and V-JEPA's predictor are
  never run on a steered activation.
- **One summary per clip, one layer.** Features are averaged over all 2,048 tokens,
  and everything after Part 1.1 uses layer 8.
- **Simpler scenes than the paper's.** The supplied clips are a flat 2-D disk; the
  paper's are rendered 3-D scenes, which may account for some of the differences
  above.

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
  manifold.py            Part 2: PCA + splines, the curve the activations occupy
  plots.py               figures
scripts/                 pipeline: each step writes artifacts
  00_sanity.py           smoke test before a full run
  01_extract.py          extraction (needs the encoder; GPU recommended)
  02_layerwise_probe.py  Part 1.1 layer-wise probing (CPU is fine)
  03_nullspace.py        Part 1.2 iterative nullspace probing (CPU is fine)
  04_steering.py         Part 1.3 multi-probe subspace steering (CPU is fine)
  05_manifold.py         Part 2 fit and evaluate the activation manifold
  06_manifold_steering.py  Part 2 steer along it, against linear and Part 1.3
  07_steering_vs_subspace.py  Part 2 head to head with Part 1.3: two judges, side effects
  figures/               reporting: read cached results, write figures
    layerwise.py         Part 1.1, all three variables, like the paper's Fig. 2c
    nullspace.py         Part 1.2 curves + the dimensionality table
    figure4c.py          Part 1.2 direction, in the form of the paper's Fig. 4c
    steering.py          Part 1.3 steering curves, like the paper's Fig. 24
    manifold.py          Part 2 the manifold, in 3D and flat, plus its evaluation
    manifold_steering.py Part 2 the steering comparison
tests/                   decode, data, fold, probe, nullspace, steering and manifold correctness
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
python scripts/05_manifold.py --variable speed            # Part 2,   ~40 s on CPU
python scripts/06_manifold_steering.py --variable speed   # Part 2,   ~40 s on CPU
python scripts/07_steering_vs_subspace.py --variable speed  # Part 2, ~40 s on CPU

python scripts/figures/layerwise.py                       # once all three variables are probed
python scripts/figures/nullspace.py
python scripts/figures/figure4c.py
python scripts/figures/steering.py
python scripts/figures/manifold.py --variable speed
python scripts/figures/manifold_steering.py --variable speed
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

## Results

| Part | Write-up |
|---|---|
| 1.1 Layer-wise probing | [results/layer-wise-probing](results/layer-wise-probing/README.md) |
| 1.2 Iterative nullspace probing | [results/nullspace-probing](results/nullspace-probing/README.md) |
| 1.3 Multi-probe subspace steering | [results/subspace-steering](results/subspace-steering/README.md) |
| 2 Manifold steering | [results/manifold-steering](results/manifold-steering/README.md) |
