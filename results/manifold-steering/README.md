# Manifold steering (Part 2)

Part 1 established that physical variables are linearly readable from V-JEPA 2's
residual stream, that they occupy tens of dimensions, and that overwriting those
dimensions changes what an independent reader sees. All of it treats activation space
as **flat**.

Part 2 asks whether that assumption is right, following Wurgaft et al., [*Manifold
Steering Reveals the Shared Geometry of Neural Network Representation and
Behavior*](https://arxiv.org/abs/2605.05115). Their argument: real activations lie on
a curved surface, and an intervention that ignores the curvature drives through states
the model never produces.

The paper does not use V-JEPA — its video experiment is a small recurrent world model
on Mountain Car — so this is a **transfer of their method**, not a reproduction. There
are no numbers of theirs to match, and the brief describes the part as intentionally
open-ended.

## Stages

| | |
|---|---|
| **[1. The geometry](activation-manifolds/README.md)** | What shape do the activations occupy? Fit it, visualise it, and test whether the curvature is real. |
| **[2. Steering along it](steering-comparison/README.md)** | Move along the curve instead of across it, and compare against Part 1.3's multi-probe subspace method. |

## What stage 1 found

**Direction is a circle.** Measured, not assumed: the distance between two direction
centroids follows the chord formula `2R·sin(Δθ/2)`, and opposite angles sit 19.6×
further apart than neighbours where a perfect circle predicts 20.4. Its manifold is
fitted with a periodic spline so the loop closes.

**Speed and acceleration are gently bowed arcs.** Cleanly ordered — PC1 correlates
0.93–0.96 with the value — but only 1.6–1.8% better described by a curve than by a
straight line.

**That difference predicted stage 2, and stage 2 confirmed it.** Manifold steering
beats linear steering by **59%** for direction — rising to **85%** for clips crossing
more than half the circle — and *loses* by 19–38% for the scalars, where the fitted
curvature is mostly noise and following it is a detour.

The useful form of that result: a cheap test run beforehand (does the curve beat a
straight line at reconstructing held-out clips?) predicts which way steering will go.
Direction +12.7%, the scalars +1.6–1.8%.

**A by-product worth noting:** projecting a clip onto the fitted curve decodes the
variable with no probe at all — 8.96° for direction, 0.31 m/s for speed — roughly half
as accurate as Part 1.1's trained probes, using only geometry.

Full detail, configuration and figures: **[activation-manifolds](activation-manifolds/README.md)**.

## Relation to Part 1

| | [1.2 Nullspace](../nullspace-probing/README.md) | [1.3 Steering](../subspace-steering/README.md) | Part 2 |
|---|---|---|---|
| Question | how many directions carry it? | can we overwrite it? | what shape does it occupy? |
| Geometry assumed | flat | flat | **curved, and measured** |
| Object fitted | a sequence of probes | a subspace from those probes | a spline through the activations |
