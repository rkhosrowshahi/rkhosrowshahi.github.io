---
layout: post
title: CurveBench — how neural networks learn curves
date: 2026-07-12 00:00:00
description: A small benchmark for 1D curve learning — comparing network width, activation, and optimizer (gradient descent vs evolution).
tags: research machine-learning optimization neural-networks
categories: research
thumbnail: assets/img/curvebench/preview.png
related_posts: true
---

[CurveBench](https://github.com/rkhosrowshahi/curvebench) is an open-source benchmark for **1D curve learning**: given samples \((x, y)\), how well can a small MLP fit \(y = f(x)\), and how does that depend on **width**, **activation**, and **optimizer**?

The repo trains three ReLU MLPs in parallel (widths 10 / 100 / 1000) with either **PyTorch gradient methods** (SGD, L-BFGS, RMSprop, Adam, AdamW) or **JAX evolutionary optimizers** via evosax (Differential Evolution, CMA-ES, Sep-CMA-ES). Every run exports a **training video** and a final **figure** so you can watch the prediction curve converge.

```bash
pip install -e .
curvebench --config configs/fourier3/adam.yaml
```

---

## Experiment 1: does width matter?

Target (Fourier sum on \(x \in [-10, 10]\)):

\[
y = \sin(x) + \tfrac{1}{2}\sin(2x) + \tfrac{1}{3}\sin(3x)
\]

| Network | Architecture \(d\) | Parameters |
|---------|-------------------|------------|
| N1 | `[1, 10, 10, 1]` | 141 |
| N2 | `[1, 100, 100, 1]` | 10,401 |
| N3 | `[1, 1000, 1000, 1]` | 1,004,001 |

Wider ReLU nets can stitch together more linear pieces, so they approximate this smooth target more closely — but **optimizer choice dominates** at a fixed width.

### Adam (1600 epochs)

Three networks trained together with full-batch Adam:

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include video.liquid path="assets/video/curvebench/fourier3-adam.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>
<div class="caption">
  Adam on the Fourier target. N3 (purple) reaches the lowest MSE; N1 struggles.
</div>

### SGD (1600 epochs)

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include video.liquid path="assets/video/curvebench/fourier3-sgd.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>
<div class="caption">
  SGD + momentum. N3 diverges (NaN); wider nets are not automatically easier for every optimizer.
</div>

Final SGD snapshot:

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/curvebench/fourier3-sgd-final.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>

### Differential Evolution (1600 steps)

Population-based search with memory-aware popsize (~\(10\sqrt{D}\)):

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include video.liquid path="assets/video/curvebench/fourier3-de.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>
<div class="caption">
  DE at 1600 steps. Usable on N1/N2; N3 is RAM-limited (small population).
</div>

With **10k steps** and \(F=0.5\), DE reaches strong N1/N2 fits (best N1 MSE ≈ 0.0066 in our runs).

### CMA-ES on N1 (1600 steps)

Full CMA-ES is only practical for N1 (~141 parameters); larger nets switch to Sep-CMA-ES:

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include video.liquid path="assets/video/curvebench/fourier3-cmaes.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>

### Takeaway (width experiment)

At 1600 steps with ReLU, **Adam / AdamW** still win on wide nets (~0.012 MSE on N2/N3). Evolution can match or beat gradient methods on small nets given enough steps, but wide-net EA remains expensive.

---

## Experiment 2: learning \(\sin(x)\) — activation and OOD

A second target asks a different question: can the network **extrapolate** a periodic signal?

- **Train** on \(x \in [-1, 1]\)
- **Evaluate** on \(x \in [-3, 3]\) (OOD regions shaded yellow)
- Single network \(d = [1, 10, 10, 1]\)

```bash
curvebench --config configs/sin/relu.yaml
curvebench --config configs/sin/tanh.yaml
curvebench --config configs/sin/sin.yaml
```

**ReLU** — piecewise-linear extrapolation fails outside the training interval:

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/curvebench/sin-relu-final.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>

**\(\sin\) activation** — periodic inductive bias fits the target on and off the training domain:

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/curvebench/sin-sin-final.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>

This mirrors the classic lesson: **architecture and activation encode prior knowledge** about the function class, not just capacity.

---

## Reproducing results

Configs live under `configs/` in the repo. Examples:

```yaml
# configs/fourier3/adam.yaml
target: fourier3
activation: relu
optimizer: adam
epochs: 1600
```

```yaml
# configs/sin/relu.yaml
target: sin
activation: relu
optimizer: adam
epochs: 10000
widths: [10]
train_x_min: -1
train_x_max: 1
```

Outputs are written to `outputs/<target>/<activation>/<Method>/` with `figure.png` and `training.mp4`.

**Repository:** [github.com/rkhosrowshahi/curvebench](https://github.com/rkhosrowshahi/curvebench) (MIT)
