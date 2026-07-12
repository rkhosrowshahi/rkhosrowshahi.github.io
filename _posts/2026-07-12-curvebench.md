---
layout: post
title: CurveBench — how neural networks learn curves
date: 2026-07-12 00:00:00
description: Watching MLPs learn 1D curves — width, optimizer, and activation change what the prediction looks like as training unfolds.
tags: research machine-learning optimization neural-networks
categories: research
thumbnail: assets/img/curvebench/preview.png
related_posts: true
---

How does a neural network learn a curve?

Not from a single loss number at the end — but from the **shape of the prediction** as training progresses. Does the fitted curve smooth out harmonic by harmonic? Does a wide net lock in quickly while a narrow one lags? Does SGD diverge while Adam stabilizes? Does a population-based search slowly crawl toward the target?

[CurveBench](https://github.com/rkhosrowshahi/curvebench) is a small benchmark built to **watch** that process. Three ReLU MLPs — narrow, medium, and wide — learn the same 1D target side by side. Each run exports a training video: orange dots are the target, colored lines are the evolving prediction.

---

## The target

We start with a smooth Fourier sum on $x \in [-10, 10]$:

$$
y = \sin(x) + \tfrac{1}{2}\sin(2x) + \tfrac{1}{3}\sin(3x)
$$

Three networks, same depth, different width:

| Network | $d$                  | Parameters |
| ------- | -------------------- | ---------- |
| N1      | `[1, 10, 10, 1]`     | 141        |
| N2      | `[1, 100, 100, 1]`   | 10,401     |
| N3      | `[1, 1000, 1000, 1]` | 1,004,001  |

A ReLU net builds its prediction from **piecewise linear segments**. A wider hidden layer can allocate more segments, so in principle a wide net has more flexibility to trace a smooth oscillating curve — but only if the optimizer actually finds a good weight configuration.

All runs below use **1600 epochs** (full-batch gradient methods) or **10,000 steps** (evolutionary methods), seed 123, unless noted.

---

## Gradient descent

### Adam and AdamW

With full-batch Adam or AdamW, the wide net (N3, purple) pulls ahead early and tracks the harmonics well. The narrow net (N1) improves, but its prediction stays visibly smoother and less accurate — capacity is not the bottleneck here; **optimization** is.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include video.liquid path="assets/video/curvebench/fourier3-adam.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>
<div class="caption">
  Adam. N3 learns the curve fastest; N1 lags behind despite the same target and loss.
</div>

AdamW behaves almost identically on this problem — same trajectory, slightly different final N3 error.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include video.liquid path="assets/video/curvebench/fourier3-adamw.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>
<div class="caption">
  AdamW. Nearly the same learning dynamics as Adam on this 1D regression task.
</div>

### RMSprop

RMSprop is the surprise among adaptive gradient methods: on N1 it reaches a **tighter fit than Adam**, while N3 still lags. The clip shows N1 (blue) locking onto the harmonics faster than in the Adam runs.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include video.liquid path="assets/video/curvebench/fourier3-rmsprop.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>
<div class="caption">
  RMSprop. Strong on N1; N2 and N3 improve but do not match Adam on the wide nets.
</div>

### L-BFGS

L-BFGS with a full-batch line search makes slow, uneven progress. All three nets stay far from the target after 1600 epochs — second-order information does not rescue this non-convex curve-fitting problem at million-parameter scale.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include video.liquid path="assets/video/curvebench/fourier3-lbfgs.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>
<div class="caption">
  L-BFGS. Little visible improvement across N1–N3 within the epoch budget.
</div>

### SGD

At the default learning rate ($\mathrm{lr} = 10^{-2}$), SGD diverges on N3 — the purple prediction blows up while N1 and N2 keep fitting. Dropping $\mathrm{lr}$ to $10^{-3}$ stabilizes training: N3 finally converges and actually reaches the **best SGD fit** among the three widths.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include video.liquid path="assets/video/curvebench/fourier3-sgd.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>
<div class="caption">
  SGD + momentum, $\mathrm{lr} = 10^{-3}$. N3 no longer diverges; wide nets need a gentler step size.
</div>

---

## Differential Evolution: patience wins

The standout result in this benchmark is **Differential Evolution with $F = 0.5$** run for **10,000 steps**.

At 1600 steps, DE is still searching — the prediction curve wiggles and only partially matches the target. Given enough steps, the same population-based search on the **smallest** network (N1, 141 parameters) reaches the **best overall fit** in our experiments:

| Method                        | Budget           | N1 MSE     | N2 MSE    | N3 MSE    |
| ----------------------------- | ---------------- | ---------- | --------- | --------- |
| RMSprop                       | 1600 ep          | **0.013**  | 0.031     | 0.099     |
| Adam                          | 1600 ep          | 0.180      | **0.012** | **0.012** |
| AdamW                         | 1600 ep          | 0.179      | **0.012** | 0.015     |
| SGD ($\mathrm{lr}{=}10^{-3}$) | 1600 ep          | 0.558      | 0.310     | 0.067     |
| L-BFGS                        | 1600 ep          | 0.666      | 0.543     | 0.386     |
| DE ($F{=}0.5$)                | **10,000 steps** | **0.0066** | **0.036** | 0.237     |

DE does not need backpropagation. It proposes weight vectors, scores them by MSE, and iterates. On N1, that slow search eventually outperforms every gradient method. On N3, memory limits shrink the population and progress stalls — evolution does not scale to million-parameter nets as easily as Adam does.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include video.liquid path="assets/video/curvebench/fourier3-de-10k.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>
<div class="caption">
  DE with $F = 0.5$, 10,000 steps. Watch N1 (blue) converge to a tight fit — the best N1 result across all methods tested.
</div>

The lesson from the clip: **learning a curve is a process, not a snapshot**. DE on a tiny net, given enough steps, can trace harmonics that gradient methods on the same architecture leave rough.

---

## CMA-ES on a small search space

CMA-ES adapts a Gaussian over weights. Full CMA-ES is practical only for N1 (~141 parameters); the video below shows it steadily improving the N1 prediction over 1600 steps — competitive, but not matching DE's 10k-step N1 result.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include video.liquid path="assets/video/curvebench/fourier3-cmaes.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>
<div class="caption">
  CMA-ES on N1. Steady improvement, but DE at 10k steps still wins on MSE.
</div>

---

## A different question: learning $\sin(x)$ beyond the training interval

The Fourier experiment trains and tests on the same interval. A second experiment changes the question: **can the network extrapolate?**

Target: $y = \sin(x)$. Train on $x \in [-1, 1]$, evaluate on $x \in [-3, 3]$.

With **ReLU** activations, the network learns reasonable fits inside the training band, but outside it the prediction is piecewise-linear — it cannot rediscover periodicity it never saw. Swap the activation to **$\sin$**, and the same architecture can extend the wave into the out-of-distribution regions, because the building blocks themselves are periodic.

That experiment is less about optimizer choice and more about **what function class the network can express** before training even starts.

---

## What the videos show

Stepping back from the numbers:

1. **Width** changes how many linear pieces are available, but does not guarantee a better learned curve under every optimizer.
2. **Optimizer** changes the _trajectory_ — Adam stabilizes wide nets; SGD needs tuning; RMSprop excels on small nets; L-BFGS stalls; DE crawls but can win on small nets with enough steps.
3. **Activation** encodes prior knowledge about the shape of the world — critical when the test domain extends beyond training.

CurveBench exists to make those differences **visible**. The [code and configs](https://github.com/rkhosrowshahi/curvebench) are open source (MIT) if you want to run your own targets and watch your own curves learn.
