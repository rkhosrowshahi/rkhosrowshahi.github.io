---
layout: post
title: "CurveBench: how neural networks learn curves"
date: 2026-07-13 00:00:00
description: Watching MLPs learn 1D curves. Width, optimizer, and activation change what the prediction looks like as training unfolds.
tags: research machine-learning optimization neural-networks
categories: research
thumbnail: assets/img/curvebench/preview.png
related_posts: true
---

How does a neural network learn a curve?

Not from a single loss number at the end, but from the shape of the prediction as training progresses. Does the fitted curve smooth out harmonic by harmonic? Does a wide net lock in quickly while a narrow one lags? Does SGD diverge while Adam stabilizes? Does a population-based search slowly crawl toward the target?

[CurveBench](https://github.com/rkhosrowshahi/curvebench) is a small benchmark built to watch that process. Three ReLU MLPs (narrow, medium, and wide) learn the same 1D target side by side. Each run exports a training video: orange dots are the target, colored lines are the evolving prediction.

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

A ReLU net builds its prediction from piecewise linear segments. A wider hidden layer can allocate more segments, so in principle a wide net has more flexibility to trace a smooth oscillating curve, but only if the optimizer actually finds a good weight configuration.

Gradient-descent runs use 2000 steps (GFLOP-matched budget per network), each under constant learning rate or cosine annealing to $\eta_{\min} = 0$. Evolutionary runs use the same GFLOP budget with a 6000 function-evaluation cap and fixed population size 100. Seed 123 throughout.

---

## Gradient descent

Every GD optimizer below is run twice: fixed step size for all 2000 steps (const), or cosine decay from the initial rate down to zero (cosine).

### Learning-rate schedule matters

On this problem the schedule interacts strongly with width. Constant LR tends to win on the wide nets (N2, N3) for Adam, AdamW, and RMSprop, because the step size stays large enough to keep fitting late in training. Cosine annealing helps on N1 for L-BFGS and RMSprop, where a long fine-tuning tail polishes the narrow net. SGD under either schedule lags the adaptive methods.

| Optimizer | Schedule | N1 MSE    | N2 MSE    | N3 MSE    |
| --------- | -------- | --------- | --------- | --------- |
| RMSprop   | const    | 0.017     | **0.004** | **0.003** |
| RMSprop   | cosine   | **0.012** | 0.006     | 0.004     |
| Adam      | const    | 0.173     | 0.009     | 0.013     |
| Adam      | cosine   | 0.246     | 0.011     | 0.073     |
| AdamW     | const    | 0.173     | 0.009     | 0.011     |
| AdamW     | cosine   | 0.246     | 0.011     | 0.077     |
| SGD       | const    | 0.506     | 0.290     | 0.063     |
| SGD       | cosine   | 0.595     | 0.324     | 0.186     |
| L-BFGS    | const    | 0.666     | 0.571     | 0.478     |
| L-BFGS    | cosine   | 0.232     | 0.307     | 0.315     |

Bold = lowest MSE in each column.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include video.liquid path="assets/video/curvebench/fourier3-adam-const-composed.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>
<div class="caption">
  Adam, constant LR, 2000 steps (N1 / N2 / N3 stacked). N3 tracks the target well. Cosine annealing on the same budget is worse on the wide nets.
</div>

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include video.liquid path="assets/video/curvebench/fourier3-adam-cosine-composed.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>
<div class="caption">
  Adam, cosine LR, 2000 steps. The learning rate decays too quickly for N2 and N3 to reach the same fit.
</div>

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include video.liquid path="assets/video/curvebench/fourier3-rmsprop-const-composed.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>
<div class="caption">
  RMSprop, constant LR. Best overall N2 and N3 MSE in the GFLOP-matched GD sweep.
</div>

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include video.liquid path="assets/video/curvebench/fourier3-rmsprop-cosine-composed.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>
<div class="caption">
  RMSprop, cosine LR. Slightly better on N1, slightly worse on N2 and N3 than const.
</div>

AdamW mirrors Adam under both schedules. SGD trails the adaptive methods, and const LR is better on every width. L-BFGS struggles with const LR, but cosine annealing improves the narrow net. On N2 and N3 it still lags RMSprop and Adam.

---

## Evolutionary algorithms at matched compute

Population-based methods do not use gradients. They propose weight vectors, score them by MSE, and iterate. We compared five EAs under the same GFLOP budget as 2000 GD steps (~6000 forward passes per network, population 100):

| Method                           | N1 MSE    | N2 MSE    | N3 MSE    |
| -------------------------------- | --------- | --------- | --------- |
| DE (F=0.5, elitist)              | 0.606     | **0.416** | 0.515     |
| jDE (adaptive F, CR, gauss init) | 0.485     | 0.532     | **0.465** |
| PSO (w=0.9, c1=c2=2)             | 0.479     | 0.599     | 0.625     |
| PSO (w=0.5)                      | **0.448** | 0.550     | 0.557     |
| Sep-CMA-ES                       | 0.478     | 0.563     | 1.555     |

No single EA wins on every width. DE is strongest on the medium net (N2). jDE adapts mutation and crossover rates per individual and takes N3. PSO with lower inertia ($w = 0.5$) beats $w = 0.9$ everywhere on this task, with less momentum and more responsiveness to personal and global bests. Sep-CMA-ES keeps up on small search spaces but collapses on the million-parameter net.

Each clip stacks N1 (top), N2 (middle), and N3 (bottom) so you can watch all three widths learn under the same algorithm.

### Differential Evolution

DE proposes trial vectors by mixing population members and accepts improvements. On N2 the prediction visibly tightens around the harmonics. On N3 progress is slower but steady.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include video.liquid path="assets/video/curvebench/fourier3-de-composed.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>
<div class="caption">
  DE ($F = 0.5$), GFLOP-fair budget. Best N2 MSE among EAs. N1 prediction improves but stays rougher than gradient methods.
</div>

### jDE

jDE carries its own mutation scale $F$ and crossover rate CR for each individual, resampling them occasionally and keeping successful pairs. Gaussian (Kaiming) initialization gives each swarm member a distinct starting function, which matters for diversity on high-dimensional weight vectors.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include video.liquid path="assets/video/curvebench/fourier3-jde-composed.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>
<div class="caption">
  jDE, GFLOP-fair budget. Best N3 MSE. Watch the bottom panel (purple) track the target while Sep-CMA-ES on N3 fails.
</div>

### Particle Swarm Optimization

We implemented pymoo-style PSO with fixed hyperparameters: per-dimension random coefficients and velocity clipping. Lower inertia ($w = 0.5$) consistently outperforms $w = 0.9$ here.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include video.liquid path="assets/video/curvebench/fourier3-pso-w05-composed.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>
<div class="caption">
  PSO ($w = 0.5$), GFLOP-fair budget. Best N1 MSE among EAs. Particles drift toward the global best without backprop.
</div>

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include video.liquid path="assets/video/curvebench/fourier3-pso-w09-composed.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>
<div class="caption">
  PSO ($w = 0.9$), same budget. Higher inertia means slower reaction and visibly worse fits on N2 and N3.
</div>

### Sep-CMA-ES

Diagonal CMA-ES adapts a per-weight step size. We initialize the search with $\sigma = 0.1$ as the initial standard deviation (`std_init`) of the Gaussian sampling distribution. It is competitive on N1 and N2, but the N3 panel barely moves because the search space is too large for this population budget.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include video.liquid path="assets/video/curvebench/fourier3-sep-cmaes-composed.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>
<div class="caption">
  Sep-CMA-ES ($\sigma = 0.1$), GFLOP-fair budget. Steady on N1/N2. N3 (bottom) stagnates.
</div>

---

## Small efficient neural networks with infinite compute

N1 has only 141 parameters, yet it is the hardest net to fit well under a tight EA budget and the easiest to polish when function evaluations are cheap. Raising the FE cap changes which algorithm wins and exposes how much population initialization and popsize matter.

### Population init: `center_gaussian`

We added three init modes. `center_gaussian` matches the original CurveBench recipe: one PyTorch seed network, a cloud of perturbations $\text{center} + \sigma \cdot \mathcal{N}(0, 1)$, and the exact center as one swarm member. `gaussian` gives every member an independent Kaiming draw plus a zero vector. `uniform` samples $U(-\sigma, \sigma)$ plus a zero vector.

On N1 at 1M FEs, DE with `center_gaussian` and popsize $\lceil 10\sqrt{D}\rceil = 119$ reaches MSE 0.0066. The same run with fixed popsize 100 stalls near 0.49. Independent Kaiming init (`gaussian`) behaves differently again. Init is not a detail here. It sets whether the search starts near one reasonable function or spreads across incompatible ones.

### Differential Evolution at raised budgets

DE uses $F = 0.5$, elitism, and `center_gaussian` with $\sigma = 0.1$. N1 and N2 use popsize $\lceil 10\sqrt{D}\rceil$. N3 is capped at 10k FEs and popsize 20 because a million-parameter forward pass per individual is too expensive at full budget.

| Network | FEs       | Pop   | Final MSE  |
| ------- | --------- | ----- | ---------- |
| N1      | 1,000,000 | 119   | **0.0066** |
| N2      | 1,000,000 | 1,020 | 0.188      |
| N3      | 10,000    | 20    | 0.401      |

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include video.liquid path="assets/video/curvebench/fourier3-de-unlimited-composed.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>
<div class="caption">
  DE with $F = 0.5$ at raised FE budgets (N1 / N2 / N3 stacked). N1 fits tightly. N2 improves but lags gradient methods. N3 moves slightly within 10k FEs on a million-parameter net.
</div>

N1 under loose compute beats every GFLOP-matched entry in the summary tables below, including RMSprop at 0.012. Patience and a well-centered cloud can trace harmonics that short-budget gradient steps leave rough on the same architecture.

### PSO catches up when init and velocity match the problem

The GFLOP-fair PSO sweep used independent Kaiming init (`gaussian`) and random initial velocities. Lower inertia ($w = 0.5$) won N1 at 0.448. Re-running PSO on N1 alone with `center_gaussian` at 1M FEs tells a different story.

We fixed $c_1 = c_2 = 2$, popsize 119, zero initial velocity, and swept $w$:

| Inertia $w$ | Init velocity | 1M FEs, N1 MSE |
| ----------- | ------------- | -------------- |
| 0.9         | random        | 0.353          |
| 0.9         | zero          | 0.339          |
| 0.5         | random        | 0.145          |
| 0.5         | zero          | **0.0095**     |
| 0.25        | zero          | 0.012          |
| 0.1         | zero          | 0.016          |

Random velocity adds noise that helps at $w = 0.9$ but hurts at $w = 0.5$. Zero velocity lets low-inertia particles converge toward personal and global bests without overshooting. At $w = 0.5$ with zero velocity, PSO reaches 0.0095, within striking distance of DE (0.0066) on the same net and budget.

Two practical lessons from the N1 unlimited-compute sweep:

1. Match popsize to dimension ($\approx 10\sqrt{D}$ for DE on small nets).
2. Match PSO inertia to initialization: `center_gaussian` pairs with low $w$ and zero velocity. High $w$ with a tight cloud around one seed barely moves.

Wide nets (N2, N3) still favor gradient methods even after raising the EA budget, but the smallest net is where evolutionary search closes the gap if you give it enough evaluations and the right starting geometry.

---

## A different question: learning $\sin(x)$ beyond the training interval

The Fourier experiment trains and tests on the same interval. A second experiment changes the question: can the network extrapolate?

Target: $y = \sin(x)$. Train on $x \in [-1, 1]$, evaluate on $x \in [-3, 3]$.

With ReLU activations, the network learns reasonable fits inside the training band, but outside it the prediction is piecewise-linear. It cannot rediscover periodicity it never saw. Swap the activation to $\sin$, and the same architecture can extend the wave into the out-of-distribution regions, because the building blocks themselves are periodic.

That experiment is less about optimizer choice and more about what function class the network can express before training even starts.

---

## Summary: all methods (GFLOP-matched budget)

Final train MSE on the Fourier target. Gradient methods: 2000 steps, reported separately for const and cosine LR. Evolutionary methods: same GFLOP budget, ~6000 function evaluations, population 100.

### Gradient descent

| Method           | N1 MSE    | N2 MSE    | N3 MSE    |
| ---------------- | --------- | --------- | --------- |
| RMSprop (const)  | 0.017     | **0.004** | **0.003** |
| RMSprop (cosine) | **0.012** | 0.006     | 0.004     |
| Adam (const)     | 0.173     | 0.009     | 0.013     |
| Adam (cosine)    | 0.246     | 0.011     | 0.073     |
| AdamW (const)    | 0.173     | 0.009     | 0.011     |
| AdamW (cosine)   | 0.246     | 0.011     | 0.077     |
| SGD (const)      | 0.506     | 0.290     | 0.063     |
| SGD (cosine)     | 0.595     | 0.324     | 0.186     |
| L-BFGS (const)   | 0.666     | 0.571     | 0.478     |
| L-BFGS (cosine)  | 0.232     | 0.307     | 0.315     |

### Evolutionary algorithms

| Method                           | N1 MSE    | N2 MSE    | N3 MSE    |
| -------------------------------- | --------- | --------- | --------- |
| DE (F=0.5, elitist)              | 0.606     | **0.416** | 0.515     |
| jDE (adaptive F, CR, gauss init) | 0.485     | 0.532     | **0.465** |
| PSO (w=0.9, c1=c2=2)             | 0.479     | 0.599     | 0.625     |
| PSO (w=0.5)                      | **0.448** | 0.550     | 0.557     |
| Sep-CMA-ES ($\sigma=0.1$)        | 0.478     | 0.563     | 1.555     |

Bold marks the lowest MSE in each column. Overall GFLOP-matched winners: RMSprop (const) on N2 and N3, RMSprop (cosine) on N1. Among EAs: PSO (w=0.5) on N1, DE on N2, jDE on N3.

With raised FE budgets on N1, DE reaches MSE 0.0066 and PSO ($w = 0.5$, zero velocity, `center_gaussian`) reaches 0.0095, both well below every GFLOP-fair entry above. N2 and N3 improve under loose EA budgets but remain above gradient methods at matched compute.

---

## What the videos show

Stepping back from the numbers:

1. Width changes how many linear pieces are available, but does not guarantee a better learned curve under every optimizer.
2. Optimizer changes the trajectory. Adam stabilizes wide nets with const LR. Cosine annealing hurts wide nets here. RMSprop excels overall. L-BFGS only becomes competitive on N1 with cosine decay. EAs crawl without gradients at matched compute, but on N1 with 1M FEs, DE and tuned PSO can approach gradient quality when init and popsize are right.
3. Activation encodes prior knowledge about the shape of the world. That matters when the test domain extends beyond training.

CurveBench exists to make those differences visible. The [code and configs](https://github.com/rkhosrowshahi/curvebench) are open source (MIT) if you want to run your own targets and watch your own curves learn.
