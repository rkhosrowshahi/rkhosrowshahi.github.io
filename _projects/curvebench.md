---
layout: page
title: CurveBench
description: Watch how MLPs learn 1D curves. Width, optimizer, and activation.
img: assets/img/curvebench/preview.png
importance: 1
category: research
---

[CurveBench](https://github.com/rkhosrowshahi/curvebench) trains small MLPs on 1D targets and exports training videos so you can watch prediction curves converge. Three-network stacked clips compare gradient descent (Adam, SGD, RMSprop, and others) and evolutionary algorithms (DE, jDE, PSO, Sep-CMA-ES) on a multi-harmonic Fourier target, plus a $\sin(x)$ out-of-distribution activation study.

At GFLOP-matched compute (2000 GD steps), RMSprop leads: const LR on N2/N3, cosine on N1. Among EAs, jDE wins on the million-parameter net, DE on the medium net, and PSO ($w = 0.5$) on the smallest.
