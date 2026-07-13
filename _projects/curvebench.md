---
layout: page
title: CurveBench
description: Watch how MLPs learn 1D curves — width, optimizer, and activation.
img: assets/img/curvebench/preview.png
importance: 1
category: research
---

[CurveBench](https://github.com/rkhosrowshahi/curvebench) trains small MLPs on 1D targets and exports **training videos** so you can watch prediction curves converge. Three-network stacked clips compare gradient descent (Adam, SGD, RMSprop, …) and evolutionary algorithms (DE, jDE, PSO, Sep-CMA-ES) on a multi-harmonic Fourier target, plus a $\sin(x)$ out-of-distribution activation study.

At GFLOP-matched compute, **jDE** wins on the million-parameter net, **DE** on the medium net, and **PSO** ($w = 0.5$) on the smallest. With unlimited budget, **DE at 10,000 steps** still achieves the tightest N1 fit (MSE 0.0066).
