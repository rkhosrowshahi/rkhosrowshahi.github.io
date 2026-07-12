---
layout: page
title: CurveBench
description: Watch how MLPs learn 1D curves — width, optimizer, and activation.
img: assets/img/curvebench/preview.png
importance: 1
category: research
github: rkhosrowshahi/curvebench
---

[CurveBench](https://github.com/rkhosrowshahi/curvebench) trains small MLPs on 1D targets and exports **training videos** so you can watch prediction curves converge. Two experiments: a multi-harmonic Fourier target (width + optimizer comparison) and a $\sin(x)$ out-of-distribution study (activation comparison).

Best result so far: **Differential Evolution** with $F = 0.5$ at 10,000 steps — lowest MSE on the smallest network (N1).
