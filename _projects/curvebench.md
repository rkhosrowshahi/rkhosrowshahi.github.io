---
layout: page
title: CurveBench
description: Benchmark 1D curve learning across network width, activation, and optimizer.
img: assets/img/curvebench/preview.png
importance: 1
category: research
github: rkhosrowshahi/curvebench
---

[CurveBench](https://github.com/rkhosrowshahi/curvebench) compares how small MLPs learn 1D functions \(y = f(x)\) under different **widths**, **activations**, and **optimizers** — including gradient descent (PyTorch) and evolutionary algorithms (JAX / evosax).

Two experiments ship with the repo:

1. **Fourier width study** — three parallel nets (141 / 10k / 1M parameters) on a multi-harmonic target.
2. **\(\sin(x)\) OOD study** — train on \([-1,1]\), test on \([-3,3]\), compare ReLU vs \(\sin\) activation.

Every run exports training videos and final figures. Install with `pip install -e .` and run via the `curvebench` CLI or YAML configs.
