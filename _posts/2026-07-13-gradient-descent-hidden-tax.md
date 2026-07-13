---
layout: post
title: "Gradient Descent's Hidden Tax: Rethinking How We Compare It to Gradient-Free Methods"
date: 2026-07-13 00:00:00
description: Every GD step costs roughly 3× a gradient-free forward pass, so matching step counts isn't a fair benchmark. Match on FLOPs instead.
tags: research machine-learning optimization neural-networks benchmarking
categories: research
related_posts: true
---

## TL;DR

Every gradient descent (GD) step in a neural network costs roughly three times the compute of a gradient-free (GF) step. GD needs one forward pass plus two backward passes per layer (except the input layer, which only needs a weight gradient, not an input gradient, since nothing upstream needs it). GF methods only need a forward pass: there's no gradient to compute, so the backward graph never gets built on the GPU at all. If you're benchmarking GD against GF optimizers, giving both the same number of steps or epochs is not a fair fight. Give the GF method roughly three times the steps for the same compute budget, or better, match on FLOPs rather than steps.

---

## Where the 2x for backward comes from

Backpropagation at a single layer has to solve two separate problems, not one:

Suppose layer $l$ takes in an input vector $x_{l-1}$ (the output of the previous layer), applies weights $W_l$ to it, and passes the result forward. $L$ is the scalar loss computed at the very end of the network, the single number the whole training run is trying to shrink. With that setup:

1. **Gradient with respect to the weights**, $\dfrac{\partial L}{\partial W_l}$, so the optimizer knows how to update that layer.
2. **Gradient with respect to the layer's input**, $\dfrac{\partial L}{\partial x_{l-1}}$, so the error signal can keep flowing backward to earlier layers.

Each of those is roughly the same amount of matrix multiplication as the forward pass itself. So a full backward pass through a layer costs about twice what the forward pass through that same layer cost. Add the forward pass back in, and one GD step is forward (1x) plus backward (2x) equals 3x the compute of a bare forward pass.

The input layer is the one exception worth naming: since nothing feeds into it from an earlier layer, there's no need to compute the input gradient there, only the weight gradient. In small networks this shaves off a noticeable sliver of compute; in deep networks with dozens or hundreds of layers, it rounds to a footnote.

Putting numbers to it: for a layer $l$ with input dimension $d_{l-1}$ and output dimension $d_l$, the forward pass costs approximately

$$
F_l \approx 2\, d_{l-1} d_l
$$

FLOPs per example (the factor of 2 is the standard multiply-add convention). Backward through that same layer needs two gradients, each costing about as much as the forward pass:

$$
\frac{\partial L}{\partial W_l} \approx 2\, d_{l-1} d_l, \qquad \frac{\partial L}{\partial x_{l-1}} \approx 2\, d_{l-1} d_l
$$

so a full hidden layer costs $F_l$ forward plus $2F_l$ backward, or $3F_l$ total. Only the input layer ($l = 1$) drops the second term, costing $2F_1$ instead of $3F_1$, since there's no $x_0$ upstream to send a gradient to. Summed over a network of $L$ layers:

$$
C_{GD} = 3\sum_{l=2}^{L} F_l \;+\; 2F_1 \;\approx\; 3\sum_{l=1}^{L} F_l
$$

for any network deep enough that one layer's discount doesn't move the total.

This isn't a hand wavy estimate. It matches the standard FLOPs accounting used in neural scaling law research, most notably [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361): forward pass compute per token is approximately

$$
C_{fwd} \approx 2N
$$

FLOPs, where $N$ is the count of non-embedding parameters ($N \approx \sum_l d_{l-1} d_l$, which is exactly $\sum_l F_l$ up to the per-example constant). A full training step, forward and backward together, comes out to approximately

$$
C_{GD} = C_{fwd} + C_{bwd} \approx 2N + 4N = 6N
$$

meaning backward alone, $C_{bwd} \approx 4N$, is about twice forward. Three components, one forward and two backward equivalents, all the same order of magnitude.

---

## Why gradient-free methods dodge all of it

Gradient-free methods ([evolution strategies](https://arxiv.org/abs/1703.03864), genetic algorithms, random search, simulated annealing, zeroth-order optimization, CMA-ES, and similar) score a network by running it forward and reading off a loss or reward. There is no chain rule to unwind, so:

- No computation graph needs to be retained between layers.
- No backward kernels get launched on the GPU.
- No activation memory needs to be kept around for a later backward pass, which is a separate, often underrated win: it frees up memory for larger batches or larger populations in the same VRAM footprint.

In the same notation as above, a GF evaluation only pays the forward term:

$$
C_{GF} = C_{fwd} \approx 2N
$$

which puts the ratio between the two at

$$
\frac{C_{GD}}{C_{GF}} = \frac{6N}{2N} = 3
$$

exactly the multiplier in the TL;DR, derived rather than asserted. That last point about memory matters more than it first appears, and it's easy to miss if you only count FLOPs.

---

## Does the 3x hold across layer types and architectures?

The derivation above used a generic dense layer as the working example, since it's the cleanest case. Once other layer types enter the picture, the ratio holds cleanly for some and drifts for others.

### Matmul-dominated layers: convolutions and linear projections

Convolutional layers are structured matrix multiplies (via im2col or an equivalent formulation), so a conv layer with $C_{in}$ input channels, $C_{out}$ output channels, kernel size $k \times k$, and output spatial size $H \times W$ costs approximately

$$
F_{conv} \approx 2\, k^2 C_{in} C_{out} H W
$$

per example. Backward needs the same two gradients as a dense layer, a weight gradient and an input gradient, each costing about $F_{conv}$. The 3x rule carries over almost unchanged: convolutional networks (ResNets, U-Nets, and similar) follow the $3F_l$ per layer accounting about as cleanly as fully connected networks do. Transformer projection layers (QKV, output projection, MLP up and down projections) are literally dense layers, so the same accounting applies directly to them too.

### The extra term in attention

Attention adds a second cost source beyond the projections: the $QK^\top$ score matrix and the score-weighted sum over values, which scale with sequence length $L$ rather than with parameter count:

$$
C_{attn} \approx 2 L^2 d_{model}
$$

per layer. This term isn't linear in $N$, so it sits outside the clean $6N$ approximation entirely. When $L \ll d_{model}$, it's small next to the projection cost and the 3x figure holds well. At long context lengths it can dominate, and forward:backward for this term still tracks close to 1:2 for the same chain rule reason as everywhere else, so the ratio survives even in the regime where the flat $6N$ absolute figure stops being accurate.

### Where the ratio drifts below 2x: elementwise layers

Activations (ReLU, GELU, SiLU), dropout, and residual adds don't have a weight gradient and input gradient to split apart, there's just one gradient to compute: the upstream signal multiplied by the local derivative at that point. So a bare elementwise layer costs roughly one unit of compute forward and about one unit backward, not two, since there's no weight to update. These layers are cheap enough in absolute FLOPs that they don't move the network wide ratio much, but they're a genuine exception to "every layer is 3x."

### Where the ratio drifts above 2x: normalization layers

LayerNorm and BatchNorm are deceptively expensive on the backward side. Forward computes a mean and variance and normalizes with them. Backward needs gradients with respect to the scale and shift parameters, plus a gradient with respect to the input that depends on the full mean and variance gradient contribution across the batch or feature group, a cross-term that forward never had to compute at all. In absolute FLOPs these layers stay small next to matmuls, but their local forward:backward ratio isn't the clean 1:2 that dense layers get.

### Embedding layers: asymmetric by construction

Forward is a lookup (a gather), essentially free. Backward is a scatter-add into the embedding table's gradient, one update per token rather than a dense matmul. Cheap in both directions, but the assumption that forward and backward share the same underlying operation doesn't really apply here, since neither direction is a matmul to begin with.

### Recurrent layers: same ratio, different memory story

RNNs and LSTMs run backpropagation through time, which unrolls the same $3F_l$ per step accounting across every timestep in the sequence. The compute ratio per step doesn't change: forward is still one matmul, backward is still two. But every intermediate hidden state across the full sequence needs to survive until backward reaches it, which is the same kind of memory pressure that made activation checkpointing necessary for transformers, just applied per timestep instead of per layer.

### Mixture-of-Experts: the ratio holds, the base count shrinks

MoE layers route each token to a handful of active experts rather than running every expert densely. Forward only pays for the active experts plus the router. Backward routes gradients back through exactly those same active experts, so the one forward, two backward accounting still applies, just to a sparse subset of the network's total parameters rather than all of them. A GF evaluation of the same MoE would also only pay for the active experts, since gradient-free evaluation is still nothing but a forward pass, so the 3x comparison holds at the reduced, sparse FLOP count rather than the dense one.

### Bringing it back to the 3x rule

None of this breaks the headline result. The 3x still holds for the layers that dominate total network compute (dense projections and convolutions), and it's a reasonable estimate for attention and MoE too once you account for what's actually active. It just isn't literally true layer by layer: elementwise layers pull the ratio down, normalization layers push it up, and recurrent or checkpointed architectures add memory costs that a flat FLOP ratio never captures in the first place. The practical implication from the earlier profiling proposal still stands: measure the real ratio for your own architecture rather than assuming the idealized 3x applies uniformly across every layer type in the network.

---

## The benchmarking mistake this causes

A common setup in papers comparing GD to GF optimizers on neural networks: train network A with SGD/Adam for N epochs, train network B with an evolutionary strategy for N epochs or N generations, then compare final accuracy. This looks fair on paper because the step count matches. It isn't fair, because each GD step did roughly 3x the work of each GF step. The GF method was handed a third of the compute budget and then judged against it.

Making the correction explicit: at equal batch size, iso-compute between $S_{GD}$ gradient descent steps and $S_{GF}$ gradient-free steps requires

$$
S_{GD} \cdot C_{GD} = S_{GF} \cdot C_{GF} \;\;\Longrightarrow\;\; S_{GF} = S_{GD} \cdot \frac{C_{GD}}{C_{GF}} = 3\, S_{GD}
$$

so a fair comparison isn't "run both for $N$ steps," it's "run GD for $N$ steps and GF for $3N$ steps," or equivalently, plot both against $C_{GD} \cdot S_{GD}$ total FLOPs and read off whichever step count each method reaches at the same point on that axis.

---

## A few ideas for fixing the comparison

### 1. Report learning curves against FLOPs, not epochs

Plotting loss or accuracy against wall clock time or total FLOPs consumed, rather than against epoch count, removes the ambiguity entirely. An epoch axis hides the fact that the units on it aren't the same size for both methods. A compute axis doesn't.

### 2. Don't trust a flat 3x, measure your own ratio

The 3x figure is a clean idealization. Real ratios drift depending on mixed precision, activation checkpointing (which trades extra recomputation for less memory, changing the backward cost), fused kernels, and optimizer state overhead (Adam keeps two extra moment buffers per parameter, adding update cost that a GF method never pays either). Before adjusting step counts, it's worth profiling the actual forward to backward wall clock ratio for your own architecture and hardware, using something like [`torch.profiler`](https://docs.pytorch.org/docs/stable/profiler.html) or a FLOP counter such as [`fvcore`](https://github.com/facebookresearch/fvcore), rather than assuming the textbook number holds exactly.

### 3. Let GF spend its memory savings, don't just give it more steps

Since GF doesn't hold onto activations for backward, it can often run a larger population per generation in the same memory as one GD forward-backward pass. A more honest comparison protocol matches both compute and memory: instead of a flat step multiplier, size the GF population so that peak memory usage across a generation equals the peak memory of one GD step, then compare at matched wall clock time. This tends to be more favorable to GF than a naive 3x step correction alone, since it captures a resource GD is quietly consuming that a step count comparison ignores.

### 4. Try a two phase hybrid schedule as a benchmark baseline

Rather than treating GD and GF as competitors that must be scored on identical axes, a useful baseline is to spend an early compute budget on cheap GF exploration (many rough forward-only evaluations to locate a promising region of parameter space), then hand off to a smaller number of expensive GD steps for fine-grained convergence. Comparing this hybrid against pure GD and pure GF, all at matched total FLOPs, tends to be more informative than any single method's learning curve in isolation, and it turns the compute asymmetry into something to exploit rather than a nuisance to correct for.

### 5. Push for a standard "compute clock" in optimizer leaderboards

Most optimizer benchmarks report best accuracy at a fixed epoch or step count. A compute-normalized alternative, where every method's x-axis is total FLOPs (or measured GPU-seconds on a reference device) instead of steps, would make GD-versus-GF comparisons, and really any comparison between methods with different per-step costs, apples to apples by construction rather than by afterthought correction.

---

## The takeaway

The 3x rule is a useful default, not a universal constant. It's grounded in real accounting (one forward, two backward, per layer, minus the input layer's input gradient), and it lines up with the FLOPs ratios used in neural scaling law literature. But mixed precision, checkpointing, optimizer state, and memory headroom all nudge the true ratio around. The one thing worth taking away regardless of the exact multiplier: if a GD and a GF method are being compared on step count or epoch count alone, that comparison is measuring something other than what it claims to measure. Compute, not steps, is the currency that makes the comparison mean something.

---

## References

- Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361). arXiv:2001.08361.
- Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017). [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864). arXiv:1703.03864.
- PyTorch. [`torch.profiler`](https://docs.pytorch.org/docs/stable/profiler.html) documentation.
- Facebook Research. [`fvcore`](https://github.com/facebookresearch/fvcore): a lightweight library with FLOP counting utilities.
