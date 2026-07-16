---
layout: post
title: "Why Bigger Neural Networks Can Be Easier to Train"
date: 2026-07-15 00:00:00
description: Bigger networks do not only represent more functions. Overparameterization can make gradient descent's job easier and leave surprising traces of initialization behind.
tags: research machine-learning neural-networks optimization overparameterization neural-tangent-kernel
categories: research
related_posts: true
---

## TL;DR

Why do larger neural networks often train better than smaller ones, even on easy tasks that a small network should be able to represent?

The usual answer is **capacity**: more layers and neurons can express more complicated functions. That is true, but incomplete. Width also changes the optimization problem. An overparameterized network gives gradient descent more directions in which to reduce the loss, more redundant features to work with, and often a better conditioned local model. The small network may contain a good solution yet remain difficult for Adam or SGD to reach.

A striking illustration comes from Depue's experiment [[1]](#references): initialize a weight matrix with a grayscale image of a face or smiley, train the network, and the image remains visible beneath the learned updates. The network learned the task without erasing its starting point. Weird, right?

That looks like lazy training, but the image alone does not prove it. It proves initialization memory. To establish the behavior predicted by Neural Tangent Kernel (NTK) theory [[2]](#references), we should also measure parameter displacement, representation drift, and changes in the tangent kernel.

---

## The first answer: bigger networks have more capacity

Depth and width contribute in different ways.

### Depth builds compositions

A layer does not merely add another collection of features. It composes new features from those produced below it. In vision, the familiar intuition is edges to textures, textures to parts, and parts to objects. The useful point is not that every trained network follows this exact hierarchy, but that depth provides an efficient language for nested computation.

This is more than an intuition. Results on depth separation construct functions that a deep network can represent compactly but a shallower network can approximate only with exponentially greater width. Eldan and Shamir [[3]](#references) showed an exponential separation between particular networks with three layers and two layers. Telgarsky [[4]](#references) proved related separations for deeper ReLU networks.

These are existence results for specially constructed function families, not a theorem that every real task gains exponentially from every additional layer. But they establish an important boundary: optimization cannot recover a function that the architecture cannot represent efficiently in the first place.

### Width supplies parallel features

A wider layer can maintain more feature directions at the same level of abstraction. The universal approximation theorem [[5]](#references) says that a single hidden layer with a suitable activation and enough units can approximate any continuous function on a compact domain.

The phrase "enough units" does most of the work. The theorem guarantees existence, not a practical width, sample complexity, or training procedure. A shallow network might need an enormous number of units for a function that a deeper network represents economically.

It is tempting to summarize the combined expressivity of $L$ layers of width $N$ as "$N^L$." That is a useful picture of repeated recombination, but not a general formula for network capacity. Expressivity depends on the activation, architecture, parameter constraints, input dimension, and which measure of complexity we care about.

So capacity matters. But it does not explain one observation from my own experiments with curve fitting.

---

## The puzzle: easy curves, hard small networks

In [CurveBench]({% post_url 2026-07-13-curvebench %}), narrow, medium, and wide ReLU MLPs fit the same smooth target in one dimension:

$$
y = \sin(x) + \frac{1}{2}\sin(2x) + \frac{1}{3}\sin(3x).
$$

The narrow network has 141 parameters. That should be enough to produce a respectable approximation from linear pieces. Yet under the same budget of 2,000 steps, Adam reaches an MSE of 0.173 on the narrow net and 0.009 on the medium net. RMSprop shows the same direction: 0.017 versus 0.004.

The result does not by itself prove that the narrow network has sufficient capacity—the best representable error would need to be established independently—but it makes an explanation based only on capacity unsatisfying. On a simple curve, the optimizer appears to benefit from surplus parameters.

The better question is:

> Does width help because the network can represent a better answer, or because gradient descent can find an answer more easily?

For finite networks trained by practical optimizers, the answer can be both.

---

## Width changes the optimization problem

Suppose the network output on the training set is collected in a vector $f_\theta(X)$. Near the initialization $\theta_0$, its local linear approximation is

$$
f_\theta(X) \approx f_{\theta_0}(X) + J_{\theta_0}(X)(\theta - \theta_0),
$$

where $J_{\theta_0}$ is the Jacobian of outputs with respect to parameters.

Adding parameters adds columns to this Jacobian. Those extra directions can make it easier to produce a desired change in the outputs without requiring any individual parameter to move far. If the Jacobian has adequate rank and is well conditioned over the relevant directions, gradient descent can reduce the training residual quickly.

This gives a more precise version of "larger networks have more escape routes":

- **Redundancy:** if some ReLU units are inactive or several units learn nearly duplicate features, others can still carry useful gradients.
- **More descent directions:** a wide model can often change the same prediction in many directions through parameter space.
- **Smaller coordinated movement:** the required function change can be distributed across many parameters instead of forcing a few weights to make large, tightly coupled moves.
- **Large sets of solutions:** in overparameterized systems, solutions with zero loss may form manifolds of high dimension rather than isolated points.

Under suitable assumptions, theory makes parts of this story rigorous. Wide networks can satisfy conditions that guarantee convergence of gradient descent to zero training loss (see the Polyak–Łojasiewicz (PL)$^*$ analysis by Liu _et al._ [[6]](#references)). This does not mean that every wide network has a globally benign landscape, or that Adam is automatically optimal. It means overparameterization can change the geometry in ways that make gradient based optimization tractable.

> **Remark (PL and PL$^*$).** The PL condition says that near the set of minimizers, a small gradient already implies a small excess loss:
>
> $$
> \|\nabla L(\theta)\|^2 \;\ge\; 2\mu \bigl(L(\theta) - L^*\bigr)
> $$
>
> for some $\mu > 0$. It is weaker than convexity, but still strong enough for gradient descent to reach a global minimum of the training loss. PL$^*$ is a variant used for overparameterized models which states the inequality needs to hold on most of parameter space, not everywhere. That is the regime Liu, Zhu, and Belkin analyze for wide networks.

### Why Adam can expose the difference

Adam applies a separate diagonal rescaling to each parameter based on gradient moments. It can help when coordinates have different gradient scales, but it cannot generally remove correlations or poor conditioning between coordinates. A narrow network may need a few neurons to move in a highly coordinated way to bend the fitted curve. A diagonal preconditioner does not turn that coupled problem into an independent one.

In a wider network, the same output correction can be spread over many partially redundant units. Adam then has more usable directions and less responsibility concentrated in each neuron. This is a plausible mechanism for the CurveBench result, not a universal theorem about Adam.

---

## Lottery tickets support the optimization story<span style="font-size:0.5em;">(with a width caveat)</span>

The Lottery Ticket Hypothesis [[7]](#references) showed that dense networks can contain sparse subnetworks that, when reset to their original initialization and trained in isolation, match the dense network's accuracy in the studied settings.

This supports the idea that raw final capacity is not the whole benefit of the dense model. Initialization and trainability matter: the same sparse architecture with a new random initialization often performs worse.

But the result is easy to overstate. The winning ticket is normally discovered by training and pruning the dense network. Its existence does not prove that the other weights were irrelevant to the search, nor does it give a cheap method for finding the right sparse subnetwork before training. A large network may serve as both the hypothesis class and the search scaffold.

Double descent supplies another clue. Test error can fall again after a model passes the interpolation threshold, as documented by Belkin _et al._ [[8]](#references). Extra parameters do not automatically cause memorization to overwhelm generalization. The optimizer's implicit bias determines which of the many interpolating solutions is selected.

---

## A face hidden in the weights

Now for the visual experiment.

Instead of letting PyTorch randomly initialize a linear layer, reshape a grayscale image into its weight matrix. Train the network on an ordinary task with Adam. When the final matrix is displayed as an image, the original face or smiley can still be visible beneath the learned changes.

Depue demonstrated this with MNIST [[1]](#references), reporting persistence across initializations, learning rates, optimizers, and weight decay settings. The result is surprising because we often speak as if training rewrites the network. In a sufficiently redundant parameterization, it may only annotate it.

Write the final weights as

$$
W_{\text{final}} = W_{\text{init}} + \Delta W.
$$

If $\|\Delta W\|$ is small relative to $\|W_{\text{init}}\|$, the image remains visible. A large network can still change its output substantially because many small parameter updates combine through the Jacobian:

$$
\Delta f(X) \approx J_{\theta_0}(X)\Delta\theta.
$$

Small motion in parameter space does not imply small motion in function space.

This is closely related to the **lazy training** or **kernel** regime. In the Neural Tangent Kernel limit at infinite width, the tangent kernel remains constant and training behaves like optimization of the network's linearization around initialization. Jacot, Gabriel, and Hongler [[2]](#references) derived this limit, and Chizat, Oyallon, and Bach [[9]](#references) analyzed lazy training more broadly.

But there is an important distinction:

> A preserved image shows that a particular parameter projection stayed correlated with initialization. Lazy training requires the model itself to remain well described by its linearization.

The image might survive even while important features change elsewhere. Its contrast could also dominate a nontrivial update visually. Conversely, a visually damaged image does not prove useful feature learning because the weights might move through symmetries that barely change the function.

So the image is a beautiful diagnostic, but not a complete one.

---

## How to test what width is really buying

The capacity and optimization effects can be separated more cleanly with a small experimental grid.

### 1. Estimate the capacity floor

Train each small network with:

- many random restarts;
- a much longer schedule;
- Adam, SGD, and RMSprop;
- L-BFGS or another second order method, where the problem size makes it practical.

The best loss reached across this aggressive search is an empirical upper bound on the architecture's best representable loss. It is not proof of global optimality, but it is more informative than one Adam run.

If the small network eventually matches the wide network, the original gap was mainly optimization under the chosen budget. If a robust gap remains across optimizers and restarts, capacity becomes the stronger explanation.

### 2. Measure initialization memory

For each layer $l$, track relative parameter displacement:

$$
r_l(t) =
\frac{\|W_l(t)-W_l(0)\|_F}
{\|W_l(0)\|_F}.
$$

Also track correlation between the initial and current flattened matrices. The image supplies an intuitive visualization and these quantities supply the measurement.

### 3. Measure feature learning

Parameter movement is not enough. Compare hidden activations at initialization and after training using centered kernel alignment (CKA), feature covariance, or another measure of representation similarity. Large activation drift indicates that the network learned new internal features even if the image in its weight space remains recognizable.

### 4. Measure changes in the tangent kernel

For a manageable subset of examples, compute the empirical tangent kernel

$$
K_t(X,X) = J_{\theta_t}(X)J_{\theta_t}(X)^\top
$$

and report

$$
\frac{\|K_t-K_0\|_F}{\|K_0\|_F}.
$$

Small kernel drift, small representation drift, and small relative parameter movement together make a much stronger case for lazy training than the surviving image alone.

### 5. Sweep width under controlled parameterization

The relation between width and laziness depends on scaling and parameterization, not width alone. Chizat and colleagues [[9]](#references) emphasize the role of scaling, while Yang and Hu [[10]](#references) show that alternative parameterizations such as $\mu$ P can retain feature learning at infinite width.

That means a useful width sweep must specify how initialization scale and learning rate change with width. Otherwise "wider networks move less" may be an artifact of the chosen parameterization.

---

## What I now think larger networks are doing

The capacity explanation and the optimization explanation answer different questions:

- **Capacity asks:** does this architecture contain a good function?
- **Optimization asks:** can this training procedure find one from this initialization within the available compute?
- **Implicit bias asks:** among all the functions it could find, which one does the procedure prefer?

Depth can make some functions dramatically more economical to represent. Width can enlarge the set of available features. Overparameterization can also make the training Jacobian better behaved and create many nearby routes to low loss. Then gradient descent can fit the task through small distributed updates, sometimes leaving a human face visible in the very weights it trained.

The practical lesson is not simply "bigger models have more capacity." It is:

> Modern neural networks are often made large partly so that simple optimizers can train them.

The smallest network capable of representing a solution need not be the smallest network we know how to optimize. That gap between representability and reachability is where much of deep learning's apparent inefficiency may actually be doing useful work.

---

## References

1. Depue, W. (2026). [Tweet on initialization images remaining visible after training](https://x.com/willdepue/status/2076581570782056523?s=20).
2. Jacot, A., Gabriel, F., & Hongler, C. (2018). [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://proceedings.neurips.cc/paper/2018/file/5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper.pdf). _NeurIPS 2018_.
3. Eldan, R., & Shamir, O. (2016). [The Power of Depth for Feedforward Neural Networks](https://arxiv.org/abs/1512.03965). _COLT 2016_.
4. Telgarsky, M. (2016). [Benefits of Depth in Neural Networks](http://proceedings.mlr.press/v49/telgarsky16.pdf). _COLT 2016_.
5. Cybenko, G. (1989). [Approximation by Superpositions of a Sigmoidal Function](https://link.springer.com/article/10.1007/BF02551274). _Mathematics of Control, Signals, and Systems_, 2, 303–314.
6. Liu, C., Zhu, L., & Belkin, M. (2022). [Loss Landscapes and Optimization in Over-parameterized Non-linear Systems and Neural Networks](https://arxiv.org/abs/2003.00307). _Applied and Computational Harmonic Analysis_, 59, 85–116.
7. Frankle, J., & Carbin, M. (2019). [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635). _ICLR 2019_.
8. Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). [Reconciling modern machine-learning practice and the classical bias–variance trade-off](https://www.pnas.org/doi/10.1073/pnas.1903070116). _PNAS_, 116(32), 15849–15854.
9. Chizat, L., Oyallon, E., & Bach, F. (2019). [On Lazy Training in Differentiable Programming](https://papers.nips.cc/paper_files/paper/2019/file/ae614c557843b1df326cb29c57225459-Paper.pdf). _NeurIPS 2019_.
10. Yang, G., & Hu, E. J. (2021). [Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks](https://proceedings.mlr.press/v139/yang21c/yang21c.pdf). _ICML 2021_.
