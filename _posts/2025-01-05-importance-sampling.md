---
layout: post
comments: true
title: "Importance sampling"
date: 2025-01-05 00:00:00
tags: deep-learning
---

> Importance sampling is a effective method used to reduce variance. In this post, I will cover key concepts and explore its applications, particularly in diffusion models.
<!--more-->

## Problem
We want to calculate the following integral
$$
F = \int_{a}^b f(x) dx
$$
Due to the complexity of $f(x)$, this integral may not have an analytical form,
 and one may need to resort to numerical methods. An option is to use Monte Carlo sampling to estimate the integral.
 The integral can be interpreted as an expectation, that is,
$$
F = (b - a) \mathbb{E}_{x\sim p(x)}\left[f(x)\right] \approx \frac{1}{N} \sum_{x_i} f(x_i) = F_p \,,
$$
where $p(x)$ is uniform distribution in $[a,b]$. Because only $N$ samples are used to estimate the expectation,
the computational cost is significantly reduced. On the other hand, we also introduce additional variance,
$$
\begin{aligned}
\mathrm{Var}[F_p] &= \mathrm{Var}\left[ (b - a) \frac{1}{N} \sum_{x_i} f(x_i)  \right] \\
                  &= \frac{(b - a)^2}{N} \mathrm{Var}[f(x)]\,.
\end{aligned}
$$
As shown in the formula, variance can be reduced by increasing the number of samples.

## Variance Reduction with Importance sampling

Let $q(x)$ denote a proposal distribution with support on the interval $[a,b]$. The integral can be rewritten as 
$$
F = (b - a) \mathbb{E}_{x\sim q(x)}\left[\frac{f(x)}{q(x)}\right] \approx (b - a) \frac{1}{N} \sum_{x_i} \frac{f(x_i)}{q(x_i)} = F_q \,.
$$
It's straigforward to show that $F_q$ is an unbiased estimator, i.e., $\mathbb{E}[F_q] = F$. However, what about the vaiance of $F_q$?
Does introducing the proposal distribution help to reduce the vaiance?

Now let's derive the variances of $F_q$.
$$
\begin{aligned}
\mathrm{Var}[F_q] &= \mathrm{Var}\left[ (b - a) \frac{1}{N} \sum_{x_i} \frac{f(x_i)}{q(x_i)}  \right] \\
                  &= \frac{(b - a)^2}{N} \mathrm{Var} \left[ \frac{f(x)}{q(x)}  \right] \\
                  &= \frac{(b - a)^2}{N} \left[ \int_a^b q(x) \frac{f^2(x)}{q^2(x)} dx - \left(\int_a^b q(x) \frac{f(x)}{q(x)} dx\right)^2  \right] \\
                  &= \frac{(b - a)^2}{N} \left[ \int_a^b \frac{f^2(x)}{q(x)} dx - F^2  \right] \\
\end{aligned}
$$

There are multiple choices for $q(x)$, each leading to a different variance. 
Our goal is to identify the proposal distribution $q(x)$ that minimizes the variance.

Taking the derivative of $\mathrm{Var}[F_q]$ w.r.t. $q(x)$ and set it to zero. 
This leads to $q(x) \propto |f(x)|$. In other words, $q(x)$ is given by
$$
q(x) = \frac{|f(x)|}{\int_a^b |f(x)| dx} = \frac{|f(x)|}{\overline{F}} \,,
$$
where $\overline{F}=\int_a^b |f(x)| dx$ is normalizing constant. The optimal sampling distribution yields
$$
\begin{aligned}
\mathrm{Var}[F_q] &= \frac{(b - a)^2}{N}(\overline{F}^2 - F^2) \\
\end{aligned}
$$
If $f(x) \ge 0$, then the variance of $F_q$ becomes zero,
meaning that with just a single sample, we can exactly estimate the integral. 

## Techniques to Learn Proposal Distribution
Computing the optimal $q(x)$ is impractical because $\overline{F}$ is unknown, 
and estimating it would require the same computational efforts as estimating $F$ itself.
Although the optimal proposal distribution is not realizable, it gives us a useful strategy. 
Next, we discuss some popular techniques.

### Neural Importance Sampling

Müller et al. [1] introduced e deep neural network for generating samples in Monte
Carlo integration. 

Assumming that  $f(x) \ge 0$, let $q^{*}(x) = f(x)/F$ denote the optimal sampling distribution.
We define $q(x; \theta)$ as the proposal distribution parameterized by $\theta$.

To learn $q(x; \theta)$, we minimize the  Kullback-Leibler Divergence
$$
\begin{aligned}
D_{\mathrm{KL}}(q^{*}(x) || q(x; \theta)) &= \int q^{*}(x) \log \frac{q^{*}(x)}{q(x; \theta)} dx \\
&= \int q^{*}(x)\log q^{*}(x) dx - \int q^{*}(x)\log q(x; \theta) dx \\
&= \mathrm{constant} - \int q^{*}(x)\log q(x; \theta) dx
\end{aligned}
$$

To miminize the KL divergence with gradient descent, we need the gradient w.r.t. the trainable paramters $\theta$
$$
\begin{aligned}
\nabla_\theta D_{\mathrm{KL}}(q^{*}(x) || q(x; \theta)) &= - \nabla_\theta\int q^{*}(x)\log q(x; \theta) dx \\
&= -\int q^{*}(x) \nabla_\theta \log q(x; \theta) dx  \\
&= -\int q(x; \theta) \frac{q^{*}(x)}{q(x; \theta)} \nabla_\theta \log q(x; \theta) dx  \\
&= \mathbb{E}_{x\sim q(x; \theta)}\left[ - \frac{q^{*}(x)}{q(x; \theta)} \nabla_\theta \log q(x; \theta) \right] \,.
\end{aligned}
$$
The samples are drawn from the learned generative model. 
The gradient corresponds to the gradient of the negative log likelihood of $q(x; \theta)$ with weighting terms given by $q^{*}(x)/q(x; \theta) = F f(x)/q(x; \theta)$.
Since $F$ is unknown, the gradient can be estimated up to the global scale $F$. This is not an issue as common stochastic gradient descent techniques such as Adam
scales the step size by the square root of the gradient variance. Thus, the scale $F$ is cancelled, giving us the correct gradient during optimization.

Finally, the maximum log-likelihood training of $\log q(x; \theta)$ can be done with normalizing flow techniques.

### Adaptive Importance Sampling


<a name="references"></a>
### References
 
[1]: Müller, Thomas, et al. "Neural importance sampling." ACM Transactions on Graphics (ToG) 38.5 (2019): 1-19.