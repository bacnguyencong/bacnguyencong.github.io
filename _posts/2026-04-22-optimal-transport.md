---
layout: post
toc:
  sidebar: left
title: Optimal transport and generative models
date: 2026-04-22 11:12:00-0400
description:  
tags: optimal-transport, generative-models
categories: deep-generative-modeling
related_posts: false
giscus_comments: true
enable_math: true
# citation: true
---
> This post covers the fundamentals of optimal transport and generative models. The following notation and theoretical framework are primarily adapted from {% cite tang2026foundations --file blog_refs %}.

# 1. Background

## 1.1. Stochastic process

Given a probability space $(\Omega, \mathcal{F}, \mathbb{P})$, a stochastic process is a collection of random variables $X = \\{X_t: t \in T \\}$ indexed by $t$. 

A  **Brownian motion** (Wiener process) is a stochastic process $B=\\{B_t \\}_{t \in T}$ with the following properties
- Starting value: $B_0 = 0$.
- Independent increments: for any set of times $0 \le t_1 < t_2 < \dots <t_n$, the increments $(B_{t_2} - B_{t_1}), (B_{t_3} - B_{t_2}), \dots, (B_{t_n} - B_{t_{n-1}})$ are independent.
- Gaussian increments: for any $t > s \ge 0$, we have $B_t - B_s \sim \mathcal{N}(0, (t - s)I)$.
- Continuity of paths: $B_t$ is continuous in $t$ but nowhere differentiable.


**Itô Integral** is a way of integrating with respect to a random process such as Brownian motion

$$
\int_{0}^T x(t) dB_t \,.
$$

The Itô integral can be viewed as the limit of discrete sums

$$
\int_{0}^T x(t) dB_t = \lim_{n\to \infty} \sum_{i=0}^{n-1} x(t_i) (B_{t_{i + 1}} - B_{t_i}) \,.
$$

The result of a define Itô integral is a single random variable (the final outcome at time T). For every realization of the noise $B$, the integral traces the cumulative total of random payoff over time.

**Itô's rule** is the chain of rule for random process. Assume that $X_t$ is an Itô process $dX_t = \mu_t dt + \sigma_t dB_t$ and consider any twice differentiable scalar function $f(X_t, t)$, then 

$$
d f = \frac{\partial f}{\partial t} dt + \sum_i \frac{\partial f}{\partial x_i} d x_i + \frac{1}{2} \sum_{i,j} \frac{\partial^2 f}{\partial x_i \partial x_j} d x_i d x_j \,.
$$




## 1.1. Optimal mass transport problem
Consider two probability distributions $\pi_0 \in \mathcal{P}(\mathcal{X})$ and $\pi_T \in \mathcal{P}(\mathcal{Y})$ and a cost function $c\colon \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$, which defines the cost of transporting one unit from $x \in \mathcal{X}$ to $y\in \mathcal{Y}$. The transport map $M\colon \mathcal{X} \to \mathcal{Y}$ generates $\pi_T$ as the pushforward of $\pi_0$. We define the space of such transport maps as

$$
\mathcal{T}(\pi_0, \pi_T) = \{ M\colon \mathcal{X} \to \mathcal{Y} \mid M_{\#} \pi_0 = \pi_T\} \,.
$$

The goal is to transport the source distribution $\pi_0$ into the target distribution $\pi_T$ as cheaply as possible.

**Monge's optimal mass transport (OMT) problem** aims to find the optimal map $M^*$ that minimizes

$$
\inf_{M\in \mathcal{T}(\pi_0, \pi_T)}  \int c(x, M(x)) d\pi_0(x) \,.
$$

The above problem is ill-posed. For example, when $\pi_0$ is concentrated at a single point and $\pi_T$ at multiple points, the transport maps $\mathcal{T}(\pi_0, \pi_T)$ is empty as the mass must be split to reach the target. 


**Kantorovich's OT** relaxes the problem as an optimization over the space of couplings $\pi_{0,T} \in \Pi(\pi_0, \pi_T)$, which is given by

$$
\Pi(\pi_0, \pi_T) = \left\{ \pi_{0,T} \in \mathcal{P}(\mathcal{X} \times \mathcal{Y}) \mid \int \pi_{0,T}(x,y) dy = \pi_0 \,, \int \pi_{0,T}(x,y) dx = \pi_T \right\} \,.
$$

The Kantorovich's OMT problem aims to find the optimal coupling $\pi^*_{0,T}$ that minimizes

$$
\inf_{\pi_{0,T}\in \Pi(\pi_0, \pi_T)}  \int c(x, y) d\pi_{0,T}(x, y) \,.
$$

The space of couplings always contains the product measure $\pi_0 \otimes \pi_T$. Therefore, it is never empty. Note that Kantorovich's formulation seeks for the best transport plan, while Monge's formulation seeks for the best transport plan.


## 1.2. Entropic optimal transport problem

Kantorovich's formulation remains a linear optimization problem in $\pi_{0,T}$, which is susceptible to non-unique and deterministic mappings. Consequently, a small change in the input can cause abrupt changes in the solution. To obtain a smoother solution, an entropy regularization is introduced, 

$$
\inf_{\pi_{0,T}\in \Pi(\pi_0, \pi_T)} \int c(x,y) d\pi_{0,T}(x,y) + \alpha \mathcal{D}_{\text{KL}}(\pi_{0,T} \| q) \,,
$$

where $\mathcal{D}_{\text{KL}}$ is the KL divergence between a transport plan and a fixed probability measure $q \in \mathcal{P}(\mathcal{X} \times \mathcal{Y})$. Introducing the entropy regularization has three advantages: **(i)** it ensures the solution is spread over the joint probability space $\mathcal{X} \times \mathcal{Y}$, rather than concentrating the mass on a small set; **(ii)** it leads to unique minimizer as the OT problem becomes strictly convex; **(iii)** it generalizes the OMT problem (i.e., when $\alpha \to 0$, the solution becomes OMT's solution and when $\alpha \to \infty$, it becomes the reference coupling $q$).

When choosing the reference coupling $q$ as the product of marginals, $q=\pi_0 \otimes \pi_T$, the KL divergence is related to the Shannon entropy as

$$
\mathcal{D}_{\text{KL}}(\pi_{0,T} \| \pi_0 \otimes \pi_T) = \mathcal{H}(\pi_{0,T}) + \text{Constant} \,,
$$

where  $\mathcal{H}(\pi_{0,T})$ denotes the entropy of $\pi_{0,T}$.


## 1.3. Static Schrödinger bridge problem

In this section, we define the static Schrödinger bridge (SB) problem, which is closely related to the entropic OT. Specifically, given two marginal distributions $\pi_0 \in \mathcal{P}(\mathcal{X})$ and $\pi_T \in \mathcal{P}(\mathcal{Y})$, let $\Pi(\pi_0, \pi_T)$ be the set of all couplings which satisfy the marginal conditions. Given a reference coupling $q\sim \pi_0 \otimes \pi_T$, the static SB problem is defined as 

$$
\pi^*_{0,T} = \underset{\pi_{0,T} \in \Pi(\pi_0, \pi_T)}{\arg \min} \mathcal{D}_\text{KL} (\pi_{0,T} \| q) \,.
$$


When the reference coupling takes the form:

$$
q(x,y) \propto e^\frac{-c(x,y)}{\alpha}(\pi_0 \otimes \pi_T)(x, y) \,,
$$

the static SB problem is equivalent to the entropic OT problem, i.e.,

$$
\pi_{0,T}^* = \underset{\pi_{0,T}\in \Pi(\pi_0, \pi_T)}{\arg\min} \int c(x,y) d\pi_{0,T}(x,y) + \alpha \mathcal{D}_{\text{KL}}(\pi_{0,T} \| q) \,.
$$

So, the static SB is an entropy-regularized OT problem. The solution is unique and has a closed-form structure. We can efficiently solve the SB problem with the Sinkhorn algorithm {% cite sinkhorn1964relationship --file blog_refs %}. Each iteration costs $O(n^2)$. 

Although the static SB identifies which starting point should map to which target point, it provides no information on the intermediate dynamics required to actually generate data. This motivates the dynamic SB problem. 

# 2. Dynamic Schrödinger bridge problem

For a generative model, we want to start with a sample from a source distribution (e.g., a simple prior) and follow a trajectory until we arrive at a sample from a target distribution (e.g., a complex data distribution) via intermediate states.

The dynamic SB problem extends the static problem from the space of couplings to the space of path measures. It provides the velocity field needed for step-by-step generation. The dynamic SB tracks the full-time evolution of the distribution. The path measures are defined over continuous-time trajectories. 

$$
\mathbb{P} \in C([0, T]; \mathbb{R}^d) \,.
$$

Let $\mathbb{Q}$ be some baseline stochastic process, which is characterized by a stochastic differential equation (SDE) of the form

$$
dX_t = f(X_t, t) dt + \sigma_t dB_t \,,
$$

where $B_t$ denotes the Brownian motion. We have freedom to choose $q_0$. For example, $q_0=p_0$, or $q_0=\mathcal{N}(0, I)$. The dynamic DB problem aims to find the optimal path measure that minimizes

$$
\mathbb{P}^* = \arg \min_\mathbb{P} \quad \{ \mathcal{D}_\mathrm{KL} (\mathbb{P} \| \mathbb{Q}): p_0 = \pi_0, p_T = \pi_T \} \,.
$$

Essentially, it searches amongst all path measures whose initial and terminal marginals match the source and target distributions, which one minimizes the KL divergence from the reference.

# 3. Solving dynamic Schrödinger bridge problem
Solving SB problem means finding a SDE that describes how to move samples from $p_0$ to $p_T$. The solution to this dynamic admits the following form:

$$
dX_t = [f(t, X_t) + g(t)^2 \nabla_x \log \Psi(t, X_t)] dt + g(t) dB_t \,,
$$

where $\Psi$ solves a forward-backward PDE system depending on the reference.


Another view of SB problem is through stochastic optimal control, which seeks an optimal control drift $v(x,t)\colon \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$ that reaches a state in the target distribution while deviating from the reference SDE minimally. We want to  a time-dependent drift $v(x, t)$, which perturbs the reference process to generate the target distribution, i.e.,

$$
d X_t = [f(X_t, t) + v_t(x_t)] dt + g(t) dB_t \,.
$$

The KL divergence between the 




# 4. Connection to previous frameworks
The common choice for $\mathbb{Q}$ is the standard Brownian motion.


# 5. Conclusions



# References

{% bibliography --style apa --group_by none --file blog_refs --cited --template bib_plain %}