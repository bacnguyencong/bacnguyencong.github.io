---
layout: post
toc:
  sidebar: left
title: Introduction to diffusion and flow models
date: 2026-02-06 11:12:00-0400
description:  
tags: diffusion flow
categories: deep-generative-modeling
related_posts: false
giscus_comments: true
enable_math: true
# citation: true
---
> In this post, I will introduce fundamentals of diffusion and flow matching models.

# 1. Background
We begin with some mathematical definitions that help us to describe flow matching and diffusion models.

**Trajectory** is a function of form
$$
X\colon [0, 1] \to \mathbb{R}^d
$$
that maps from time $t$ to a vector in $\mathbb{R}^d$, i.e., $t \mapsto X_t$.

**Vector field** is defined as
$$
u\colon \mathbb{R}^d \times [0,1] \to \mathbb{R}^d\,,
$$
which maps from a location $x$ at time $t$ to a vector $u(x, t) \in \mathbb{R}^d$, indicating the velocity in the space, i.e., $(x, t) \mapsto u_t(x)$.

**Ordinary differential equation (ODE)** describes a condition on the trajectory. We want a trajectory $X_t$ follows a line specified by the vector field $u_t(.)$. Given an initial condition, $X_0=x_0$, we define such a trajectory as the solution to the ODE equation

$$
\begin{align*}
\frac{dX_t}{dt} &= u_t(X_t) \\
X_0 &= x_0
\end{align*}
$$

**Flow** is a set of trajectories with different initial conditions. In other words, if we start at $X_0 = x_0$ and $t=0$, where we are at time $t$.

$$
\begin{align*}
\psi \colon \mathbb{R}^d \times [0, 1] &\to \mathbb{R}^d \\
\psi_0(x_0) &= x_0 \\
\frac{d}{dt} \psi_t(x_0) &= u_t(\psi_t(x_0))
\end{align*}
$$

For a given condition $X_0=x_0$, one can recover the trajectory via $X_t = \psi_t(x_0)$. Under a mild condition $u$ is continuously differentiable with a bounded derivative, then the ODE has a unique solution given by a flow $\psi_t$. As a result, $\psi_t$ is a *diffeomorphism* for all $t$ (i.e., continuously differentiable with a continuously differentiable inverse $\psi_t^{-1}$).

{% include figure.liquid 
    path="assets/img/ode.png" 
    title="Relationship between ODE, vector field, and flow" 
    caption="<b style='color: var(--global-theme-color)'>Figure 1:</b> Relationship between ODE, vector field, and flow" 
    class="img-fluid rounded w-75 mx-auto d-block" 
%}


**ODE solver:** One of the simplest way to solve an ODE is the **Euler method**. Starting with an initial condition $X_0=x_0$, we update the solution as

$$
X_{t + h} = X_t + h \,. u_t(X_t) \,,
$$

where $h=1/N$ is the step size.

Another second order method is **Heun's method**, which defines the update via

$$
\begin{align*}
X'_{t + h} &= X_t + h \,. u_t(X_t) \\
X_{t + h} &= X_t + \frac{h}{2} \,.  \Big( u_t(X_t) + u_{t+h}(X'_{t + h})\Big)
\end{align*}
$$

Intuitively, it takes the first solution guess, then corrects the solution using the updated guess.

# 2. Flow and diffusion models
We aim to convert a base distribution $p_\mathrm{init}$ into a complex distrution $p_\mathrm{data}$. This allows us to generate samples from $p_\mathrm{data}$.

## 2.1. Flow models
Let consider an ODE to construct the transformation. A flow model is defined by an ODE

$$
\begin{aligned}
X_0  &\sim p_\mathrm{init} \\
\frac{d}{dt} X_t &= u_t^\theta (X_t)
\end{aligned}
$$

where the vector field is parameterized by a neural network $u^\theta$. The goal is to learn this vector field such that

$$
\begin{aligned}
X_1 \sim  p_\mathrm{data} \Leftrightarrow \psi_1^\theta(X_0) \sim p_\mathrm{data}\,,
\end{aligned}
$$

where $\psi^\theta$ is the flow derived from vector field $u^\theta$. Once the flow model is trained, we can generate samples from $p_\mathrm{data}$ using any ODE solver.



## 2.2. Diffusion models
Instead of using an ODE that defines deterministic trajectories, diffusion models use stochastic differential equation (SDE) to define the transformation from $p_\mathrm{init}$ to $p_\mathrm{data}$. A stochastic trajectory is a stochastic process $X(t)$ and it's given by

$$
\begin{align*}
X_0 &\sim p_\mathrm{init}  \\
dX_t &= u_t^\theta(X_t) dt + \sigma(t) dW 
\end{align*}
$$

where $W$ is a Brownian motion, the function $\sigma\colon \mathbb{R} \to \mathbb{R}$ is called *diffusion coefficient* and the function $u_t^\theta\colon \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$ is called *drift coefficient*. Every ODE is also an SDE with $\sigma(t)=0$. Note that $\sigma(t)$ is fixed and $X_t$ is a random variable for all $0 \le t \le 1$.

A SDE can be solved using a simple method such as Euler-Maruyama, which updates

$$
X_{t + h} = X_t + h\,. u_t^\theta (X_t) + \sqrt{h} \sigma(t) \epsilon
$$

The above update rule looks very similar to the Euler method except the last element, where we add a Gaussian noise $\epsilon$ scaled by $\sqrt{h} \sigma(t)$.

# 3. Flow matching
As described above, both flow and diffusion models can be parameterized by a neural network $u_t^\theta(.)$ representing a vector filed. We can obtain samples from this model by solving the ODE

$$
X_0 \sim p_\mathrm{init}, \, dX_t = u_t^\theta(X_t)dt \,.
$$

Our goal is to make $X_1 \sim p_\mathrm{data}$. So, the question is how do we train this vector field? This section will describe flow matching{% cite lipman2022flow liu2022flow --file blog_refs %}, a technique to train $u_t^\theta$.

## 3.1. Conditional and marginal probability path
Noe that only two endpoints at $t=0$ and $t=1$ have to satisfy our conditions $p_0=p_\mathrm{init}$ and $p_1=p_\mathrm{data}$. We have some freedom to design probability distributions $p_t$ in between $0 < t < 1$. In the following, we describe our design.

**Conditional probability path** is a set of distributions that gradually convert the initial distribution $p_\mathrm{init}$ into a Dirac delta distribution (i.e., a single point).

$$
p_0(.|z) = p_\mathrm{init}, \, p_1(.|z) = \delta_z \quad \text{for all} \, z \in \mathbb{R}^d
$$

Essentially, conditional probability path can be considered as a trajectory in the space of distributions.

**Probability path** defines a set of distributions obtained by marginalizing the conditional probability path

$$
p_t(x) = \int p_t(x|z) p_{\mathrm{data}}(z) dz \,.
$$

It's straightforward to see that the marginal probability path $p_t$ interpolates between $p_\mathrm{init}$ and $p_\mathrm{data}$. 

$$
p_0(.|z) = p_\mathrm{init}, \, p_1(.|z) = p_\mathrm{data}\,. 
$$

Although $p_t$ is intractable, but we can sample from it as the conditional probability path is often tractable by design.

> ##### Example: Gaussian conditional probability path
>
> Let $\alpha_t, \beta_t$ denote two continuously differentiable, monotonic functions with $\alpha_0=\beta_1=0$ and $\alpha_1=\beta_0=1$. The Gaussian conditional probability path is defined as
> 
> $$
> p_t (.|z) = \mathcal{N}(\alpha_t z, \beta^2 I_d)
> $$
>
> At the endpoints, we have
> 
> $$
> p_0(.|z) =  \mathcal{N}(0, I_d) \quad \text{and} \quad p_1(.|z) = \mathcal{N}(z, 0) = \delta_z
> $$
> 
> Sampling from the marginal $p_t$ can be expressed as
>
> $$
> z\sim p_\mathrm{data}, \epsilon \sim p_\mathrm{init} \rightarrow x = \alpha_t z + \beta_t \epsilon \,.
> $$
{: .block-tip }



## 3.2. Conditional and marginal vector field

So far, we have designed a marginal probability path $p_t$ that the points $X_t$ along a trajectory should have. The remaining task is to find a vector field such that trajectories $X_t$ follow the probability path.

**Conditional vector field** is a vector field $u^\mathrm{target}_t(.\vert z)$ such that the corresponding ODE yields the conditional probability path $$p_t(.\vert z)$$, i.e.,

$$
\frac{d}{dt} X_t = u^\mathrm{target}_t(.| z) \,, X_0 \sim p_\mathrm{init} \rightarrow X_t \sim p_t(.|z)
$$

The conditional vector field can be derived analytically by hand.

> ##### Example: Conditional Gaussian vector field
>
> Let $p(. \vert z)=\mathcal{N}(\alpha_t z, \beta_t^2 I_d)$, the conditional Gaussian vector field is given by as
> 
> $$
> u^\mathrm{target}_t (x | z) = \left(  \dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t \right) z + \frac{\dot{\beta}_t}{\beta_t} x
> $$
> 
{: .block-tip }

Given a conditional vector field $u^\mathrm{target}_t(.\vert z)$, the **marginal vector field** $u^\mathrm{target}_t(x)$ can be expressed as

$$
\begin{equation}
u^\mathrm{target}_t(x) = \int u^\mathrm{target}_t(x \vert z) \frac{p_t(x|z) p_\mathrm{data}(z)}{p_t (x)} dz \label{eq:marginal_vector_field}
\end{equation}
$$

The marginal vector field follows the marginal probability path

$$
X_0 \sim p_\mathrm{init}, \, \frac{d}{dt} X_t = u^\mathrm{target}_t(X_t) \rightarrow X_t \sim p_t(.) \quad (0 \le t \le 1)
$$



## 3.3. How to train the marginal vector field?

Equation $\eqref{eq:marginal_vector_field}$ provides an intuitive way to learn the marginal vector field $u^\mathrm{target}_t(X_t)$, e.g., using mean-squared error

$$
\mathcal{L}_\mathrm{FM} (\theta) = \mathbb{E}_{t \sim \mathcal{U}(0,1), z\sim p_\mathrm{data}, x \sim p_t(x|z)} [ \|u^\mathrm{target}_t(x) - u^\theta_t(x) \|^2] 
$$

We first draw a random time $t\in [0,1]$, then a data point $z$ from our data set. Finally, we sample $x$ from the conditional distribution and compute $u_t^\theta$. While the formula to compute $u^\mathrm{target}_t(x)$ is known, we cannot compute it efficiently. Instead, we use the conditional flow matching to learn the vector field as

$$
\mathcal{L}_\mathrm{CFM} (\theta) = \mathbb{E}_{t \sim \mathcal{U}(0,1), z\sim p_\mathrm{data}, x \sim p_t(x|z)} [ \|u^\mathrm{target}_t(x| z) - u^\theta_t(x) \|^2] 
$$

As the conditional vector field is tractable, one can minimize the above loss easily. Although the target vector field is not the same, the marginal flow matching loss equals the conditional flow matching up to a constant, i.e.,

$$
\mathcal{L}_\mathrm{FM}(\theta) = \mathcal{L}_\mathrm{CFM}(\theta) + C
$$

Therefore, optimizing the conditional flow matching loss is equivalent to minimizing the flow matching loss.

<!-- {% cite ruby --file blog_refs %} -->


# References

{% bibliography --style apa --group_by none --file blog_refs --cited --template bib_plain %}