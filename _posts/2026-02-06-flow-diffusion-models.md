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
> This post covers the fundamentals of diffusion and flow matching models. The following notation and theoretical framework are primarily adapted from {% cite flowmodels2026 --file blog_refs %}.

# 1. Background
We begin with some mathematical definitions that help us to describe flow matching and diffusion models.

A **Trajectory** is a function of form
$$
X\colon [0, 1] \to \mathbb{R}^d
$$
that maps from time $t$ to a vector in $\mathbb{R}^d$, i.e., $t \mapsto X_t$.

A **Vector field** is defined as
$$
u\colon \mathbb{R}^d \times [0,1] \to \mathbb{R}^d\,,
$$
which maps from a location $x$ at time $t$ to a vector $u(x, t) \in \mathbb{R}^d$, indicating the velocity in the space, i.e., $(x, t) \mapsto u_t(x)$.

**Ordinary differential equation (ODE)** describes a condition on the trajectory. We want a trajectory $X_t$ to follow a line specified by the vector field $u_t(.)$. Given an initial condition, $X_0=x_0$, we define such a trajectory as the solution to the ODE equation

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

For a given condition $X_0=x_0$, one can recover the trajectory via $X_t = \psi_t(x_0)$. Under the mild condition that $u$ is continuously differentiable with a bounded derivative, then the ODE has a unique solution given by a flow $\psi_t$. As a result, $\psi_t$ is a *diffeomorphism* for all $t$ (i.e., continuously differentiable with a continuously differentiable inverse $\psi_t^{-1}$).

{% include figure.liquid 
    path="assets/img/ode.png" 
    title="Relationship between ODE, vector field, and flow" 
    caption="<b style='color: var(--global-theme-color)'>Figure 1:</b> Relationship between ODE, vector field, and flow" 
    class="img-fluid rounded w-75 mx-auto d-block" 
%}


**ODE solver:** One of the simplest ways to solve an ODE is the **Euler method**. Starting with an initial condition $X_0=x_0$, we update the solution as

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
We aim to convert a base distribution $p_\mathrm{init}$ into a complex distribution $p_\mathrm{data}$. This allows us to generate samples from $p_\mathrm{data}$.

## 2.1. Flow models
Consider an ODE to construct the transformation. A flow model is defined by an ODE

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
Instead of using an ODE that defines deterministic trajectories, diffusion models use a stochastic differential equation (SDE) to define the transformation from $p_\mathrm{init}$ to $p_\mathrm{data}$. A stochastic trajectory is a stochastic process $X(t)$ and it is given by

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
As described above, both flow and diffusion models can be parameterized by a neural network $u_t^\theta(.)$ representing a vector field. We can obtain samples from this model by solving the ODE

$$
X_0 \sim p_\mathrm{init}, \, dX_t = u_t^\theta(X_t)dt \,.
$$

Our goal is to make $X_1 \sim p_\mathrm{data}$. So, the question is how do we train this vector field? This section will describe flow matching {% cite lipman2022flow liu2022flow --file blog_refs %}, a technique to train $u_t^\theta$.

## 3.1. Conditional and marginal probability path
Note that only two endpoints at $t=0$ and $t=1$ have to satisfy our conditions $p_0=p_\mathrm{init}$ and $p_1=p_\mathrm{data}$. We have some freedom to design probability distributions $p_t$ in between $0 < t < 1$. Changing the conditional field changes how you travel, but not where you start or end. While the distributions $p_0$ and $p_1$ don't change, the pathway between them can change. In the following, we describe our design.

**Conditional probability path** is a set of distributions that gradually convert the initial distribution $p_\mathrm{init}$ into a Dirac delta distribution (i.e., a single point).

$$
p_0(.|z) = p_\mathrm{init}, \, p_1(.|z) = \delta_z \quad \text{for all} \, z \in \mathbb{R}^d
$$

Essentially, the conditional probability path can be considered as a trajectory in the space of distributions.

**Probability path** defines a set of distributions obtained by marginalizing the conditional probability path

$$
p_t(x) = \int p_t(x|z) p_{\mathrm{data}}(z) dz \,.
$$

It's straightforward to see that the marginal probability path $p_t$ interpolates between $p_\mathrm{init}$ and $p_\mathrm{data}$. 

$$
p_0(.|z) = p_\mathrm{init}, \, p_1(.|z) = p_\mathrm{data}\,. 
$$

Although $p_t$ is intractable, we can sample from it as the conditional probability path is often tractable by design.

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


> ##### Example: Flow matching for Gaussian conditional probability path
>
> Let consider the straight-path (Optimal Transport) case $\alpha_t = t$ and $\beta_t = 1 - t$, then we have
> 
> $$
> u^\mathrm{target}_t (x | z) = z - \epsilon \,.
> $$
> 
> Therefore, the objective function becomes
>
> $$
\mathcal{L}_\mathrm{CFM} (\theta) = \mathbb{E}_{t \sim \mathcal{U}(0,1), z\sim p_\mathrm{data}, \epsilon \sim \mathcal{N}(0, I_d)} [ \|u^\mathrm{target}_t(tz + (1 -t)\epsilon) - (z - \epsilon) \|^2] 
> $$
> 
{: .block-tip }

# 4. Score matching
Unlike flow models, diffusion models {% cite sohl2015deep song2020score --file blog_refs %} use SDEs to define the transformation from $p_\mathrm{init}$ to $p_\mathrm{data}$. This section describes diffusion models and how to train them using score matching.

## 4.1. Conditional and marginal score functions
**Conditional score function** is defined as $\nabla_x \log p_t(x \vert z)$ i.e., the gradient of the log-likelihood of the conditional density $p_t(x \vert z)$ with respect to $x$.

**Marginal score function** can be derived as

$$
\nabla_x \log p_t(x) = \frac{\nabla_x p_t(x)}{p_t(x)} = \int \nabla_x \log p_t(x | z) \frac{p_t(x|z)p_\mathrm{data}(z)}{p_t(x)} dz \,.
$$

The result looks very similar to the relationship between the conditional and marginal vector fields.

> ##### Example: Score function for Gaussian probability path
>
> For the Gaussian probability path $p_t(x \vert z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$, the conditional score function is defined as
> 
> $$
> \nabla_x \log p_t(x | z) = - \frac{x - \alpha_t z}{\beta_t^2}
> $$
>
> Note that the score function for the Gaussian path is a linear combination of $x$ and $z$. As a result, the conditional (marginal) vector field  can be recovered from the conditional (marginal) score function.
>
{: .block-tip }

For any diffusion coefficient $\sigma_t \ge 0$, one can construct an SDE as follows:

$$
\begin{align}
dX_t = \left [u^{\mathrm{target}}(X_t) + \frac{\sigma_t^2}{2} \nabla_x \log p_t (X_t) \right ] dt + \sigma_t dW_t \,. \label{eq:sde_diffusion}
\end{align}
$$

The marginal distribution $p_t$ will be the same as in flow models, i.e., $X_t \sim p_t$. Now the trajectories are stochastic due to the nature of the SDE's evolution. Although Equation \eqref{eq:sde_diffusion} holds for an arbitrary choice of $\sigma_t$, in practice one must carefully choose $\sigma_t$, which must be empirically determined.

> ##### Example: SDE for Gaussian probability path
>
> For the Gaussian probability path $p_t(x \vert z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$, we don't need to train the vector field $u^\theta_t$ and score function $s^\theta_t$ separately. We can simulate the SDE as
> 
> $$
> dX_t = \left[ \left(a_t + \frac{\sigma_t^2}{2} s_t^\theta(X_t) + b_t Xt \right) \right] dt + \sigma_t dW_t
> $$
>
> where 
>
> $$
> a_t = \left( \beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t\beta_t \right)\,, b_t = \frac{\dot{\alpha}_t}{\alpha_t}
> $$
>
{: .block-tip }


## 4.2. How to train score function?
Similarly to the marginal vector field, we can learn the score function $s^\theta_t(x)$ using the score matching loss and denoising score matching loss

$$
\begin{align*}
\mathcal{L}_\mathrm{SM}(\theta) &= \mathbb{E}_{t \sim \mathcal{U}(0,1), z\sim p_\mathrm{data}, x \sim p_t(x|z)} [ \|\nabla_x \log p_t(x) - s^\theta_t(x) \|^2]  \\
\mathcal{L}_\mathrm{CSM}(\theta) &= \mathbb{E}_{t \sim \mathcal{U}(0,1), z\sim p_\mathrm{data}, x \sim p_t(x|z)} [ \|\nabla_x \log p_t(x| z) - s^\theta_t(x) \|^2]
\end{align*}
$$

Although the target for score function is not the same, the score matching loss is equal to the denoising score matching loss up to a constant.

$$
\mathcal{L}_\mathrm{SM}(\theta) = \mathcal{L}_\mathrm{DSM}(\theta) + C \,.
$$

> ##### Example: Score matching for Gaussian probability path
>
> For the Gaussian probability path $p_t(x \vert z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$, the denoising score matching loss becomes
> 
> $$
> \mathcal{L}_\mathrm{CSM}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0,1), z\sim p_\mathrm{data}, x \sim p_t(x|z)} \left[ \left\|\frac{\epsilon}{\beta_t} + s^\theta_t(\alpha_t z + \beta_t \epsilon) \right \|^2 \right]
> $$
>
> To avoid numerical instability for $\beta_t\approx 0$, we can drop $1/\beta_t$ in the loss and reparameterize $s^\theta$ into a noise predictor network $\epsilon^\theta$ {% cite ho2020denoising --file blog_refs %}, i.e.,
>
> $$
>  \mathcal{L}_\mathrm{DDPM}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0,1), z\sim p_\mathrm{data}, x \sim p_t(x|z)} \left[ \left\|\epsilon - \epsilon^\theta_t(\alpha_t z + \beta_t \epsilon) \right \|^2 \right]
> $$
> 
> Note that one can recover the score as $s_t^\theta(x) = -\epsilon_t^\theta(x)/\beta_t$.
{: .block-tip }


<!-- {% cite ruby --file blog_refs %} -->


# References

{% bibliography --style apa --group_by none --file blog_refs --cited --template bib_plain %}