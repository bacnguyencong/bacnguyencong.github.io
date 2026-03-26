---
layout: post
toc:
  sidebar: left
title: Drifting models
date: 2026-03-06 11:12:00-0400
description:  
tags: drifting
categories: deep-generative-modeling
related_posts: false
giscus_comments: true
enable_math: true
# citation: true
---
> This post explores some key insights behind drifting models {% cite deng2026generative --file blog_refs %}, a framework that enables high-fidelity one-step generation. By shifting the iterative process to the training phase, this approach has achieved state-of-the-art results on ImageNet.


# 1. Training objective
Let $f_\theta\colon \mathbb{R}^C \to  \mathbb{R}^D$ be a function (parameterized by $\theta$) that maps a noise $\epsilon \sim p_\epsilon$ in $\mathbb{R}^C$ to an output $x=f_\theta(\epsilon) \sim q_\theta$ in $\mathbb{R}^D$. The distribution of outputs $q_\theta$ is also referred to as the *pushforward* distribution under $f_\theta$, i.e.,

$$
q_\theta=(f_\theta)\#p_\epsilon \,.
$$

In other words, $f_\theta$ transforms a base distribution $p_\epsilon$ into the pushforward distribution $q_\theta$. Our objective is to find $f_\theta$ such that $q_\theta \approx p$ under some statistical divergence, where $p$ denotes the data distribution. 

Let $\Delta_{p,q_\theta}\colon \mathbb{R}^D \to \mathbb{R}^D$ denote a **drifting field** that describes how the output sample $x\sim q_\theta$ should move to match the data distribution $p$. It can be seen as a guidance force for the model output during training. It measures the force required to push samples from $q_\theta$ to $p$. The drifting field must satisfy an equilibrium condition, i.e.,

$$
\begin{align}
\Delta_{p, q_\theta}(x) = 0 \, \text{ if and only if } \, q_\theta = p\,. \label{eq:equilibrium}
\end{align}
$$

The equilibrium condition is critical for ensuring that the training process converges accurately and stays stable. It yields the following two properties:
- **Correctness** ($\Delta_{p, q_\theta}=0 \rightarrow p=q_\theta$): This direction guarantees that if the drifting field vanishes, the generator has successfully matched the pushforward distribution $q_\theta$ to the target data distribution $p$.
- **Stability** ($p=q_\theta \rightarrow \Delta_{p, q_\theta}=0$):  This direction ensures that once the distributions match, the drifting field becomes zero, making the evolution of distribution stops changing and remains at the equilibrium.

Once the drifting field $\Delta_{p,q_\theta}$ is defined, the primary objective is to minimize its norm. Essentially, the training process aims to make the drift vanishes, indicating the model has reached its target distribution:

$$
\begin{align}
\mathcal{L}(\theta) = \mathbb{E}_{\epsilon \sim p_\epsilon}\left[ \| \Delta_{p,q_\theta}(f_\theta(\epsilon)) \|^2 \right] \label{eq:vanilla} \,.
\end{align}
$$

It is important to note that the drifting field is not a learnable neural network; rather, it is a fixed mathematical operator that determines the relationship between  $p$ and $q_\theta$ at a given point $x$. Directly minimizing the objective in Equation \eqref{eq:vanilla} is challenging because it requires back-propagating gradients through the drifting field $\Delta_{p,q_\theta}$. As this field depends on the current generated distribution $q_\theta$, computing gradients becomes complex.

To address the challenge of back-propagating through the distribution-dependent drifting field, {% cite deng2026generative --file blog_refs %} proposed an indirect optimization approach using the following surrogate objective:

$$
\begin{align}
\mathcal{L}_\mathrm{drift}(\theta) = \mathbb{E}_{\epsilon \sim p_\epsilon}\left[ \| f_\theta(\epsilon) - \mathrm{stopgrad} \big(f_\theta(\epsilon) - \Delta_{p,q_\theta}(f_\theta(\epsilon)) \big) \|^2 \right] \label{eq:surrogate}
\end{align}
$$

While this loss function is mathematically equivalent to the original objective, the integration of a stop-gradient operator eliminates the need to differentiate through the complex drifting field itself.

Once the objective function is defined, the logical next question becomes: **how exactly is this drifting field constructed?** We will explore the specific mechanisms for calculating it in the following section.

# 2. Drifting fields
In this section, we describe a specific method to construct the drifting field. Note that this is not the only feasible approach. Any formulation of the drifting field is considered valid as long as the equilibrium condition \eqref{eq:equilibrium} is satisfied.

To define the drifting field component for a given distribution $\pi$ (which can be either the data $p$ or generated distribution $q_\theta$), we use the mean-shift vector field formulation

$$
V_{\pi,k}(x) = \frac{\int \pi(y) k(x, y)y dy}{ \int \pi(y) k(x, y) dy} - x\,,
$$

where $k(.,.) \ge 0$ is a similarity kernel, denoting how much each sample $y$ influences the drift at $x$. In a discrete setting with a mini-batch of samples $\{y_j\}_{j=1}^N$ drawn from $\pi$, we can approximate it as

$$
V_{\pi,k}(x) \approx \frac{\sum_{j=1}^N k(x, y_j)y_j}{ \sum_{j=1}^N k(x, y_j)} - x\,.
$$

The drifting field $\Delta_{p,q_\theta}$ is then defined as

$$
\begin{align}
\Delta_{p,q_\theta}(x) = V_{p,k}(x) - V_{q_\theta,k}(x) \,.
\end{align}
$$

The drifting field is driven by two opposing terms, <span style="color: red;">attraction</span> ($V_{p,k}(x)$) and  <span style="color: blue;">repulsion</span> ($V_{q_\theta,k}(x)$). Intuitively, this mechanism pulls samples toward the real data distribution while pushing them away from the generated distribution.

We employ a Laplace kernel $k_\tau(.,.)$ scaled by a temperature parameter $\tau$:

$$
k_{\tau}(x, y) = \mathrm{exp}\left(-\frac{1}{\tau}\|x - y\|\right)\,.
$$

{% include figure.liquid 
    path="assets/img/drifting_models/drifting_field.png" 
    title="Drifting field" 
    caption="<b style='color: var(--global-theme-color)'>Figure 1:</b> Generated samples drift toward data" 
    class="img-fluid rounded w-75 mx-auto d-block" 
%}
{: #fig-drifting-field}

As a visual illustration, [Figure 1](#fig-drifting-field) demonstrates how the drifting vectors effectively guide the generated points toward the real data manifold.


# 3. Connection to previous works
This section explores how drifting models relate to existing generative paradigms, highlighting how they bridge the gap between iterative refinement and single-step generation.

## 3.1. Score-Based Generative Models (SGM)
A recent study by {% cite lai2026unified --file blog_refs %} established that drifting can be interpreted as a score-based method acting on kernel-smoothed distributions. Specifically, by defining a smoothing field $\pi_k(x)$ via a kernel $k$

$$
\pi_k(x) = \mathbb{E}_{y\sim \pi}[k(x, y)] = \int k(x, y) \pi(y) dy \,,
$$

we can derive the associated score function $s_{\pi,k} = \nabla_x \log \pi_{k}(x)$.  When employing a Gaussian kernel $k_\tau=\mathrm{exp}\left(-\frac{1}{2\tau^2}\Vert x - y \Vert^2\right)$, the objective function \eqref{eq:surrogate} simplifies to

$$
\mathcal{L}_\mathrm{drift}(\theta) \propto \mathbb{E}_{x\sim q_\theta}\left[ \| s_{p, k_\tau} - s_{q_\theta, k_\tau} \|^2\right] \,.
$$

This result demonstrates that minimizing the drifting objective is mathematically equivalent to minimizing the reverse Fisher divergence between the kernel-smoothed target and model distributions. Note that the expectation in this formulation is taken with respect to samples drawn from $q_\theta$.


## 3.2. Distribution Matching Distillation (DMD)
DMD {% cite yin2024one --file blog_refs %} assumes a high-quality, pretrained diffusion model $p$ is available to act as a "teacher". Much like Generative Moment Matching Networks (GMMNs) {% cite li2015generative --file blog_refs %}, the core objective of DMD is to align the distribution produced by a student network ($q_\theta$) with that of the teacher ($p$). Because the score function is poorly defined in low-density regions, DMD perturbs the generated sample $f_\theta(\epsilon)$ using a diffusion process 

$$ 
\hat{x}_t = \alpha_t f_\theta(z) + \sigma_t \epsilon
$$

This perturbation yields a set of marginal distributions $q_{\theta,t}$ and $p_t$ for any given timestep $t$. DMD minimizes a weighted sum of KL divergences between these marginals:

$$
\mathcal{L}_\mathrm{DMD}(\theta) =  \mathbb{E}_{t}\left[ \omega(t) \mathcal{D}_\mathrm{KL}(q_{\theta,t} \| p_t)\right] \,.
$$

Taking the gradient of this loss with respect to the generator parameters, we get a clear "push-pull" signal based on the difference between the student and teacher scores

$$
\nabla_\theta \mathcal{L}_\mathrm{DMD}(\theta) = \mathbb{E}_{t,z,\epsilon}\left[ \omega(t) \left(\nabla_x \log q_{\theta}(\hat{x}_t, t) - \nabla_x \log p(\hat{x}_t, t) \right)\frac{d f_\theta}{d\theta}\right] \,.
$$

In DMD, the teacher scores ($s_{p_t}$) are directly provided by the frozen pre-trained diffusion model, while the student scores ($s_{q_{\theta,t}}$) are estimated by an auxiliary model that is trained alongside the generator.


Taking the gradient of $\mathcal{L}_\mathrm{drift}(\theta)$ in Equation \eqref{eq:surrogate} with respect to the parameters $\theta$, we obtain

$$
\nabla_x \mathcal{L}_\mathrm{drift}(\theta) = \mathbb{E}_{\epsilon}\left[ \Delta_{p,q_\theta}(f(\epsilon)) \frac{df_\theta}{d\theta} \right] \,.
$$

While DMD relies on a "push-pull" signal derived from the difference between two learned score functions, drifting models simplify this process by using the drifting field itself as the guiding signal. This field acts as a direct vector to distill the data distribution into the generator without the need for auxiliary score-based models.

## 3.3. Generative Adversarial Networks (GANs)

Coulomb GANs {% cite unterthiner2017coulomb --file blog_refs %} consider the GAN optimization problem as learning a potential field, where generated examples are pulled to the training data while being pushed away from one another. The generator updates its parameters to move samples along a vector field defined by the gradient of a potential field, which is learned by the discriminator. The potential $\Phi$ at any point $x$ is: 

$$
\Phi_{p,q_\theta}(x) = \int q_\theta(x) k(x, y) dy - \int p(x) k(x, y) dy = \mathbb{E}_{y\sim q_\theta}[k(x, y)] - \mathbb{E}_{y\sim p}[k(x, y)] \,.
$$

This formulation looks very simiar to the mean-shift vector field used in the drifting models, with one key difference: it lacks the normalization factor. In Coulomb GANs, the discrimiator ($D(.)$) task is to approximate this potential field $\Phi$, while the generator aims to shift its output toward regions where $D(x)$ is minimized. In contrast, drifting models completely eliminate the need for an adversarial training loop.


# 4. Practical implementation

## 4.1. Dependence on $\tau$

The theoretical solution depends on the equilibrium condition. As long as the kernel $k_\tau(.,.)$ is positive and characteristic, the only distribution that makes the drift vanish is the data distribution $p$. The temperature $\tau$ does not change what the model is trying to learn. However, the temperature does changes the loss landscape.

- Low temperature (small $\tau$): The drifting field becomes sharp and local as only data points close to the generative sample have strong influence. This helps the model to capture fine-grained and local structure of the data distribution.
- High temperature (large $\tau$): The drifting field becomes smooth and global as data points that are far away also have influence. This helps the model to capture coarse and global structure of the data distribution.

In the example below we visualize the evolution of $q$ (<span style="color: blue;">blue</span> points) toward data distribution $p$ (<span style="color: red;">red</span> points). Clearly, the training dynamics are affected by temperatures. To ensure the model is robust against specific kernel settings, the authors employ multiple temperatures simultaneously,  by summing the contributions from each temperature scale.

<div class="row justify-content-sm-center">
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include video.liquid path="assets/video/drifting_models/t_0.1.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include video.liquid path="assets/video/drifting_models/t_2.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include video.liquid path="assets/video/drifting_models/t_0.1.2.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>


## 4.2. Feature encoder
The objective function in Equation \eqref{eq:surrogate} is originally defined in the raw data space, which means the drifting field must be estimated in an extremely high-dimensional space. This poses a significant challenge: as dimensionality increases, the kernels can "degenerate". In high-dimensional space, the distance between any two random points tend to converge to a constant value, making data points appears extremely sparse (i.e., *curse of dimensionality*). To address this limitation, the authors proposed training drifting models within a more compact feature space,

$$
\mathcal{L}_\mathrm{drift}(\theta) = \mathbb{E}_{\epsilon \sim p_\epsilon}\left[ \| \phi(f_\theta(\epsilon)) - \mathrm{stopgrad} \big(\phi(f_\theta(\epsilon)) - \Delta_{p,q_\theta}( \phi(f_\theta(\epsilon))) \big) \|^2 \right] \,,
$$

where $\phi$ denotes a pre-trained feature extractor operating on both real or generated samples. By calculating the drift in this semantic space, the model can more effectively capture high-level structures while receiving meaningful training signal.


# 5. Conclusions
Drifting models introduce a novel paradigm in generative modeling by shifting the iterative refinement process—traditionally handled during inference—entirely into the training phase. By utilizing a mathematically stable drifting field, they provide a robust and efficient alternative to the adversarial instabilities of GANs and the complex requirements of diffusion model distillation.

# References

{% bibliography --style apa --group_by none --file blog_refs --cited --template bib_plain %}