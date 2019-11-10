---
layout: post
comments: true
title: "Discrete Fourier Transform"
date: 2019-11-10 14:01:00
tags: Fourier-transform
---


> Discrete Fourier Transform (DFT) is an important technique in the science of speech and sound measurement. It provides the frequency information of the signal. This post will be devoted to introduce the use of DFT in data science.
<!--more-->

{: class="table-of-content"}
* TOC
{:toc}


## Audio Signal Processing

An audio signal can be represented as an encoding of air pressure over time. Formally, it is a function $$S\colon \mathbb{R} \to \mathbb{R}$$ that maps every every point $$t$$ in time to sound pressure value $$S_t = S(t)$$. The representation of how air pressure moves over time is called *analog* or *continuous* time signal. In order to process and store audio signals digitally on computers or electronic devices, it often involves discretizing signals, referred to as *sampling*. The sound wave is converted into data through a series of snapshot measurements, or samples. A sample is taken at a particular time in the audio wave, recording amplitude. This information is then converted into binary data.

![EA]({{ '/assets/images/450px-Signal_Sampling.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 1. Signal sampling representation. (Image source: [Wikipedia](https://en.wikipedia.org/wiki/File:Signal_Sampling.png))*

The quality of sound depends on the following two factors

1. **Sample rate** (also called *sample frequency*) is the number of samples per second and is measured in Hertz (Hz). The most common sample rate is 44.1 kHz, or 44100 samples per second for audio. Human can hear between 20 Hz and 20kHz. 
2. **Bit depth** determines the number of possible values that we can record for each sample. The most common bit depths are 16-bit, 24-bit, and 32-bit. Bit depth affects the quantization, which is a process of discretize real amplitude values using a finite set of integers.

Sampling, in general, is not invertible function and results in loss of information (i.e., [aliasing](https://en.wikipedia.org/wiki/Aliasing)). The higher the sampling rate, the more accurate would be to stored information and the construction from its samples. Which is the minimum necessary sampling frequency for a given type of signal that allows accurate reconstruction?  The answer is given by Nyquist-Shannon Sampling Theorem.
> The sampling frequency should be at least twice the highest frequency contained in the signal

Frequency is defined as the number of cycles that the wave completes in one second. This value is measured in Hertz (Hz).

To understand sound signals, a common step is to decompose it into fundamental components, known as frequencies (see Fig. 2). In the next section, we will discuss the Discrete Fourier Transform (DFT) for frequency representation. 

![EA]({{ '/assets/images/FFT-Time-Frequency-View-540.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 2. How signal is viewed in the domain of time and frequency (Image source: [Nti-audio](https://www.nti-audio.com/en/support/know-how/fast-fourier-transform-fft))*

## Discrete Fourier Transform

Given any time series $$x_0, x_1, \dots, x_{K-1}$$, it can be expressed as follows

$$
x_i = \sum_{k=0}^{K-1} \hat{x}_k \exp(-2\pi ki/K)\,,
$$ 

where 

$$
\hat{x}_i = \frac{1}{K}\sum_{k=0}^{K-1} x_k \exp(2\pi ki/K)\,.
$$

## Applications of DFT

## References
