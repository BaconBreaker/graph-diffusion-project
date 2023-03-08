from math import pi

import torch


def cosine_schedule_step(t, T, s):
    """Cosine noise schedule step."""
    t_inner = (t / T + s) / (1.0 + s) * pi / 2.0
    return torch.square(torch.cos(t_inner))


def cosine_noise_schedule(T=1000, s=0.008, *_args, **_kwargs):
    """Computes a cosine noise schedule for T steps."""
    f0 = cosine_schedule_step(torch.tensor(0), T, s)
    ft = cosine_schedule_step(torch.arange(0, T + 1), T, s)

    alpha_hat = ft / f0
    beta = 1 - alpha_hat[1:] / alpha_hat[:-1]
    beta[beta < 0] = 0
    beta[beta > 0.999] = 0.999
    alpha = 1 - beta
    alpha_hat = alpha_hat[1:]
    return alpha, alpha_hat, beta


def linear_noise_schedule(T, beta_start=1e-4, beta_end=0.02, *_args, **_kwargs):
    """Computes a linear noise schedule for T steps."""
    beta = torch.linspace(beta_start, beta_end, T)
    alpha = 1 - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    return alpha, alpha_hat, beta
