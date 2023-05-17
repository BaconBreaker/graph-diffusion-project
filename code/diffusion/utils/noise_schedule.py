from math import pi

import torch


def clamp_greeks(alpha, alpha_hat, beta):
    """Clamps the Greeks to avoid numerical issues in place."""
    alpha = torch.clamp(alpha, 0.001, 0.999)
    alpha_hat = torch.clamp(alpha_hat, 0.001, 0.999)
    beta = torch.clamp(beta, 0.001, 0.999)
    return alpha, alpha_hat, beta


def cosine_schedule_step(t, T, s):
    """Cosine noise schedule step."""
    t_inner = (t / T + s) / (1.0 + s) * pi / 2.0
    return torch.square(torch.cos(t_inner))


def cosine_noise_schedule(T=1000, s=0.008, device="cpu",
                          *_args, **_kwargs):
    """Computes a cosine noise schedule for T steps."""
    f0 = cosine_schedule_step(torch.tensor(0), T, s)
    ft = cosine_schedule_step(torch.arange(0, T + 1), T, s)

    alpha_hat = ft / f0
    beta = 1 - alpha_hat[1:] / alpha_hat[:-1]
    alpha = 1 - beta
    alpha_hat = alpha_hat[1:]

    alpha, alpha_hat, beta = clamp_greeks(alpha, alpha_hat, beta)
    alpha, alpha_hat, beta = alpha.to(device), alpha_hat.to(device), beta.to(device)
    return alpha, alpha_hat, beta


def linear_noise_schedule(T, beta_start=1e-4, beta_end=0.02,
                          device="cpu", *_args, **_kwargs):
    """Computes a linear noise schedule for T steps."""
    beta = torch.linspace(beta_start, beta_end, T)
    alpha = 1 - beta
    alpha_hat = torch.cumprod(alpha, dim=0)

    alpha, alpha_hat, beta = clamp_greeks(alpha, alpha_hat, beta)
    alpha, alpha_hat, beta = alpha.to(device), alpha_hat.to(device), beta.to(device)
    return alpha, alpha_hat, beta
