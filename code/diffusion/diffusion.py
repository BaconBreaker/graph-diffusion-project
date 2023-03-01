from tqdm.auto import tqdm
import logging

import torch
import numpy as np

from noise import GaussianNoise, Noise


def generate_gaussian_noise(n, shape, device):
    return torch.randn((n, *shape)).to(device)


def generate_noise(noise_function, n, *args, **kwargs):
    return noise_function(n, *args, **kwargs)


def cosine_f(t, T, s):
    return torch.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2


def cosine_noise_schedule(T=1000, s=0.008, *_args, **_kwargs):
    f0 = cosine_f(torch.tensor(0), T, s)
    ft = cosine_f(torch.arange(0, T + 1), T, s)

    alpha_hat = ft / f0
    beta = 1 - alpha_hat[1:] / alpha_hat[:-1]
    alpha = 1 - beta
    alpha_hat = alpha_hat[1:]
    return alpha, alpha_hat, beta


def linear_noise_schedule(T, beta_start=1e-4, beta_end=0.02, *_args, **_kwargs):
    beta = torch.linspace(beta_start, beta_end, T)
    alpha = 1 - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    return alpha, alpha_hat, beta


def prepare_noise_schedule(noise_schedule, T, *args, **kwargs):
    if noise_schedule == "linear":
        alpha, alpha_hat, beta = linear_noise_schedule(T=T, *args, **kwargs)
    elif noise_schedule == "cosine":
        alpha, alpha_hat, beta = cosine_noise_schedule(T=T, *args, **kwargs)
    else:
        alpha, alpha_hat, beta = noise_schedule(T=T, *args, **kwargs)
    return alpha, alpha_hat, beta


def x_t_sub_from_noise(alpha, alpha_hat, beta, noise, predicted_noise, x_t):
    return 1 / torch.sqrt(alpha) \
        * (x_t - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) \
        + torch.sqrt(beta) * noise


def x_t_sub_from_x0(alpha, alpha_hat, alpha_hat_sub_1, beta, noise, x_0, x_t):
    mu_part1 = torch.sqrt(alpha) * (1 - alpha_hat) * x_t
    mu_part2 = torch.sqrt(alpha_hat_sub_1) * (1 - alpha) * x_0
    mu = (mu_part1 + mu_part2) \
        / 1 - alpha_hat
    variance = ((1 - alpha) * (1 - alpha_hat_sub_1)) / (1 - alpha_hat)
    return mu + torch.sqrt(variance) * noise


def image_sample_post_process(x):
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)
    return x


class Diffusion:
    def __init__(self,
                 noise_schedule,
                 noise_function,
                 noise_steps=1000,
                 device="cpu",
                 model_target="noise",
                 **d_kwargs):
        self.noise_steps = noise_steps
        self.device = device
        self.model_target = model_target

        if noise_schedule == "linear":
            self.noise_schedule = linear_noise_schedule
        elif noise_schedule == "cosine":
            self.noise_schedule = cosine_noise_schedule
        else:
            self.noise_schedule = noise_schedule

        self.alpha, self.alpha_hat, self.beta = self.noise_schedule(
            T=noise_steps, **d_kwargs
        )

        if noise_function == "gaussian":
            self.noise_function = GaussianNoise(**d_kwargs, device=self.device)
        elif isinstance(noise_function, Noise):
            self.noise_function = noise_function
        else:
            raise ValueError(f"Unknown noise function {noise_function}")

    def diffuse(self, x, t):
        if (t <= 0).any().item():
            raise ValueError("all t's must be greater than 0 in order to diffuse.")

        # Unsqueeze over multiple dimensions in one go
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        n = x.size(0)
        epsilon = self.noise_function.sample(n)
        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon
        return x_t, epsilon

    def sample_time_steps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, sample_post_process=None):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = self.noise_function.sample(n,
                                           alpha=self.alpha,
                                           alpha_hat=self.alpha_hat,
                                           beta=self.beta)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                x = self.sample_previous_x(i, labels, model, n, x)

        model.train()
        if sample_post_process is not None:
            x = sample_post_process(x)

        return x

    def sample_previous_x(self, i, labels, model, n, x):
        t = (torch.ones(n) * i).long().to(self.device)
        prediction = model(x, t, labels)
        alpha = self.alpha[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        alpha_hat_sub_1 = self.alpha_hat[t - 1][:, None, None, None]
        beta = self.beta[t][:, None, None, None]
        noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
        if self.model_target == "noise":
            x = x_t_sub_from_noise(alpha, alpha_hat, beta, noise, prediction, x)
        else:
            x = x_t_sub_from_x0(alpha, alpha_hat, alpha_hat_sub_1, beta, noise, prediction, x)
        return x
