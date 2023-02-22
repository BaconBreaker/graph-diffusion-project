import torch
from tqdm.auto import tqdm
import logging

import numpy as np


def generate_gaussian_image_noise(n, img_size, device):
    """Used to generate a noise sample. Change this to suit needs."""
    return torch.randn((n, 3, img_size, img_size)).to(device)


def generate_noise(n, noise_function, *args):
    return noise_function(n, *args)


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02,
                 img_size=64, device='cpu', noise_schedule='linear',
                 noise_function=None):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.alpha, self.alpha_hat, self.beta = self.prepare_noise_schedule(noise_schedule)

        if noise_function is not None:
            self.noise_function = noise_function
        else:
            self.noise_function = generate_gaussian_image_noise

    def __cosine_f(self, t, s=0.008):
        return torch.cos((t / self.noise_steps + s) / (1 + s) * np.pi / 2) ** 2

    def prepare_noise_schedule(self, method):
        methods_supported = ["linear", "cosine"]
        if method == "linear":
            alpha, alpha_hat, beta = self.prepare_linear_noise_schedule()
        elif method == "cosine":
            alpha, alpha_hat, beta = self.prepare_cosine_noise_schedule()
        else:
            raise NotImplementedError(
                f"Method {method} not supported. Choose from {methods_supported}."
            )
        return alpha, alpha_hat, beta

    def prepare_cosine_noise_schedule(self):
        f0 = self.__cosine_f(0)
        ft = self.__cosine_f(torch.arange(0, 101))

        alpha_hat = ft / f0
        alpha_hat_0_99 = alpha_hat[:-2]
        alpha_hat_1_100 = alpha_hat[1:]
        beta = 1 - alpha_hat_1_100 / alpha_hat_0_99
        alpha = 1 - beta
        return alpha, alpha_hat_1_100, beta

    def prepare_linear_noise_schedule(self):
        beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        alpha = 1 - beta
        alpha_hat = torch.cumprod(alpha, dim=0)
        return alpha, alpha_hat, beta

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon
        return x_t, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():

            x = generate_noise(n, self.noise_function, self.img_size, self.device)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_value = model(x, t, None)
                    predicted_noise = torch.lerp(predicted_noise, uncond_predicted_value, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) \
                    * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) \
                    + torch.sqrt(beta) * noise

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
