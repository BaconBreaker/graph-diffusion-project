import torch
from tqdm.auto import tqdm
import logging

import torch.nn.functional as f

from BaseDiffusion import Diffusion
from ..utils.stuff import unsqueeze_n


def x_t_sub_from_noise(alpha, alpha_hat, beta, noise, predicted_noise, x_t):
    res = 1 / torch.sqrt(alpha + 1e-5) \
        * (x_t - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) \
        + torch.sqrt(beta) * noise
    return res


def generate_gaussian_noise(n, shape, device):
    return torch.randn((n, *shape)).to(device)


def generate_noise(noise_function, n, *args, **kwargs):
    """Generates noise for diffusion process through a given noise function with args and kwargs."""
    return noise_function(n, *args, **kwargs)


def image_sample_post_process(x):
    """Post-processing function for generated images."""
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)
    return x



def x_t_sub_from_x0(alpha, alpha_hat, alpha_hat_sub_1, _beta, noise, x_0, x_t):
    """Computes x_{t-1} from x_0, x_t given, along with alpha, alpha_hat and beta."""
    mu_part1 = torch.sqrt(alpha) * (1 - alpha_hat) * x_t
    mu_part2 = torch.sqrt(alpha_hat_sub_1) * (1 - alpha) * x_0
    mu = (mu_part1 + mu_part2) \
        / 1 - alpha_hat
    variance = ((1 - alpha) * (1 - alpha_hat_sub_1)) / (1 - alpha_hat)
    return mu + torch.sqrt(variance) * noise


class GaussianDiffusion(Diffusion):
    def __init__(self, noise_shape, model_target="noise", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_shape = noise_shape
        self.model_target = model_target

    def noise_function(self, shape):
        return torch.randn(shape)

    def sample_from_noise_fn(self, n):
        samples = [self.noise_function(self.noise_shape) for _ in range(n)]
        return torch.stack(samples).to(self.device)

    def diffuse(self, x, t):
        """Computes the diffusion process. Returns x_t and epsilon_t."""
        n = x.size(0)
        x_n_dims = len(x.shape[1:])
        sqrt_alpha_hat = unsqueeze_n(torch.sqrt(self.alpha_hat[t]), x_n_dims)
        sqrt_one_minus_alpha_hat = unsqueeze_n(torch.sqrt(1 - self.alpha_hat[t]), x_n_dims)
        n = x.size(0)
        epsilon = self.sample_from_noise_fn(n)
        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon
        return x_t, epsilon

    def sample(self, model, n, labels=None, sample_post_process=None):
        """Sample n examples from the model, with optional labels for conditional sampling.
        The `labels` argument is ignored if the model is not conditional.
        """
        if self.conditional:
            sample_args = [n, model, labels]
        else:
            sample_args = [n, model]

        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = self.sample_from_noise_fn(n)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                x = self.sample_previous_x(x, i, *sample_args)

        model.train()
        if sample_post_process is not None:
            x = sample_post_process(x)

        return x

    def sample_previous_x(self, x, i, n, model, labels):
        t = (torch.ones(n) * i).long().to(self.device)
        n = x.size(0)
        prediction = model(x, t, labels)

        alpha = self.alpha[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        alpha_hat_sub_1 = self.alpha_hat[t - 1][:, None, None, None]
        beta = self.beta[t][:, None, None, None]
        noise = self.sample_from_noise_fn(n)

        if self.model_target == "noise":
            x = x_t_sub_from_noise(alpha, alpha_hat, beta, noise, prediction, x)
        else:
            x = x_t_sub_from_x0(alpha, alpha_hat, alpha_hat_sub_1, beta, noise, prediction, x)
        return x

    def loss(self,  prediction, noise, _batch):
        """Computes the loss for the diffusion process."""
        return f.mse_loss(prediction, noise)
