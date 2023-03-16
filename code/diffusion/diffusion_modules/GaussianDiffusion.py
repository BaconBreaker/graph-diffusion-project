from os import path

import torch
from tqdm.auto import tqdm

import torch.nn.functional as f

from diffusion_modules.BaseDiffusion import Diffusion
from utils.stuff import unsqueeze_n
from utils.plots import save_graph


def x_t_sub_from_noise(alpha, alpha_hat, beta, noise, predicted_noise, x_t):
    res = 1 / torch.sqrt(alpha + 1e-5) \
        * (x_t - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) \
        + torch.sqrt(beta) * noise
    return res


def generate_gaussian_noise(n, shape):
    return torch.randn((n, *shape))


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
    def __init__(self, args_, *args, **kwargs):
        super().__init__(args_, *args, **kwargs)
        self.noise_shape = args_.noise_shape
        self.model_target = args_.model_target
        self.tensors_to_diffuse = args_.tensors_to_diffuse

    def noise_function(self, shape):
        return torch.randn(shape)

    def sample_from_noise_fn(self, noise_shape):
        samples = self.noise_function(noise_shape)
        return samples

    def diffuse(self, batch_dict, t):
        noise = []
        # Tensors_to_diifuse is a parser arg that specifies which tensors to diffuse
        for name in self.tensors_to_diffuse:
            n, x = self._diffuse(batch_dict[name], t)
            noise.append(n)
            batch_dict[name] = x
        return batch_dict, noise

    def _diffuse(self, x, t):
        """Computes the diffusion process. Returns x_t and epsilon_t."""
        x_n_dims = len(x.shape[1:])
        sqrt_alpha_hat = unsqueeze_n(torch.sqrt(self.alpha_hat[t]), x_n_dims)
        sqrt_one_minus_alpha_hat = unsqueeze_n(torch.sqrt(1 - self.alpha_hat[t]), x_n_dims)
        epsilon = self.sample_from_noise_fn(x.shape)
        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon

        return x_t, epsilon

    def sample(self, model, batch_dict, save_output=False, post_process=None):
        """Sample n examples from the model, with optional labels for conditional sampling.
        The `labels` argument is ignored if the model is not conditional.
        """

        model.eval()
        with torch.no_grad():
            # Make a copy of the batch_dict, and replace the tensors to diffuse with noise.
            # This ensures that vectors used for conditioning are not modified.
            sample_dict = batch_dict.copy()
            for name in self.tensors_to_diffuse:
                noise_shape = batch_dict[name].shape
                x = self.sample_from_noise_fn(noise_shape)
                sample_dict[name] = x

            if save_output and post_process is not None:
                save_graph(sample_dict, self.noise_steps, self.run_name, post_process)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                sample_dict = self.sample_previous_x(sample_dict, i, model,
                                                     save_output=save_output,
                                                     post_process=post_process)

        model.train()

        return sample_dict

    def sample_previous_x(self, sample_dict, i, model,
                          save_output=False, post_process=None):
        n = sample_dict[self.tensors_to_diffuse[0]].size(0)
        t = (torch.ones(n) * i).long()
        prediction = model(sample_dict, t)
        
        for name in self.tensors_to_diffuse:
            x = sample_dict[name]
            pred = prediction[name]
            n_unsqueezed = len(x.shape[1:])

            alpha = unsqueeze_n(self.alpha[t], n_unsqueezed)
            alpha_hat = unsqueeze_n(self.alpha_hat[t], n_unsqueezed)
            alpha_hat_sub_1 = unsqueeze_n(self.alpha_hat[t - 1], n_unsqueezed)
            beta = unsqueeze_n(self.beta[t], n_unsqueezed)

            noise = self.sample_from_noise_fn(x.shape)
            if self.model_target == "noise":
                x = x_t_sub_from_noise(alpha, alpha_hat, beta, noise, pred, x)
            else:
                x = x_t_sub_from_x0(alpha, alpha_hat, alpha_hat_sub_1, beta, noise, pred, x)

            sample_dict[name] = x

            if save_output and post_process is not None:
                save_graph(sample_dict, i, self.run_name, post_process)

        return sample_dict

    def loss(self,  prediction, noise, _batch):
        """Computes the loss for the diffusion process."""
        return f.mse_loss(prediction, noise, reduction="sum")
