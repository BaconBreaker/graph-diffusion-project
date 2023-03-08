from tqdm.auto import tqdm
import logging
from abc import ABC, abstractmethod
from typing import Callable, Union
from math import pi

import torch
import torch.nn.functional as F

from utils import unsqueeze_n, cum_matmul, cat_dist
from noise import GaussianNoise, Noise, SymmetricGaussianNoise


def generate_gaussian_noise(n, shape, device):
    return torch.randn((n, *shape)).to(device)


def generate_noise(noise_function, n, *args, **kwargs):
    """Generates noise for diffusion process through a given noise function with args and kwargs."""
    return noise_function(n, *args, **kwargs)


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


def prepare_noise_schedule(noise_schedule, T, *args, **kwargs):
    """Prepares a noise schedule for T steps. Use `noise_schedule` param to select a build in or custom schedule.
    Custom schedules are functions taking a T parameter and returning alpha, alpha_hat, beta of length T, with
    optional Args and Kwargs.
    """
    if noise_schedule == "linear":
        alpha, alpha_hat, beta = linear_noise_schedule(T=T, *args, **kwargs)
    elif noise_schedule == "cosine":
        alpha, alpha_hat, beta = cosine_noise_schedule(T=T, *args, **kwargs)
    else:
        alpha, alpha_hat, beta = noise_schedule(T=T, *args, **kwargs)
    return alpha, alpha_hat, beta


def x_t_sub_from_noise(alpha, alpha_hat, beta, noise, predicted_noise, x_t):
    res = 1 / torch.sqrt(alpha + 1e-5) \
        * (x_t - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) \
        + torch.sqrt(beta) * noise
    return res


def x_t_sub_from_x0(alpha, alpha_hat, alpha_hat_sub_1, _beta, noise, x_0, x_t):
    """Computes x_{t-1} from x_0, x_t given, along with alpha, alpha_hat and beta."""
    mu_part1 = torch.sqrt(alpha) * (1 - alpha_hat) * x_t
    mu_part2 = torch.sqrt(alpha_hat_sub_1) * (1 - alpha) * x_0
    mu = (mu_part1 + mu_part2) \
        / 1 - alpha_hat
    variance = ((1 - alpha) * (1 - alpha_hat_sub_1)) / (1 - alpha_hat)
    return mu + torch.sqrt(variance) * noise


def image_sample_post_process(x):
    """Post-processing function for generated images."""
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)
    return x


class Diffusion(ABC):
    def __init__(self,
                 noise_schedule: Union[str, Callable] = "linear",
                 noise_steps: int = 1000,
                 device: str = "cpu",
                 conditional: bool = True,
                 **d_kwargs):
        self.noise_steps = noise_steps
        self.device = device
        self.conditional = conditional

        if noise_schedule == "linear":
            self.noise_schedule = linear_noise_schedule
        elif noise_schedule == "cosine":
            self.noise_schedule = cosine_noise_schedule
        else:
            self.noise_schedule = noise_schedule

        self.alpha, self.alpha_hat, self.beta = self.noise_schedule(
            T=noise_steps, **d_kwargs
        )

    def sample_time_steps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    @abstractmethod
    def diffuse(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass


class GaussianDiffusionForNoise(Diffusion):
    def __init__(self, noise_shape, *args, model_target="noise", **kwargs):
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
        """Sample n examples from the model, with optional labels for conditional sampling. The `labels` argument
        is ignored if the model is not conditional.
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
        x_n_dims = len(x.shape[1:])
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


class UniformCategoricalDiffusion(Diffusion):
    def __init__(self, n_categorical, n_vals, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_categorical = n_categorical
        self.n_vals = n_vals

    def sample_from_noise_fn(self, ts, bar=False):
        if bar:
            samples = [self.get_qt_bar(t) for t in ts]
        else:
            samples = [self.get_qt(t) for t in ts]
        return torch.stack(samples).to(self.device)

    def get_qt(self, t):
        """
        Defined by Austin et al. 2023, appendix 2.1. We use the linear algebra notation.
        Obtains the matrix Q_t.
        """
        if isinstance(t, torch.Tensor):
            t = t.clone()
        t -= 1
        k = self.n_vals
        q1 = unsqueeze_n(1 - self.beta[t], 2) * torch.eye(k).unsqueeze(0)
        q2 = unsqueeze_n(self.beta[t], 2) * torch.ones([k, k]).unsqueeze(0)
        q = q1 + q2 / k
        q = q.unsqueeze(1).repeat(1, self.n_categorical, 1, 1)
        return q

    def get_qt_bar(self, t):
        """Obtains the \\bar{Q}_t matrix."""
        if isinstance(t, int):
            t = unsqueeze_n(torch.tensor(t), 2)

        if t.ndim in [0, 1]:
            t = unsqueeze_n(t, 2 - t.ndim)

        qt_bars = []
        for individual_t in t:
            sample = [self.get_qt(i) for i in range(1, individual_t + 1)]
            sample = torch.cat(sample, dim=0)
            sample = cum_matmul(sample, dim=0)
            qt_bars.append(sample)
        return torch.stack(qt_bars)

    def q_xt_given_x0(self, x0, t):
        qt_bar = self.get_qt_bar(t)
        p = torch.einsum("abi,abij->abj", x0, qt_bar)
        return p

    def q_xtsub1_given_xt_x0(self, xt, x0, t):
        """
        Algorithm used from Austin et al. 2023, section 3.
        Computes p_theta(x_{t-1} | x_t) which is approximated by
        the sum of q(x_{t-1} | x_t, x_0)p_\theta(x_0 | x_t) over all
        probabilities generated by the model.
        """
        qt = self.get_qt(t)  # qt(t)
        qt_transpose = qt.transpose(-1, -2)
        qt_sub_bar = self.get_qt_bar(t - 1)  # qt_bar(t-1)
        qt_bar = self.get_qt_bar(t)  # qt_bar(t)
        q1 = torch.einsum("abi,abij->abj", xt, qt_transpose)
        q2 = torch.einsum("abi,abij->abj", x0, qt_sub_bar)
        q3 = torch.einsum("abi,abij->abj", x0, qt_bar)
        q4 = torch.einsum("abi,abi->ab", q3, xt)
        q5 = q1 * q2
        q = q5 / q4.unsqueeze(-1)
        return q

    def q_xtsub1_x1_given_x0(self, xt, x0, t):
        xtsub1_given_xt_x0 = self.q_xtsub1_given_xt_x0(xt, x0, t)
        x1_given_x0 = self.q_xt_given_x0(x0, t)
        xtsub1_x1_given_x0 = torch.einsum("abi,abj->abij", xtsub1_given_xt_x0, x1_given_x0)
        return xtsub1_x1_given_x0

    def p_previous_x(self, xt, model_out, t):
        """
        Computes p_theta(x_{t-1} | x_t) according to Austin et al. 2023, section 3.
        Also handles cases where t == 1, where it returns the model output p_theta(x_0 | x_1)
        """
        # Computes the model out, i.e. p_theta(x_0 | x_t)

        # If t == 1 then p_theta(x_{t-1} | x_t) = p_theta(x_0 | x_1)
        # Which is the model output.
        if t == 1:
            return model_out
        # Here we create a tensor of all different states, i.e. one hot encodings, with the proper shape
        all_eyes = torch.eye(model_out.size(-1)).unsqueeze(1).unsqueeze(1)
        all_eyes = all_eyes.repeat(1, model_out.size(0), model_out.size(1), 1)

        # For each one hot vector representing x0, we compute q(x_{t-1},x_t | x_0)
        p_tmp = torch.stack(
            [self.q_xtsub1_x1_given_x0(xt, tmp_x0, t) for tmp_x0 in all_eyes]
        )

        # Since we already know x_t, we can index in and select the correct probability
        # via a batched dot product.
        p_tmp_2 = torch.einsum("tncij,ncj->tnci", p_tmp, xt)

        # We then map out all p_\theta(x_0 | x_t) and reshape properly
        model_ps = torch.stack(
            [model_out[:, :, i] for i in range(model_out.size(-1))]
        ).unsqueeze(-1).repeat(1, 1, 1, model_out.size(-1))

        # Finally, we compute the sum over all x_0 and normalize
        p_final = (p_tmp_2 * model_ps).sum(dim=0)
        p_final = p_final / p_final.sum(dim=-1, keepdim=True)
        return p_final

    def sample_previous_x(self, xt, t, n, model, labels=None):
        """
        Samples x_{t-1} given x_t, according to Austin et al. 2023, section 3.
        """
        model_out = model(xt, t, labels=labels)
        model_out = F.softmax(model_out, dim=-1)
        p = self.p_previous_x(xt, model_out, t)
        o = cat_dist(p, self.n_vals).float()
        return o

    def diffuse(self, x_0, t):
        """Computes the diffusion process. Takes x_0 and t, and returns x_t and epsilon_t = Q_t."""
        q_t_bar = self.get_qt_bar(t)
        # probs = x_0 @ q_t_bar
        probs = torch.einsum("abi,abij->abj", x_0, q_t_bar)
        o = cat_dist(probs, self.n_vals).float()
        return o, q_t_bar

    def uniform_x(self, n):
        x = torch.randint(0, self.n_vals, (n, self.n_categorical))
        x = F.one_hot(x, self.n_vals).float()
        return x

    def sample(self, model, n, labels=None):
        """
        Sample n examples from the model, with optional labels for conditional sampling. The `labels` argument
        The label argument is ignored if the model is unconditional.
        """
        if self.conditional:
            sample_args = [n, model, labels]
        else:
            sample_args = [n, model]

        logging.info(f"Sampling {n} new categorical features....")
        model.eval()
        with torch.no_grad():
            # x begins as a one hot sample from a uniform distribution.
            x = self.uniform_x(n)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                x = self.sample_previous_x(x, i, *sample_args)

        model.train()
        return x
