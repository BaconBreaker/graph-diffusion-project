from abc import ABC, abstractmethod
import torch
import logging

logger = logging.getLogger(__name__)


# A noise class should support
# 1. A noise function capable of generating standard noise
# 2. A method to sample from this noise function n times
class Noise(ABC):
    """Abstract base class for noise functions."""
    def __init__(self):
        pass

    @abstractmethod
    def noise_function(self, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def sample(self, n, *args, **kwargs):
        pass


class GaussianNoise(Noise):
    """Gaussian noise function."""
    def __init__(self, noise_shape, device="cpu"):
        super().__init__()
        self.noise_shape = noise_shape
        self.device = device

    def noise_function(self, n):
        return torch.randn

    def sample(self, n, _alpha_t, _alpha_hat_t, _beta_t):
        return self.noise_function((n, *self.noise_shape)).to(self.device)

    def sample_like(self, x):
        return torch.randn_like(x).to(self.device)


class UniformCategoricalNoise(Noise):
    """Uniform categorical noise function."""
    def __init__(self, n_categoricals, device="cpu"):
        super().__init__()
        self.n_categoricals = n_categoricals
        self.device = device

    def noise_function(self, n, _alpha_t, _alpha_hat_t, beta_t):
        q_base_val = 1 / self.n_categoricals * beta_t
        q_shape = (n, self.n_categoricals, self.n_categoricals)
        q = torch.full(q_shape, q_base_val).to(self.device)
        for i in range(self.n_categoricals):
            q[:, i, i] = 1 - (self.n_categoricals - 1) / self.n_categoricals * beta_t
        return q
