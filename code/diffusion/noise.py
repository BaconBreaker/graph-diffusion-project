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
    def noise_function(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample(self, n):
        pass


class GaussianNoise(Noise):
    """Gaussian noise function."""
    def __init__(self, noise_shape, device="cpu"):
        super().__init__()
        self.noise_shape = noise_shape
        self.device = device

    def noise_function(self, n):
        return torch.randn

    def sample(self, n):
        return torch.randn((n, *self.noise_shape)).to(self.device)

    def sample_like(self, x):
        return torch.randn_like(x).to(self.device)
