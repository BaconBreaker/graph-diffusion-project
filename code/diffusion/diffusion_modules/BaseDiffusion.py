from abc import ABC, abstractmethod
from typing import Callable, Union

import torch

from ..utils.noise_schedule import linear_noise_schedule, cosine_noise_schedule

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
