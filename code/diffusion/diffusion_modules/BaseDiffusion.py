import logging
from abc import ABC, abstractmethod
from typing import Callable, Union

import torch

from utils.noise_schedule import linear_noise_schedule, cosine_noise_schedule


class Diffusion(ABC):
    def __init__(self, args, **d_kwargs):
        self.noise_steps = args.diffusion_timesteps
        noise_schedule = args.noise_schedule
        self.run_name = args.run_name
        self.device = args.device
        self.conditional = args.conditional
        self.d_kwargs = d_kwargs

        if noise_schedule == "linear":
            self.noise_schedule = linear_noise_schedule
        elif noise_schedule == "cosine":
            self.noise_schedule = cosine_noise_schedule
        else:
            self.noise_schedule = noise_schedule

        self.__create_greeks()

    def set_device(self, device):
        self.device = device
        self.__create_greeks()

    def sample_time_steps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,)).to(self.device)

    def __create_greeks(self):
        self.alpha, self.alpha_hat, self.beta = self.noise_schedule(
            T=self.noise_steps, device=self.device, **self.d_kwargs
        )

    @abstractmethod
    def diffuse(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass

    @abstractmethod
    def loss(self, prediction, noise, batch):
        pass
