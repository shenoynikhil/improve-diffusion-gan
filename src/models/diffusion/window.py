"""Code from Diffusion-GAN Repository: https://github.com/Zhendong-Wang/Diffusion-GAN/"""
import numpy as np
import torch

from .base import Diffusion


def q_sample_window(
    x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, w, noise_type="gauss", noise_std=1.0
):
    if noise_type == "gauss":
        noise = torch.randn_like(x_0, device=x_0.device) * noise_std
    elif noise_type == "bernoulli":
        noise = (torch.bernoulli(torch.ones_like(x_0) * 0.5) * 2 - 1.0) * noise_std
    else:
        raise NotImplementedError(noise_type)

    alphas_t_sqrt = alphas_bar_sqrt[t].view(-1, w, 1, 1, 1)
    one_minus_alphas_bar_t_sqrt = one_minus_alphas_bar_sqrt[t].view(-1, w, 1, 1, 1)
    x_t = (
        alphas_t_sqrt * x_0.unsqueeze(1) + one_minus_alphas_bar_t_sqrt * noise.unsqueeze(1)
    ).view(-1, *x_0.shape[1:])
    return x_t


class WindowDiffusion(Diffusion):
    """Modification of Diffusion process,
    Instead of sampling a single timestep, sample a window of continuous timesteps.
    """

    def __init__(self, window_length: int = 3, *args, **kwargs):
        self.window_length = window_length
        super().__init__(*args, **kwargs)

    def sample_t(self, batch_size):
        # sample time steps with length of window_length from t_epl for each sample
        # shape: (batch_size, window_length)
        return torch.from_numpy(
            np.random.choice(self.t_epl, size=batch_size * self.window_length, replace=True)
        ).view(batch_size, -1)

    def forward(self, x_0, t):
        x_0 = self.aug(x_0)
        assert isinstance(x_0, torch.Tensor) and x_0.ndim == 4
        device = x_0.device

        alphas_bar_sqrt = self.alphas_bar_sqrt.to(device)
        one_minus_alphas_bar_sqrt = self.one_minus_alphas_bar_sqrt.to(device)

        # modify t by adding a window of timesteps at each sample
        t = t.to(device)
        x_t = q_sample_window(
            x_0,
            alphas_bar_sqrt,
            one_minus_alphas_bar_sqrt,
            t,
            self.window_length,
            noise_type=self.noise_type,
            noise_std=self.noise_std,
        )
        return x_t, t
