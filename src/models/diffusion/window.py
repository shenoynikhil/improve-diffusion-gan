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

    def update_T(self):
        """Adaptively updating T"""
        # if self.aug_type == "ada":
        #     _p = min(self.p, self.ada_maxp) if self.ada_maxp else self.p
        #     self.aug.p.copy_(torch.tensor(_p))

        t_adjust = round(self.p * self.t_add)
        t = np.clip(int(self.t_min + t_adjust), a_min=self.t_min, a_max=self.t_max)

        # update beta values according to new T
        self.set_diffusion_process(t, self.beta_schedule)

        # sampling t
        # more zeros to allow for non-denoised images
        self.t_epl = np.zeros(64, dtype=np.int)
        diffusion_ind = 32
        t_diffusion = np.zeros((diffusion_ind,)).astype(np.int)
        # subtract by window length to avoid sampling timesteps that overflow
        t = t - (self.window_length - 1)
        if self.ts_dist == "priority":
            prob_t = np.arange(t) / np.arange(t).sum()
            t_diffusion = np.random.choice(np.arange(1, t + 1), size=diffusion_ind, p=prob_t)
        elif self.ts_dist == "uniform":
            t_diffusion = np.random.choice(np.arange(1, t + 1), size=diffusion_ind)
        self.t_epl[:diffusion_ind] = t_diffusion

    def forward(self, x_0, t):
        x_0 = self.aug(x_0)
        assert isinstance(x_0, torch.Tensor) and x_0.ndim == 4
        device = x_0.device

        alphas_bar_sqrt = self.alphas_bar_sqrt.to(device)
        one_minus_alphas_bar_sqrt = self.one_minus_alphas_bar_sqrt.to(device)

        t = t.to(device)
        # modify t by adding a window of timesteps at each sample
        t = t.view(-1, 1) + torch.arange(
            0, self.window_length, dtype=torch.int32, device=device
        ).view(1, -1)
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
