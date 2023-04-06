"""Code from Diffusion-GAN Repository: https://github.com/Zhendong-Wang/Diffusion-GAN/"""
import numpy as np
import torch

from src.models.utils import DiffAugment

# ----------------------------------------------------------------------------
# Helpers for doing defusion process.


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    def continuous_t_beta(t, T):
        b_max = 5.0
        b_min = 0.1
        alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T**2)
        return 1 - alpha

    if beta_schedule == "continuous_t":
        betas = continuous_t_beta(
            np.arange(1, num_diffusion_timesteps + 1), num_diffusion_timesteps
        )
    elif beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == "cosine":
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        s = 0.008
        steps = num_diffusion_timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
        return betas_clipped
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def q_sample(
    x_0,
    alphas_bar_sqrt,
    one_minus_alphas_bar_sqrt,
    t,
    noise_type="gauss",
    noise_std=1.0,
    reverse=False,
):
    if noise_type == "gauss":
        noise = torch.randn_like(x_0, device=x_0.device) * noise_std
    elif noise_type == "bernoulli":
        noise = (torch.bernoulli(torch.ones_like(x_0) * 0.5) * 2 - 1.0) * noise_std
    else:
        raise NotImplementedError(noise_type)

    if reverse:
        # use alpha_bar_sqrt where t == 0 rest one_minus_alphas_bar_sqrt
        alphas_t_sqrt = torch.where(t == 0, alphas_bar_sqrt[t], one_minus_alphas_bar_sqrt[t]).view(
            -1, 1, 1, 1
        )
        one_minus_alphas_bar_t_sqrt = torch.where(
            t == 0, one_minus_alphas_bar_sqrt[t], alphas_bar_sqrt[t]
        ).view(-1, 1, 1, 1)
    else:
        alphas_t_sqrt = alphas_bar_sqrt[t].view(-1, 1, 1, 1)
        one_minus_alphas_bar_t_sqrt = one_minus_alphas_bar_sqrt[t].view(-1, 1, 1, 1)

    x_t = alphas_t_sqrt * x_0 + one_minus_alphas_bar_t_sqrt * noise
    return x_t


def q_sample_c(
    x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, noise_type="gauss", noise_std=1.0
):
    batch_size, num_channels, _, _ = x_0.shape
    if noise_type == "gauss":
        noise = torch.randn_like(x_0, device=x_0.device) * noise_std
    elif noise_type == "bernoulli":
        noise = (torch.bernoulli(torch.ones_like(x_0) * 0.5) * 2 - 1.0) * noise_std
    else:
        raise NotImplementedError(noise_type)
    alphas_t_sqrt = alphas_bar_sqrt[t].view(batch_size, num_channels, 1, 1)
    one_minus_alphas_bar_t_sqrt = one_minus_alphas_bar_sqrt[t].view(batch_size, num_channels, 1, 1)
    x_t = alphas_t_sqrt * x_0 + one_minus_alphas_bar_t_sqrt * noise
    return x_t


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Diffusion(torch.nn.Module):
    def __init__(
        self,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=2e-2,
        t_min=10,
        t_max=1000,
        noise_std=0.05,
        aug="no",
        ada_maxp=None,
        ts_dist="priority",
        reverse=False,
    ):
        super().__init__()
        self.p = 0.0  # Overall multiplier for augmentation probability.
        self.aug_type = aug
        self.ada_maxp = ada_maxp
        self.noise_type = self.base_noise_type = "gauss"
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.t_min = t_min
        self.t_max = t_max
        self.t_add = int(t_max - t_min)
        self.ts_dist = ts_dist
        self.reverse = reverse

        # Image-space corruptions.
        self.noise_std = float(noise_std)  # Standard deviation of additive RGB noise.
        self.noise_type = "gauss"
        # Skipping AdaAugment for now: TODO: add it later
        if aug == "diff":
            self.aug = DiffAugment()
        else:
            self.aug = Identity()

        # called at initialization to set t_epl values
        self.update_T()

    def set_diffusion_process(self, t, beta_schedule):
        betas = get_beta_schedule(
            beta_schedule=beta_schedule,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            num_diffusion_timesteps=t,
        )

        betas = self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = betas.shape[0]

        alphas = self.alphas = 1.0 - betas
        alphas_cumprod = torch.cat([torch.tensor([1.0]), alphas.cumprod(dim=0)])

        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)

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
        self.t_epl = np.zeros(64, dtype=np.int)
        diffusion_ind = 32
        t_diffusion = np.zeros((diffusion_ind,)).astype(np.int)
        if self.ts_dist == "priority":
            prob_t = np.arange(t) / np.arange(t).sum()
            t_diffusion = np.random.choice(np.arange(1, t + 1), size=diffusion_ind, p=prob_t)
        elif self.ts_dist == "uniform":
            t_diffusion = np.random.choice(np.arange(1, t + 1), size=diffusion_ind)
        self.t_epl[:diffusion_ind] = t_diffusion

    def sample_t(self, batch_size):
        return torch.from_numpy(np.random.choice(self.t_epl, size=batch_size, replace=True))

    def forward(self, x_0, t):
        x_0 = self.aug(x_0)
        assert isinstance(x_0, torch.Tensor) and x_0.ndim == 4
        device = x_0.device

        alphas_bar_sqrt = self.alphas_bar_sqrt.to(device)
        one_minus_alphas_bar_sqrt = self.one_minus_alphas_bar_sqrt.to(device)

        t = t.to(device)
        x_t = q_sample(
            x_0,
            alphas_bar_sqrt,
            one_minus_alphas_bar_sqrt,
            t,
            noise_type=self.noise_type,
            noise_std=self.noise_std,
            reverse=self.reverse,
        )
        return x_t, t.view(-1, 1)

    def __repr__(self):
        return (
            f"Diffusion(\n"
            f"beta_schedule={self.beta_schedule},\n"
            f"beta_start={self.beta_start},\n"
            f"beta_end={self.beta_end},\n"
            f"t_min={self.t_min},\n"
            f"t_max={self.t_max},\n"
            f"noise_std={self.noise_std},\n"
            f"aug={self.aug_type},\n"
            f"ada_maxp={self.ada_maxp},\n"
            f"ts_dist={self.ts_dist},\n"
            f"reverse={self.reverse}"
            ")"
        )
