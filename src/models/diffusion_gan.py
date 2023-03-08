"""LightningModule for DiffusionGAN
Build upon Vanilla GAN implementation
"""

import torch

from .diffusion import Diffusion
from .utils import compute_metrics_no_aux
from .vanilla_gan import VanillaGAN


class DiffusionGAN(VanillaGAN):
    """VanillaGAN Implementation

    Parameters
    ----------
    generator: nn.Module
        Generator model
    discriminator: nn.Module
        Discriminator model
    lr: float
        Learning rate for the optimizer
    """

    def __init__(
        self,
        generator,
        discriminator,
        lr,
        output_dir: str,
        ada_interval: int = 4,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=2e-2,
        t_min=10,
        t_max=1000,
        noise_std=0.05,
        aug="no",
        ada_maxp=None,
        ts_dist="priority",
    ):
        super().__init__(generator, discriminator, lr, output_dir)
        # intialize diffusion module
        self.diffusion = Diffusion(
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            t_min=t_min,
            t_max=t_max,
            noise_std=noise_std,
            aug=aug,
            ada_maxp=ada_maxp,
            ts_dist=ts_dist,
        )
        # adaptive diffusion rate
        self.ada_interval = ada_interval

        # counter for steps taken
        self.training_step_count = 0

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch
        batch_size = imgs.size(0)

        # update diffusion time steps for adaptive diffusion
        self.training_step_count += 1
        if self.training_step_count % self.ada_interval == 0:
            self.diffusion.update_T()

        # sets to same device as imgs
        valid = torch.ones(batch_size, 1).type_as(imgs)
        fake = torch.zeros(batch_size, 1).type_as(imgs)

        # generate noise
        z = torch.normal(0, 1, (batch_size, self.latent_dim)).type_as(imgs)

        # Generate a batch of images
        gen_imgs = self.generator(z)

        # Diffuse into both real and generated images
        t = self.diffusion.sample_t(batch_size)
        imgs, _ = self.diffusion(imgs, t)
        gen_imgs, _ = self.diffusion(gen_imgs, t)

        # construct step output
        step_output = {"gen_imgs": gen_imgs}

        # train generator
        if optimizer_idx == 0:
            # Loss measures generator's ability to fool the discriminator
            validity = self.discriminator(gen_imgs)
            g_loss = self.adversarial_loss(validity, valid)

            # update storage and logs with generator loss
            self.storage["g_loss"].append(g_loss)
            self.log("g_loss", g_loss, prog_bar=True)

            # update step_output
            step_output["loss"] = g_loss

            return step_output

        # train discriminator
        if optimizer_idx == 1:
            # Loss for real images
            real_pred = self.discriminator(imgs)
            d_real_loss = self.adversarial_loss(real_pred, valid)

            # update storage with discriminator scores on real images
            self.storage["real_scores"].append(torch.mean(real_pred.data.cpu()))

            fake_pred = self.discriminator(gen_imgs)
            d_fake_loss = self.adversarial_loss(fake_pred, fake)

            # update storage with fake scores
            self.storage["fake_scores"].append(torch.mean(fake_pred.data.cpu()))

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            # update storage and logs with discriminator loss
            self.storage["d_loss"].append(d_loss)
            self.log("d_loss", d_loss, prog_bar=True)

            # compute metrics
            metrics = compute_metrics_no_aux(
                real_pred,
                fake_pred,
                valid,
                fake,
            )

            # update storage with metrics
            for metric, metric_val in metrics.items():
                self.storage[metric].append(metric_val)

            self.log_dict(metrics, prog_bar=True)

            # update step_output
            step_output["loss"] = d_loss

            return step_output
