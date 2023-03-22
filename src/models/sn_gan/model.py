"""Spectral Normalization GAN
Implementation from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
"""

import torch
import torch.nn.functional as F

from src.models.diffusion import Diffusion
from src.models.vanilla_gan import VanillaGAN

from ..utils import compute_metrics_no_aux
from .dg import Discriminator, Generator


class SpectralNormGAN(VanillaGAN):
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
        generator: Generator,
        discriminator: Discriminator,
        output_dir: str,
        lr: float = 0.0001,
        disc_iters: int = 5,
        # top_k training
        top_k_critic: int = 0,
        # Diffusion Module and related args
        diffusion_module: Diffusion = None,
        ada_interval: int = 4,
        loss_type: str = "wasserstein",
    ):
        super().__init__(
            generator,
            discriminator,
            output_dir,
            lr,
            disc_iters,
            top_k_critic,
            diffusion_module,
            ada_interval,
        )
        self.loss_type = loss_type

    def generator_loss(self, y_hat, y):
        """Binary Cross Entropy loss between y_hat and y"""
        if self.loss_type == "wasserstein":
            if self.top_k_critic > 0:
                y_hat, _ = torch.topk(y_hat, self.top_k_critic, dim=0)
            return -torch.mean(y_hat)
        else:
            if self.top_k_critic > 0:
                y_hat, _ = torch.topk(y_hat, self.top_k_critic, dim=0)
                y = torch.ones_like(y_hat).to(y_hat.device)
            return F.binary_cross_entropy_with_logits(y_hat, y)

    def discriminator_loss(self, real_pred, fake_pred, real_labels, fake_labels):
        if self.loss_type == "wasserstein":
            return -torch.mean(real_pred) + torch.mean(fake_pred)
        else:
            real_loss = F.binary_cross_entropy_with_logits(real_pred, real_labels)
            fake_loss = F.binary_cross_entropy_with_logits(fake_pred, fake_labels)
            return real_loss + fake_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch
        batch_size = imgs.size(0)

        # sets to same device as imgs
        valid = torch.ones(batch_size, 1).type_as(imgs)
        fake = torch.zeros(batch_size, 1).type_as(imgs)

        # generate noise
        z = torch.normal(0, 1, (batch_size, self.latent_dim, 1, 1)).type_as(imgs)

        # Generate a batch of images
        gen_imgs = self.generator(z)

        # construct step output
        step_output = {"gen_imgs": gen_imgs}

        # train generator
        if optimizer_idx == 0:
            # Loss measures generator's ability to fool the discriminator
            validity = self.discriminator(gen_imgs)
            g_loss = self.generator_loss(validity, valid)

            # log generator loss
            self.log("g_loss", g_loss, prog_bar=True)

            # update step_output
            step_output["loss"] = g_loss

            return step_output

        # train discriminator
        if optimizer_idx == 1:
            # Loss for real images
            real_pred = self.discriminator(imgs)
            fake_pred = self.discriminator(gen_imgs)

            # compute discriminator loss
            d_loss = self.discriminator_loss(real_pred, fake_pred, valid, fake)

            # log discriminator loss
            self.log("d_loss", d_loss, prog_bar=True)

            # compute metrics
            metrics = compute_metrics_no_aux(
                real_pred,
                fake_pred,
                valid,
                fake,
            )

            self.log_dict(metrics, prog_bar=True)

            # update step_output
            step_output["loss"] = d_loss

            return step_output
