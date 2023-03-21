"""Spectral Normalization GAN
Implementation from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
"""
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from ..utils import compute_metrics_no_aux, sample_image
from .dg import Discriminator, Generator


class SpectralNormGAN(LightningModule):
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
        lr: float,
        output_dir: str,
        loss_type: str = "wasserstein",
        disc_iters: int = 5,
        top_k_critic: int = 0,
    ):
        super().__init__()
        self.generator: Generator = generator
        self.discriminator: Discriminator = discriminator

        assert hasattr(generator, "latent_dim"), "Generator must have latent_dim attribute"
        self.latent_dim = generator.latent_dim
        self.loss_type = loss_type
        self.disc_iters = disc_iters
        self.top_k_critic = top_k_critic

        # check output dir for saving generated images
        self.output_dir = output_dir

        self.lr = lr

        self.storage = defaultdict(list)

    def configure_optimizers(self):
        """Configure optimizers for generator and discriminator"""
        # instantiate ADAM optimizer
        # TODO: make this more flexible
        optimizer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.lr,
        )
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr,
        )
        return {"optimizer": optimizer_g, "frequency": 1}, {
            "optimizer": optimizer_d,
            "frequency": self.disc_iters,
        }

    def forward(self, z):
        return self.generator(z)

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

    def training_step(self, batch, _, optimizer_idx):
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
            fake_pred = self.discriminator(gen_imgs)

            # compute discriminator loss
            d_loss = self.discriminator_loss(real_pred, fake_pred, valid, fake)

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

    def on_train_epoch_end(self):
        """At the end of training epoch, generate synthetic images"""
        # Get labels ranging from 0 to n_classes for n rows, do this every 10 epochs
        path = os.path.join(self.output_dir, "gen_images")
        os.makedirs(path, exist_ok=True)
        if self.current_epoch % 10 == 0:
            gen_imgs = self.generate_images(batch_size=100)
            sample_image(
                gen_imgs=gen_imgs,
                n_row=10,
                epochs_done=self.current_epoch,
                output_dir=path,
            )

    def generate_images(self, batch_size: int):
        with torch.no_grad():
            return self(
                torch.normal(0, 1, (batch_size, self.latent_dim, 1, 1)).to(self.device),
            )
