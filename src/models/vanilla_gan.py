"""LightningModule to setup ACGAN setup. Contains instructions about
- Generator and Discriminator
- Optimizer to be used (currently hardcoded)
- training_step given batch and optimizer_idx (whether to optimize generator or discriminator)
"""

import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from .gan_utils import compute_metrics_no_aux, sample_image


class VanillaGAN(LightningModule):
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

    def __init__(self, generator, discriminator, lr, output_dir: str):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

        assert hasattr(generator, "latent_dim"), "Generator must have latent_dim attribute"
        self.latent_dim = generator.latent_dim

        # check output dir for saving generated images
        self.output_dir = output_dir

        self.lr = lr

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
        return [optimizer_g, optimizer_d], []

    def forward(self, z, labels):
        return self.generator(z, labels)

    def adversarial_loss(self, y_hat, y):
        """Binary Cross Entropy loss between y_hat and y"""
        return F.binary_cross_entropy(y_hat, y)

    def on_train_start(self):
        """Created storage to store some metrics throughout training"""
        self.storage = defaultdict(list)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch
        batch_size = imgs.size(0)

        # sets to same device as imgs
        valid = torch.ones(batch_size, 1).type_as(imgs)
        fake = torch.zeros(batch_size, 1).type_as(imgs)

        # generate noise
        z = torch.normal(0, 1, (batch_size, self.latent_dim)).type_as(imgs)

        # Generate a batch of images
        gen_imgs = self.generator(z)

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
                torch.normal(0, 1, (batch_size, self.latent_dim)).to(self.device),
            )
