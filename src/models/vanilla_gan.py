"""LightningModule to setup ACGAN setup. Contains instructions about
- Generator and Discriminator
- Optimizer to be used (currently hardcoded)
- training_step given batch and optimizer_idx (whether to optimize generator or discriminator)
"""

import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from .utils import compute_metrics_no_aux, sample_image, weights_init_normal


class Generator(nn.Module):
    def __init__(self, img_size: int, channels: int, latent_dim: int = 100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

        # intialize with normal weights
        self.apply(weights_init_normal)

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, channels: int, img_size: int):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        if img_size == 28:
            ds_size = 2
        else:
            # useful for imgs with size mod 2 == 0
            ds_size = img_size // 2**4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size**2, 1), nn.Sigmoid())

        # intialize with normal weights
        self.apply(weights_init_normal)

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        # output real/fake labels (validity)
        return validity


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
        return [optimizer_g, optimizer_d], []

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        """Binary Cross Entropy loss between y_hat and y"""
        return F.binary_cross_entropy(y_hat, y)

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
