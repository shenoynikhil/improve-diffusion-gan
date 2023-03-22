"""LightningModule to setup VanillaGAN setup"""

import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from .diffusion import Diffusion
from .utils import compute_metrics_no_aux, sample_image


class Generator(nn.Module):
    """Generator Framework for GAN"""

    def __init__(self, latent_dim: int, channels: int, img_size: int = 32):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        assert img_size == 32 or img_size == 28, "Final size must be 32 or 28"
        self.latent_dim = latent_dim
        self.channels = channels
        self.img_size = img_size

        self.main_module = nn.Sequential(
            # Z latent vector 100
            # (1 - 1) * 1 + 1 * (4 - 1) + 1 = 4 -> (b, 1024, 4, 4) for img_size = 32/28
            nn.ConvTranspose2d(
                in_channels=latent_dim,
                out_channels=1024,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),
            # State (1024x4x4)
            # (4 - 1) * 2 - 2 * 1 + 1 * (4 - 1) + 1 = 8 -> (b, 512, 8, 8) for img_size = 32
            # (4 - 1) * 2 - 2 * 1 + 1 * (3 - 1) + 1 = 7 -> (b, 512, 7, 7) for img_size = 28
            nn.ConvTranspose2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=4 if img_size == 32 else 3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),
            # State (512x8x8)
            # (8 - 1) * 2 - 2 * 1 + 1 * (4 - 1) + 1 = 16 -> (b, 256, 16, 16) for img_size = 32
            # (7 - 1) * 2 - 2 * 1 + 1 * (4 - 1) + 1 = 14 -> (b, 256, 14, 14) for img_size = 28
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),
            # State (256x16x16)
            # (16 - 1) * 2 - 2 * 1 + 1 * (4 - 1) + 1 = 32 -> (b, 128, 32, 32)
            # (14 - 1) * 2 - 2 * 1 + 1 * (4 - 1) + 1 = 28 -> (b, 128, 28, 28)
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )
        # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, z):
        return self.output(self.main_module(z))


class Discriminator(nn.Module):
    """Discriminator for WACGAN-GP"""

    def __init__(self, channels: int, conv_channel_list: List[int] = [128, 256, 512]):
        """Initialize the Discriminator

        Parameters
        ----------
        channels : int
            Number of channels in the input image
        conv_channel_list : List[int], optional, default=[128, 256, 512]
            List of input_channel, output_channel for each convolutional layer.
            if len(conv_channel_list) = 3, then the discriminator will have
            3 convolutional layers.
            Ensure len(conv_channel_list) < 5 (kernel size = 4
            can only have 4 conv layers)
        """
        super().__init__()
        assert len(conv_channel_list) < 5, "With kernel size = 4, max 4 conv layers"
        channel_list = [channels] + conv_channel_list
        conv_list = []
        for i in range(len(channel_list) - 1):
            conv_list.extend(
                [
                    nn.Conv2d(
                        in_channels=channel_list[i],
                        out_channels=channel_list[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        # for gradient penalty based training, Batch Norm should not be there
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training
            # objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with
            # respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization -->
            # using per instance normalization nn.InstanceNorm2d() Image (Cx32x32)
            *conv_list,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # output of main module --> State (1024x4x4)
        self.fc = nn.Sequential(nn.Linear(channel_list[-1], 128), nn.LeakyReLU(0.2, inplace=True))

        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = self.main_module(x)
        x = x.view(x.size()[0], -1).flatten(1)
        x = self.fc(x)
        return self.output(x)


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
    output_dir: str
        Directory to save generated images
    disc_iters: int
        Number of discriminator iterations per generator iteration
    # top_k training [Optional]
    top_k_critic: int
        Number of top critic predictions to consider
        for top-k training
    # Diffusion Module and related args [Optional]
    diffusion_module: Diffusion
        Diffusion Module for DiffionGAN based training
    ada_interval: int
        Interval for adaptive diffusion rate
    """

    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        output_dir: str,
        lr: float = 0.0001,
        disc_iters: int = 5,
        # top_k training
        top_k_critic: int = 0,
        # Diffusion Module and related args
        diffusion_module: Diffusion = None,
        ada_interval: int = 4,
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.disc_iters = disc_iters

        assert hasattr(generator, "latent_dim"), "Generator must have latent_dim attribute"
        self.latent_dim = generator.latent_dim

        # check output dir for saving generated images
        self.output_dir = output_dir
        self.lr = lr

        # Modification for top-K training
        self.top_k_critic = top_k_critic
        self.initial_k = top_k_critic  # will be reduced as training progresses

        # Diffusion Module
        self.diffusion_module = diffusion_module
        if self.diffusion_module is not None:
            assert isinstance(
                self.diffusion_module, Diffusion
            ), "diffusion_module must be an instance of Diffusion"
            self.ada_interval = ada_interval

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

    def discriminator_loss(self, y_hat, y):
        """Binary Cross Entropy loss between y_hat and y"""
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def generator_loss(self, y_hat, y):
        """Binary Cross Entropy loss with Logits between y_hat and y"""
        if self.top_k_critic > 0:
            valid_top_k, indices = torch.topk(y_hat, self.top_k_critic, dim=0)
            return F.binary_cross_entropy_with_logits(valid_top_k, y[indices.squeeze()])

        return F.binary_cross_entropy_with_logits(y_hat, y)

    def training_step_end(self, step_output):
        """Perform Diffusion Module update after each training step"""
        if self.diffusion_module is not None and self.global_step % self.ada_interval == 0:
            self.diffusion_module.update_T()

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

        # Peform diffusion if module present
        if self.diffusion_module is not None:
            t = self.diffusion_module.sample_t(batch_size)
            imgs, _ = self.diffusion_module(imgs, t)
            gen_imgs, _ = self.diffusion_module(gen_imgs, t)

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
            d_real_loss = self.discriminator_loss(real_pred, valid)

            fake_pred = self.discriminator(gen_imgs)
            d_fake_loss = self.discriminator_loss(fake_pred, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True)
            step_output["loss"] = d_loss

            # compute metrics
            metrics = compute_metrics_no_aux(
                real_pred,
                fake_pred,
                valid,
                fake,
            )

            self.log_dict(metrics, prog_bar=True)
            return step_output

    def training_epoch_end(self, outputs):
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

        # update self.initial_k if needed
        if self.top_k_critic > 0:
            # 75% of batch size is the minimum value possible
            min_value_possible = int(0.75 * self.trainer.datamodule.batch_size)
            self.initial_k = max(min_value_possible, 0.99 * self.initial_k)
            # log value of k
            self.log("k", self.initial_k, prog_bar=True)

    def generate_images(self, batch_size: int):
        with torch.no_grad():
            return self(
                torch.normal(0, 1, (batch_size, self.latent_dim, 1, 1)).to(self.device),
            )
