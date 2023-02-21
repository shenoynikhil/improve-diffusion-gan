"""LightningModule to setup WACGAN setup.
"""
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn

from .acgan import ACGAN
from .gan_utils import compute_metrics_no_aux


class Generator(nn.Module):
    """Generator Framework for WGAN-GP"""

    def __init__(self, latent_dim: int, channels: int):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.latent_dim = latent_dim
        self.channels = channels

        self.main_module = nn.Sequential(
            # Z latent vector 100
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
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),
            # State (512x8x8)
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),
            # State (256x16x16)
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


class Discriminator(torch.nn.Module):
    """Discriminator for WACGAN-GP"""

    def __init__(self, channels: int):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training
            # objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with
            # respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization -->
            # using per instance normalization nn.InstanceNorm2d() Image (Cx32x32)
            nn.Conv2d(
                in_channels=channels,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            # nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State (256x16x16)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            # nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State (512x8x8)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            # nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # output of main module --> State (1024x4x4)
        self.fc = nn.Sequential(nn.Linear(512, 128), nn.LeakyReLU(0.2, inplace=True))

        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = self.main_module(x)
        x = x.view(x.size()[0], -1).flatten(1)
        x = self.fc(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)


class WGAN_GP(ACGAN):
    """WGAN_GP Network"""

    def __init__(
        self,
        generator,
        discriminator,
        critic_iter: int = 5,
        lambda_term: int = 10,
        lr: float = 0.002,
        output_dir: str = None,
    ):
        pl.LightningModule.__init__(self)
        self.generator = generator
        self.discriminator = discriminator

        # signifies how many iterations disciminator is optimized rel to generator
        self.critic_iter = critic_iter
        # used to multiply this with gradient clipping value
        self.lambda_term = lambda_term

        # set latent_dim
        assert hasattr(self.generator, "latent_dim") and hasattr(
            self.generator, "channels"
        ), "Generator must have attribute latent_dim and channels"
        self.latent_dim = self.generator.latent_dim
        self.channels = self.generator.channels

        # check output dir for saving generated images
        self.output_dir = os.path.join(output_dir, "gen_images")
        os.makedirs(self.output_dir, exist_ok=True)

        self.lr = lr

    def forward(self, z):
        return self.generator(z)

    def configure_optimizers(self):
        """Configure Optimizers"""
        optimizer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.lr,
        )
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr,
        )
        # ensure optimizer_d frequency is self.critic_iter times more
        return (
            {"optimizer": optimizer_g, "frequency": 1},
            {"optimizer": optimizer_d, "frequency": self.critic_iter},
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        """Describes the Training Step / Forward Pass of a WACGAN with Gradient Clipping"""
        imgs, _ = batch
        batch_size = imgs.size(0)

        # sets to same device as imgs
        valid = torch.ones(batch_size, 1).type_as(imgs)
        fake = torch.zeros(batch_size, 1).type_as(imgs)

        # generate images, will be used in both optimizer (generator and discriminator) updates
        z = torch.randn((batch_size, self.latent_dim, 1, 1)).type_as(imgs)
        # get generated images
        gen_imgs = self.generator(z)

        # this will be updated with loss and returned
        step_output = {"gen_imgs": gen_imgs.detach().cpu()}

        # train generator
        if optimizer_idx == 0:
            # Loss measures generator's ability to fool the discriminator
            gen_pred = self.discriminator(gen_imgs)

            # gen_pred_loss by mean across batch dim, shape => (batch_size, 1)
            gen_pred_loss = torch.mean(gen_pred)

            # compute g_loss
            g_loss = -gen_pred_loss / 2

            # update storage and logs with generator loss
            self.log("g_loss", g_loss, prog_bar=True)

            # update step_output with loss
            step_output["loss"] = g_loss
            return step_output

        # train discriminator
        if optimizer_idx == 1:

            # Loss for real/fake images
            # 1. discriminator forward pass on imgs/gen_imgs
            # 2. real/fake_pred_loss by mean across batch dim, shape => (batch_size, 1)
            # 3. loss = (gen_pred_loss - realpred_loss) / 2

            # For real imgs
            real_pred = self.discriminator(imgs)
            real_pred_loss = torch.mean(real_pred)
            real_loss = -real_pred_loss / 2

            # For gen imgs
            fake_pred = self.discriminator(gen_imgs)
            fake_pred_loss = torch.mean(fake_pred)
            fake_loss = fake_pred_loss / 2

            # Compute Gradient Penalty
            gradient_penalty = self.compute_gradient_penalty(imgs.data, gen_imgs.data)

            # total loss
            d_loss = fake_loss + real_loss + self.lambda_term * gradient_penalty

            # update step_output with loss
            step_output["loss"] = d_loss

            # compute metrics
            metrics = compute_metrics_no_aux(
                real_pred,
                fake_pred,
                valid,
                fake,
                apply_sigmoid=True,
            )

            # update storage with metrics
            for metric, metric_val in metrics.items():
                self.storage[metric].append(metric_val)

            self.log_dict(metrics, prog_bar=True)
            return step_output

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_samples.size(0), self.channels, 1, 1)).to(self.device)

        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates = interpolates.to(self.device)

        # calculate probability of interpolated examples
        d_interpolates = self.discriminator(interpolates)

        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def generate_images(self, batch_size: int):
        """Generate Images function"""
        with torch.no_grad():
            return self(
                torch.randn((batch_size, self.latent_dim, 1, 1)).to(self.device),
            )
