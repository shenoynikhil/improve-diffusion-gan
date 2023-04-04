"""LightningModule to setup WGAN_GP setup.
"""

from typing import List

import torch
import torch.nn as nn

from .utils import compute_metrics_no_aux
from .vanilla_gan import VanillaGAN


class Generator(nn.Module):
    """Generator Framework for WGAN-GP"""

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


class Discriminator(torch.nn.Module):
    """Discriminator for WGAN-GP"""

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


class WGAN_GP(VanillaGAN):
    """WGAN_GP Network"""

    def __init__(self, lambda_term: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_term = lambda_term
        self.channels = self.generator.channels

    def training_step(self, batch, batch_idx, optimizer_idx):
        """Describes the Training Step / Forward Pass of a WGAN with Gradient Penalty"""
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

        if self.diffusion_module is not None:
            imgs, gen_imgs, _, _ = self.perform_diffusion_ops(imgs, gen_imgs, batch_idx)

        # train generator
        if optimizer_idx == 0:
            # Loss measures generator's ability to fool the discriminator
            gen_pred = self.discriminator(gen_imgs)

            # gen_pred_loss by mean across batch dim, shape => (batch_size, 1)
            if self.top_k_critic > 0:
                valid_gen_pred, _ = torch.topk(gen_pred, self.initial_k, dim=0)
                g_loss = -torch.mean(valid_gen_pred)
            else:
                g_loss = -torch.mean(gen_pred)

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
            # 3. loss = (gen_pred_loss - realpred_loss)

            # For real imgs
            real_pred = self.discriminator(imgs)
            real_pred_loss = torch.mean(real_pred)
            real_loss = -real_pred_loss

            # For gen imgs
            fake_pred = self.discriminator(gen_imgs)
            fake_pred_loss = torch.mean(fake_pred)
            fake_loss = fake_pred_loss

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
