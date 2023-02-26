"""LightningModule to setup WACGAN setup.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn

from .acgan import ACGAN
from .wgan import WGAN_GP
from .utils import compute_metrics_no_aux
from .noiseAdder import GNadder
import os

class WGAN_NOISE(WGAN_GP):
    """WGAN_GP Network"""

    def __init__(
        self,
        generator,
        discriminator,
        noise_adder=None,
        critic_iter: int = 5,
        lambda_term: int = 10,
        lr: float = 0.002,
        output_dir: str = None,
    ):
        super().__init__(generator, discriminator, critic_iter, lambda_term, lr, output_dir)
        
        if noise_adder is None:
            self.noise_adder = GNadder()
        else:
            self.noise_adder = noise_adder
        

        
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
            # 0. apply generator noise
            # 1. discriminator forward pass on imgs/gen_imgs
            # 2. real/fake_pred_loss by mean across batch dim, shape => (batch_size, 1)
            # 3. loss = (gen_pred_loss - realpred_loss) / 2

            # For real imgs
            imgs = self.noise_adder.apply_noise(imgs, self)
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

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        self.noise_adder.log_noise(self)