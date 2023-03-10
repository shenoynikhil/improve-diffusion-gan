"""LightningModule to setup DiffusionWGAN setup.
"""
import torch

from .diffusion import Diffusion
from .utils import compute_metrics_no_aux
from .wgan import WGAN_GP


class DiffusionWGAN(WGAN_GP):
    """WGAN_GP Network"""

    def __init__(
        self,
        generator,
        discriminator,
        critic_iter: int = 5,
        lambda_term: int = 10,
        lr: float = 0.002,
        output_dir: str = None,
        ada_interval: int = 4,
        top_k_critic: int = 0,
        diffusion_module: Diffusion = None,
    ):
        super().__init__(generator, discriminator, critic_iter, lambda_term, lr, output_dir)
        # intialize diffusion module
        assert diffusion_module is not None, "Diffusion module cannot be None"
        self.diffusion = diffusion_module

        # adaptive diffusion rate
        self.ada_interval = ada_interval

        # counter for steps taken
        self.training_step_count = 0

        # top k critic predictions to consider
        self.top_k_critic = top_k_critic

    def training_step(self, batch, batch_idx, optimizer_idx):
        """Describes the Training Step / Forward Pass of a WACGAN with Gradient Clipping"""
        imgs, _ = batch
        batch_size = imgs.size(0)

        # update diffusion time steps for adaptive diffusion
        self.training_step_count += 1
        if self.training_step_count % self.ada_interval == 0:
            self.diffusion.update_T()

        # sets to same device as imgs
        valid = torch.ones(batch_size, 1).type_as(imgs)
        fake = torch.zeros(batch_size, 1).type_as(imgs)

        # generate images, will be used in both optimizer (generator and discriminator) updates
        z = torch.randn((batch_size, self.latent_dim, 1, 1)).type_as(imgs)
        # get generated images
        gen_imgs = self.generator(z)

        # this will be updated with loss and returned
        step_output = {"gen_imgs": gen_imgs.detach().cpu()}

        # Diffuse into both real and generated images
        t = self.diffusion.sample_t(batch_size)
        imgs_noised, _ = self.diffusion(imgs, t)
        gen_imgs_noised, _ = self.diffusion(gen_imgs, t)

        # train generator
        if optimizer_idx == 0:
            # Loss measures generator's ability to fool the discriminator
            gen_pred = self.discriminator(gen_imgs_noised)

            # perform top_k if > 0
            if self.top_k_critic > 0:
                valid_gen_pred, _ = torch.topk(gen_pred, self.top_k_critic, dim=0)
                g_loss = -torch.mean(valid_gen_pred) / 2
            else:
                g_loss = -torch.mean(gen_pred) / 2

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
            real_pred = self.discriminator(imgs_noised)
            real_pred_loss = torch.mean(real_pred)
            real_loss = -real_pred_loss / 2

            # For gen imgs
            fake_pred = self.discriminator(gen_imgs_noised)
            fake_pred_loss = torch.mean(fake_pred)
            fake_loss = fake_pred_loss / 2

            # Compute Gradient Penalty
            gradient_penalty = self.compute_gradient_penalty(imgs_noised.data, gen_imgs_noised.data)

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
