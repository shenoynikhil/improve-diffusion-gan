"""WACGAN_GP Implementation built on top of Vanilla_ACGAN"""

import torch
import torch.nn.functional as F

from .utils import compute_metrics
from .vanilla_acgan import Vanilla_ACGAN


class WACGAN_GP(Vanilla_ACGAN):
    """WACGAN_GP Implementation using WACGAN_GP"""

    def __init__(self, lambda_term: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_term = lambda_term

        assert hasattr(self.generator, "channels"), "Generator must have channels attribute"
        self.channels = self.generator.channels

    def generator_loss(self, gen_img_scores, gen_label_preds, real_labels):
        """Generator loss for WACGAN_GP"""
        # only consider top-k critic scores for loss if top_k_critic > 0
        if self.top_k_critic > 0:
            valid_gen_pred, indices = torch.topk(gen_img_scores, self.initial_k, dim=0)
            img_loss = -torch.mean(valid_gen_pred)
            aux_loss = F.cross_entropy(
                gen_label_preds[indices.squeeze()], real_labels[indices.squeeze()]
            )
        else:
            img_loss = -torch.mean(gen_img_scores)
            aux_loss = F.cross_entropy(gen_label_preds, real_labels)
        return img_loss + self.lambda_aux_loss * aux_loss

    def discriminator_loss(
        self,
        imgs,
        gen_imgs,
        real_img_scores,
        gen_img_scores,
        real_label_scores,
        gen_label_scores,
        real_labels,
        gen_labels,
    ):
        """Discriminator loss for WACGAN_GP"""
        # Compute Gradient Penalty
        gradient_penalty = self.compute_gradient_penalty(imgs.data, gen_imgs.data)

        img_loss = torch.mean(gen_img_scores) - torch.mean(real_img_scores)
        aux_loss = F.cross_entropy(real_label_scores, real_labels) + F.cross_entropy(
            gen_label_scores, gen_labels
        )

        return img_loss + self.lambda_aux_loss * aux_loss + self.lambda_term * gradient_penalty

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, real_labels = batch
        batch_size = imgs.size(0)

        # sets to same device as imgs
        valid = torch.ones(batch_size, 1).type_as(imgs)
        fake = torch.zeros(batch_size, 1).type_as(imgs)

        # generate noise
        z = torch.normal(0, 1, (batch_size, self.latent_dim)).type_as(imgs)
        gen_labels = torch.randint(0, self.n_classes, (batch_size,)).type_as(real_labels)

        # Generate a batch of images
        gen_imgs = self.generator(z, gen_labels)

        # construct step output
        step_output = {"gen_imgs": gen_imgs}

        # Peform diffusion if module present
        if self.diffusion_module is not None:
            imgs, gen_imgs = self.perform_diffusion_ops(imgs, gen_imgs, batch_idx)

        # train generator
        if optimizer_idx == 0:
            # Loss measures generator's ability to fool the discriminator
            gen_img_scores, pred_labels = self.discriminator(gen_imgs)
            g_loss = self.generator_loss(gen_img_scores, pred_labels, gen_labels)

            # log generator loss
            self.log("g_loss", g_loss, prog_bar=True)

            # update step_output
            step_output["loss"] = g_loss

            return step_output

        # train discriminator
        if optimizer_idx == 1:
            # Loss for real images
            real_img_scores, real_label_scores = self.discriminator(imgs)
            gen_img_scores, gen_label_scores = self.discriminator(gen_imgs)

            # Compute discriminator loss
            d_loss = self.discriminator_loss(
                imgs,
                gen_imgs,
                real_img_scores,
                gen_img_scores,
                real_label_scores,
                gen_label_scores,
                real_labels,
                gen_labels,
            )

            # log discriminator loss
            self.log("d_loss", d_loss, prog_bar=True)

            # compute metrics
            metrics = compute_metrics(
                real_img_scores,
                gen_img_scores,
                real_label_scores,
                gen_label_scores,
                valid,
                fake,
                real_labels,
                gen_labels,
                apply_sigmoid=True,
                multi_label=True,
            )

            self.log_dict(metrics, prog_bar=True)

            # update step_output
            step_output["loss"] = d_loss

            return step_output
