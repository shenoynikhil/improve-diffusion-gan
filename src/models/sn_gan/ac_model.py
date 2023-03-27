"""Spectral Normalization ACGAN
Implementation from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
with auxillary classifier
"""

import torch
import torch.nn.functional as F

from src.models.vanilla_acgan import Vanilla_ACGAN

from ..utils import compute_metrics


class SpectralNormACGAN(Vanilla_ACGAN):
    """SpectralNormACGAN Implementation

    Parameters
    ----------
    generator: nn.Module
        Generator model
    discriminator: nn.Module
        Discriminator model
    lr: float
        Learning rate for the optimizer
    """

    def __init__(self, loss_type: str = "wasserstein", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type

    def generator_loss(self, gen_img_scores, gen_label_preds, real_labels):
        """Binary Cross Entropy loss between y_hat and y"""
        device = gen_img_scores.device
        valid = torch.ones(gen_img_scores.size(0), 1).to(device)

        if self.loss_type == "wasserstein":
            if self.top_k_critic > 0:
                valid_gen_pred, indices = torch.topk(gen_img_scores, self.initial_k, dim=0)
                img_loss = -torch.mean(valid_gen_pred)
                aux_loss = F.cross_entropy(
                    gen_label_preds[indices.squeeze()], real_labels[indices.squeeze()]
                )
            else:
                img_loss = -torch.mean(gen_img_scores)
                aux_loss = F.cross_entropy(gen_label_preds, real_labels)
        else:
            if self.top_k_critic > 0:
                valid_top_k, indices = torch.topk(gen_img_scores, self.initial_k, dim=0)
                img_loss = F.binary_cross_entropy_with_logits(valid_top_k, valid[indices.squeeze()])
                aux_loss = F.cross_entropy(
                    gen_label_preds[indices.squeeze()], real_labels[indices.squeeze()]
                )
            else:
                img_loss = F.binary_cross_entropy_with_logits(gen_img_scores, valid)
                aux_loss = F.cross_entropy(gen_label_preds, real_labels)

        return img_loss + self.lambda_aux_loss * aux_loss

    def discriminator_loss(
        self,
        real_img_scores,
        gen_img_scores,
        real_label_scores,
        gen_label_scores,
        real_labels,
        gen_labels,
    ):
        if self.loss_type == "wasserstein":
            # wasserstein loss
            img_loss = -torch.mean(real_img_scores) + torch.mean(gen_img_scores)
            aux_loss = F.cross_entropy(real_label_scores, gen_labels) + F.cross_entropy(
                gen_label_scores, gen_labels
            )
        else:
            real_img_labels = torch.ones_like(real_img_scores).to(real_img_scores.device)
            gen_img_labels = torch.zeros_like(gen_img_scores).to(gen_img_scores.device)
            img_loss = F.binary_cross_entropy_with_logits(
                real_img_scores, real_img_labels
            ) + F.binary_cross_entropy_with_logits(gen_img_scores, gen_img_labels)
            aux_loss = F.cross_entropy(real_label_scores, real_labels) + F.cross_entropy(
                gen_label_scores, gen_labels
            )

        # return combined loss
        return img_loss + self.lambda_aux_loss * aux_loss

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
            imgs, gen_imgs = self.perform_diffusion_ops(imgs, gen_imgs, batch_idx, auxillary=True)

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
