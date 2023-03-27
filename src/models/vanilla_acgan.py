"""Vanilla_ACGAN Implementation built on top of VanillaGAN"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import Diffusion
from .utils import compute_metrics, weights_init_normal
from .vanilla_gan import VanillaGAN


class ConditionalGenerator(nn.Module):
    """Conditional Generator
    Generates images given a latent vector and a label
    """

    def __init__(
        self,
        latent_dim: int,
        channels: int,
        img_size: int,
        n_classes: int,
    ):
        super(ConditionalGenerator, self).__init__()
        assert img_size == 32 or img_size == 28, "Final size must be 32 or 28"
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.channels = channels

        # create label embedding
        self.label_emb = nn.Embedding(n_classes, latent_dim)

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

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class ConditionalDiscriminator(nn.Module):
    """Conditional Discriminator
    Critic that takes an image and a label as input
    and provides a real or fake image prediction (2 classes)
    and a label prediction (n_classes)
    """

    def __init__(self, channels: int, img_size: int, n_classes: int):
        super(ConditionalDiscriminator, self).__init__()

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
        ds_size = img_size // 2**4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size**2, 1))
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size**2, n_classes))

        # intialize with normal weights
        self.apply(weights_init_normal)

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        # output real/fake labels (validity) and the digit labels (0-9)
        return validity, label


class Vanilla_ACGAN(VanillaGAN):
    """ACGAN Implementation using ACGAN

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
        generator: nn.Module,
        discriminator: nn.Module,
        output_dir: str,
        lr: float = 0.0001,
        disc_iters: int = 1,
        lambda_aux_loss: float = 0.5,
        # top_k training
        top_k_critic: int = 0,
        # Diffusion Module and related args
        diffusion_module: Diffusion = None,
        ada_interval: int = 4,  # from original code
        ada_target: float = 0.6,  # from original code
        ada_kimg: int = 100,  # from original code
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.disc_iters = disc_iters
        self.lambda_aux_loss = lambda_aux_loss

        assert hasattr(generator, "latent_dim") and hasattr(
            self.generator, "n_classes"
        ), "Generator must have latent_dim attribute and n_classes attribute"
        self.latent_dim = generator.latent_dim
        self.n_classes = generator.n_classes

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
            self.ada_target = ada_target
            self.ada_kimg = ada_kimg

    def forward(self, z, labels):
        return self.generator(z, labels)

    def generator_loss(self, gen_img_scores, gen_label_preds, real_labels):
        """Generator loss"""
        device = gen_img_scores.device
        valid = torch.ones(gen_img_scores.size(0), 1).to(device)

        # only consider top-k critic scores for loss if top_k_critic > 0
        if self.top_k_critic > 0:
            valid_top_k, indices = torch.topk(gen_img_scores, self.initial_k, dim=0)
            return F.binary_cross_entropy_with_logits(
                valid_top_k, valid[indices.squeeze()]
            ) + self.lambda_aux_loss * F.cross_entropy(
                gen_label_preds[indices.squeeze()], real_labels[indices.squeeze()]
            )

        return (
            F.binary_cross_entropy_with_logits(gen_img_scores, valid)
            + self.lambda_aux_loss * F.cross_entropy(gen_label_preds, real_labels)
        ) / 2

    def discriminator_loss(
        self,
        real_img_scores,
        gen_img_scores,
        real_label_scores,
        gen_label_scores,
        real_labels,
        gen_labels,
    ):
        device = real_img_scores.device
        real = torch.ones(real_img_scores.size(0), 1).to(device)
        gen = torch.zeros(gen_img_scores.size(0), 1).to(device)

        return (
            # real img scores loss
            F.binary_cross_entropy_with_logits(real_img_scores, real)
            +
            # real label scores loss
            F.cross_entropy(real_label_scores, real_labels)
            +
            # gen img scores loss
            F.binary_cross_entropy_with_logits(gen_img_scores, gen)
            +
            # gen label scores loss
            F.cross_entropy(gen_label_scores, gen_labels)
        ) / 4

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

    def generate_images(self, batch_size: int):
        n_cls = self.n_classes
        with torch.no_grad():
            return self(
                torch.normal(0, 1, (batch_size, self.latent_dim)).to(self.device),
                torch.randint(0, n_cls, (batch_size,)).to(self.device),
            )
