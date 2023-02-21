"""LightningModule to setup ACGAN setup. Contains instructions about
- Generator and Discriminator
- Optimizer to be used (currently hardcoded)
- training_step given batch and optimizer_idx (whether to optimize generator or discriminator)
"""

from collections import defaultdict

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from .gan_utils import compute_metrics, sample_image


class ACGAN(LightningModule):
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

    def __init__(self, generator, discriminator, lr):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
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

    def auxiliary_loss(self, y_hat, y):
        """Cross Entropy loss between y_hat and y"""
        return F.cross_entropy(y_hat, y)

    def on_train_start(self):
        """Created storage to store some metrics throughout training"""
        self.storage = defaultdict(list)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, labels = batch
        batch_size = imgs.size(0)

        # sets to same device as imgs
        valid = torch.ones(batch_size, 1).type_as(imgs)
        fake = torch.zeros(batch_size, 1).type_as(imgs)

        # generate noise
        z = torch.normal(0, 1, (batch_size, self.opt.latent_dim)).type_as(imgs)
        gen_labels = torch.randint(0, self.opt.n_classes, (batch_size,)).type_as(labels)

        # Generate a batch of images
        gen_imgs = self.generator(z, gen_labels)

        # construct step output
        step_output = {"gen_imgs": gen_imgs}

        # train generator
        if optimizer_idx == 0:
            # Loss measures generator's ability to fool the discriminator
            validity, pred_label = self.discriminator(gen_imgs)
            g_loss = (
                self.adversarial_loss(validity, valid) + self.auxiliary_loss(pred_label, gen_labels)
            ) / 2

            # update storage and logs with generator loss
            self.storage["g_loss"].append(g_loss)
            self.log("g_loss", g_loss, prog_bar=True)

            # update step_output
            step_output["loss"] = g_loss

            return step_output

        # train discriminator
        if optimizer_idx == 1:
            # Loss for real images
            real_pred, real_aux = self.discriminator(imgs)
            d_real_loss = (
                self.adversarial_loss(real_pred, valid) + self.auxiliary_loss(real_aux, labels)
            ) / 2
            # update storage with discriminator scores on real images
            self.storage["real_scores"].append(torch.mean(real_pred.data.cpu()))

            fake_pred, fake_aux = self.discriminator(gen_imgs)
            d_fake_loss = (
                self.adversarial_loss(fake_pred, fake) + self.auxiliary_loss(fake_aux, gen_labels)
            ) / 2
            # update storage with fake scores
            self.storage["fake_scores"].append(torch.mean(fake_pred.data.cpu()))

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            # update storage and logs with discriminator loss
            self.storage["d_loss"].append(d_loss)
            self.log("d_loss", d_loss, prog_bar=True)

            # compute metrics
            metrics = compute_metrics(
                real_pred,
                fake_pred,
                real_aux,
                fake_aux,
                valid,
                fake,
                labels,
                gen_labels,
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
        if self.current_epoch % 10 == 0:
            gen_imgs = self.generate_images(batch_size=100)
            sample_image(
                gen_imgs=gen_imgs,
                n_row=10,
                epochs_done=self.current_epoch,
                output_dir=self.opt.output_dir,
            )

    def generate_images(self, batch_size: int):
        n_cls = self.opt.n_classes
        with torch.no_grad():
            return self(
                torch.normal(0, 1, (batch_size, self.opt.latent_dim)).to(self.device),
                torch.randint(0, n_cls, (batch_size)).to(self.device),
            )
