"""ACGAN architecture"""
import os
from typing import Any, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics.functional as Fm
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image

from utils import weights_init_normal


def generate_and_save_images(
    gan,
    number_of_images: int,
    output_dir: str,
    batch_size: int = 100,
):
    """Generates number_of_images using generator of the GAN"""
    gan.eval()
    os.makedirs(output_dir, exist_ok=True)
    counter = 0
    epochs = int(number_of_images / batch_size)
    for _ in range(epochs):
        gen_imgs = gan.generate_images(batch_size=batch_size)
        # denormalize images, as we normalize images
        gen_imgs = 0.5 + (gen_imgs * 0.5)
        for j in range(gen_imgs.shape[0]):
            save_image(
                gen_imgs[j],
                os.path.join(output_dir, f"{counter}.png"),
            )
            counter += 1


class Generator(nn.Module):
    def __init__(self, opt: dict):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size**2))

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
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
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


class Discriminator(nn.Module):
    def __init__(self, opt: dict):
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
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2**4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size**2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(
            nn.Linear(128 * ds_size**2, opt.n_classes), nn.Softmax(dim=-1)
        )

        # intialize with normal weights
        self.apply(weights_init_normal)

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        # output real/fake labels (validity) and the digit labels (0-9)
        return validity, label


def compute_metrics(
    real_pred,
    fake_pred,
    real_aux,
    fake_aux,
    valid,
    fake,
    labels,
    gen_labels,
    apply_sigmoid: bool = False,
    multi_label: bool = False,
):
    """Utility function to compute a bunch of metrics

    Parameters
    ----------
    real_pred: torch.Tensor
        real/not predictions on real images from discriminator
    fake_pred: torch.Tensor
        real/not predictions on gen images from discriminator
    real_aux: torch.Tensor
        ground-truth label predictions on real images from discriminator
    fake_aux: torch.Tensor
        ground-truth label predictions on gen images from discriminator
    valid: torch.Tensor
        real labels (all 1) real images from discriminator
    fake: torch.Tensor
        fake labels (all 0) real images from discriminator
    labels: torch.Tensor
        ground-truth labels real images from discriminator
    gen_labels: torch.Tensor
        ground-truth labels gen images from discriminator
    apply_sigmoid: bool, default False
        Applies sigmoid on real_pred and fake_pred if raw scores
    multi_label: bool, default False
        In the Multi Label Case, auxillary scores accuracy scores are calculaled differently
    """
    if apply_sigmoid:
        real_pred = torch.sigmoid(real_pred)
        fake_pred = torch.sigmoid(fake_pred)

    # Calculate discriminator accuracy
    pred = np.concatenate(
        [real_pred.data.cpu().numpy(), fake_pred.data.cpu().numpy()], axis=0
    )
    gt = np.concatenate([valid.data.cpu().numpy(), fake.data.cpu().numpy()], axis=0)
    # d_acc = np.mean(np.argmax(pred, axis=1) == gt)
    pred = np.where(pred >= 0.5, 1, 0)
    d_acc = np.mean(pred == gt)

    if not multi_label:
        class_pred = np.concatenate(
            [real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0
        )
        c_gt = np.concatenate(
            [labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0
        )
        d_class_acc = np.mean(np.argmax(class_pred, axis=1) == c_gt)
    else:
        # considering multi label binary case, therefore num_classes = 2
        class_pred = torch.cat([real_aux, fake_aux], axis=0).detach().cpu()
        c_gt = torch.cat([labels, gen_labels], axis=0).detach().cpu()
        class_pred = 1.0 * (np.sig(class_pred) > 0.5)
        d_class_acc = Fm.accuracy(class_pred, c_gt, num_classes=2)

    return {
        "D Accuracy": d_acc * 100,
        "D Class Accuracy": d_class_acc * 100,
    }


def compute_metrics_no_aux(
    real_pred,
    fake_pred,
    valid,
    fake,
    apply_sigmoid: bool = False,
):
    """Compute Basic Metrics"""
    if apply_sigmoid:
        real_pred = torch.sigmoid(real_pred)
        fake_pred = torch.sigmoid(fake_pred)

    # Calculate discriminator accuracy
    pred = np.concatenate(
        [real_pred.data.cpu().numpy(), fake_pred.data.cpu().numpy()], axis=0
    )
    gt = np.concatenate([valid.data.cpu().numpy(), fake.data.cpu().numpy()], axis=0)
    # d_acc = np.mean(np.argmax(pred, axis=1) == gt)
    pred = np.where(pred >= 0.5, 1, 0)
    d_acc = np.mean(pred == gt)

    return {"D Accuracy": d_acc * 100}


def sample_image(gen_imgs, n_row: int, epochs_done: int, output_dir: str) -> None:
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Get labels ranging from 0 to n_classes for n rows and generate images from noise
    save_dir = os.path.join(output_dir, "images/")
    os.makedirs(save_dir, exist_ok=True)
    save_image(
        gen_imgs.data,
        os.path.join(save_dir, f"{epochs_done}.png"),
        nrow=n_row,
        normalize=True,
    )


# FID Computation Callback
class FID(pl.Callback):
    """Callback to Compute the Frechet Inception Distance between real and generated images"""

    def __init__(self, feature: int = 64):
        super().__init__()
        self.feature = feature
        assert self.feature in [
            64,
            192,
            768,
            2048,
        ], f"Feature size {feature} inputted is not supported"

        # initialize metric
        self.fid = FrechetInceptionDistance(
            feature=self.feature, reset_real_features=True
        )

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Check if pl_module is of type ACGAN or WACGAN_GP"""
        class_name = type(pl_module).__name__
        assert class_name in [
            "ACGAN",
            "WACGAN_GP",
        ], f"{class_name} not supported with this Callback"
        assert hasattr(
            pl_module, "generate_images"
        ), f"{class_name} does not have generate_images Callback"

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        """Compute FID Score at the end of epoch"""
        # retrieve generated and real images from outputs, batch respectively
        gen_imgs = outputs["gen_imgs"].detach().cpu()
        real_imgs = batch[0].detach().cpu()

        # # both images need to be converted to uint8 and values between 0 and 255
        f = lambda x: (255 * (x - x.min()) / (x.max() - x.min())).type(torch.uint8)
        gen_imgs = f(gen_imgs)
        real_imgs = f(real_imgs)

        # generate two slightly overlapping image intensity distributions
        self.fid.update(real_imgs, real=True)
        self.fid.update(gen_imgs, real=False)

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        pl_module.log("epoch-fid", self.fid.compute())


class SaveGeneratedImages(pl.Callback):
    """Saves Generated Images: For Training and Testing during MIA Attack
    A Callback that during the training process saves generated images `every_k_epochs` and
    in the last epoch

    Parameters
    ----------
    output_dir: str
        Output directory with respect to where the generated images are stored
    every_k_epochs: int
        Images will be stored every k epochs
    number_of_images: int
        Number of images will be stored
    batch_size: int
        Batch size of generation
    """

    def __init__(
        self,
        output_dir: str,
        every_k_epochs: int = 50,
        number_of_images: int = 1000,
        batch_size: int = 100,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.every_k_epochs = every_k_epochs
        self.number_of_images = number_of_images
        self.batch_size = batch_size
        self.gen_images_dirs = []
        self.gen_test_images_dirs = []

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        current_epoch = pl_module.current_epoch
        condition = (
            (current_epoch % self.every_k_epochs == 0)
            or (current_epoch == trainer.max_epochs - 1)
            and (current_epoch > 0)
        )
        if condition:
            # generate images for training
            gen_images_dir = os.path.join(
                self.output_dir, f"gen_images_{current_epoch}/"
            )
            os.makedirs(gen_images_dir, exist_ok=True)
            generate_and_save_images(
                gan=pl_module,
                number_of_images=self.number_of_images,
                output_dir=os.path.join(gen_images_dir, "0"),
                batch_size=self.batch_size,
            )
            self.gen_images_dirs.append(gen_images_dir)

            # generate images for testing
            gen_test_images_dir = os.path.join(
                self.output_dir, f"gen_test_images_{current_epoch}/"
            )
            os.makedirs(gen_test_images_dir, exist_ok=True)
            generate_and_save_images(
                gan=pl_module,
                number_of_images=self.number_of_images,
                output_dir=os.path.join(gen_test_images_dir, "0"),
                batch_size=self.batch_size,
            )
            self.gen_test_images_dirs.append(gen_test_images_dir)
