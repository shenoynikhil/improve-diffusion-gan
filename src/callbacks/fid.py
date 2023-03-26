"""Callback to Compute the Frechet Inception Distance between real and generated images"""
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import Metric
from torchmetrics.image.fid import FrechetInceptionDistance


def normalize(x):
    return (255 * (x - x.min()) / (x.max() - x.min())).type(torch.uint8)


# FID Computation Callback
class FID(pl.Callback):
    """Callback to Compute the Frechet Inception Distance between real and generated images"""

    def __init__(self, feature: int = 2048):
        super().__init__()
        self.feature = feature
        assert self.feature in [
            64,
            192,
            768,
            2048,
        ], f"Feature size {feature} inputted is not supported"

        # initialize metric
        self.fid: Metric = FrechetInceptionDistance(feature=self.feature, reset_real_features=False)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Compute FID Score at the end of epoch"""
        # retrieve generated and real images from outputs, batch respectively
        gen_imgs = outputs[0]["gen_imgs"].detach()
        real_imgs = batch[0].detach()

        # # both images need to be converted to uint8 and values between 0 and 255
        gen_imgs = normalize(gen_imgs)
        real_imgs = normalize(real_imgs)

        # move to correct device
        device = pl_module.device
        self.fid.to(device=device)

        # generate two slightly overlapping image intensity distributions
        self.fid.update(real_imgs.to(device), real=True)
        self.fid.update(gen_imgs.to(device), real=False)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Compute FID Score at the end of epoch"""
        pl_module.log("epoch-fid", self.fid.compute(), on_epoch=True, on_step=False)
