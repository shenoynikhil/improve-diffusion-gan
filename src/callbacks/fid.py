"""Callback to Compute the Frechet Inception Distance between real and generated images"""
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics.image.fid import FrechetInceptionDistance


def normalize(x):
    return (255 * (x - x.min()) / (x.max() - x.min())).type(torch.uint8)


# FID Computation Callback
class FID(pl.Callback):
    """Callback to Compute the Frechet Inception Distance between real and generated images"""

    def __init__(self, feature: int = 64, every_k_epochs: int = 50):
        super().__init__()
        self.feature = feature
        assert self.feature in [
            64,
            192,
            768,
            2048,
        ], f"Feature size {feature} inputted is not supported"

        self.every_k_epochs = every_k_epochs

        # initialize metric
        self.fid = FrechetInceptionDistance(feature=self.feature, reset_real_features=False)

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
        # only compute FID score every k epochs
        if trainer.current_epoch % self.every_k_epochs != 0:
            return

        # retrieve generated and real images from outputs, batch respectively
        gen_imgs = outputs[0]["gen_imgs"].detach().cpu()
        real_imgs = batch[0].detach().cpu()

        # # both images need to be converted to uint8 and values between 0 and 255
        gen_imgs = normalize(gen_imgs)
        real_imgs = normalize(real_imgs)

        # generate two slightly overlapping image intensity distributions
        self.fid.update(real_imgs, real=True)
        self.fid.update(gen_imgs, real=False)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Compute FID Score at the end of epoch"""
        # only compute FID score every k epochs
        if trainer.current_epoch % self.every_k_epochs != 0:
            return

        pl_module.log("epoch-fid", self.fid.compute())
