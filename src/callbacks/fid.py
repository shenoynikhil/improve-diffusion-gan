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
        self.fid = FrechetInceptionDistance(feature=self.feature, reset_real_features=True)

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
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
        gen_imgs = normalize(gen_imgs)
        real_imgs = normalize(real_imgs)

        # generate two slightly overlapping image intensity distributions
        self.fid.update(real_imgs, real=True)
        self.fid.update(gen_imgs, real=False)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.log("epoch-fid", self.fid.compute())
