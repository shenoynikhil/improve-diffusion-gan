from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics.image.inception import InceptionScore

from .fid import normalize


class IS(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.inscep = InceptionScore()
        self.gen_imgs = None

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.gen_imgs = outputs[0]["gen_imgs"].detach().cpu()
        self.gen_imgs = normalize(self.gen_imgs)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.inscep.update(self.gen_imgs)
        pl_module.log("epoch-is", self.inscep.compute()[0])
