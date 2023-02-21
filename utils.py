"""Utils for running the experiments"""
import os
from os.path import join

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar


def weights_init_normal(m):
    """Initialize the weights normally"""
    if isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def check_config(opt: dict):
    """Checks if the config is valid"""
    # check if output directory exists
    assert "output_dir" in opt, "Output directory not specified in config"
    if not os.path.exists(opt["output_dir"]):
        os.makedirs(opt["output_dir"])

    # check if the following keys exist
    keys = ["datamodule", "trainer", "model"]
    for key in keys:
        assert key in opt, f"{key} not specified in config"

    # check if datamodule key exists
    datamodule_cfg = opt.get("datamodule")
    # check if the following keys exist
    keys = ["data_type"]
    for key in keys:
        assert key in datamodule_cfg, f"{key} not specified in config"

    # check model config
    model_cfg = opt.get("model")
    # check if the following keys exist
    keys = ["network", "optimizer"]
    for key in keys:
        assert key in model_cfg, f"{key} not specified in config"


def get_trainer(
    opt,
    network_type: str,
    max_epochs: int = 1,
    ckpt_saving: bool = True,
    early_stopping: bool = False,
    addn_callbacks: list = [],
    every_n_epochs: int = 50,
):
    """returns pytorch lightning training

    Parameters
    ----------
    opt: dict
        experiment dictionary
    network_type: str
        Used for setting the output_dir
    max_epochs: int
        Epochs to train the model with
    ckpt_saving: bool
        checkpoint saving callback will be added
    early_stopping: bool
        Will enable early saving callback
    addn_callbacks: list
        will append callbacks to the callback list
    """
    kwargs = {
        "enable_checkpointing": False,
    }
    output_dir_f = join(opt.output_dir, network_type)
    callbacks = [TQDMProgressBar(refresh_rate=20)]
    # if saving activated
    if ckpt_saving:
        callbacks.append(
            ModelCheckpoint(
                filename="{epoch:02d}",
                dirpath=output_dir_f,
                every_n_epochs=every_n_epochs,
                save_top_k=-1,
            )
        )
        del kwargs["enable_checkpointing"]
    if early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val/epoch_loss",
                mode="min",
                patience=max(10, int(0.05 * max_epochs)),
                verbose=True,
                min_delta=0.05,
            )
        )
    # add addn_callbacks if passed
    if len(addn_callbacks) > 0:
        callbacks.extend(addn_callbacks)

    return Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=max_epochs,
        default_root_dir=output_dir_f,
        logger=False,
        callbacks=callbacks,
        **kwargs,
    )
