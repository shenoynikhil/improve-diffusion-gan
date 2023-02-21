"""Script to run entire MIA attack. To run this script, choose a suitable experiment config
from `experiments/`

```python
python run.py --path=experiments/default.py
```
"""
# Authors: Mishaal Kazmi, Nikhil Shenoy

import sys
sys.path.append('.')

import argparse
import logging
import os
from datetime import datetime
from os.path import join
from pathlib import Path

import pytorch_lightning as pl
import torchvision.transforms as transforms
import yaml
from attrdict import AttrDict

from datamodule import BaseDataModule
from models import ACGAN, FID, WGAN_GP, SaveGeneratedImages
from utils import get_trainer, check_config

# Set so as to pick checkpoints from the right place
os.environ[
    "TORCH_HOME"
] = "/scratch/st-jiaruid-1/shenoy/projects/cpsc533r-project/data/checkpoints"


def main(opt: dict):
    # set seed in order to ensure reproducibility
    seed = opt.get("seed", 42)
    pl.seed_everything(seed=seed)
    logging.info(f"Starting Experiment with seed: {seed}")

    # get dataloader
    dm_config = opt.get("datamodule")
    logging.info(f"Creating dataloader using dataset of type: {dm_config.data_type}")
    dm = BaseDataModule(
        data_type=dm_config.data_type,
        transforms=transforms.Compose(
            [
                transforms.Resize((opt.img_size, opt.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        ),
        output_dir=opt.output_dir,
        batch_size=opt.get("batch_size", 100),
        num_workers=opt.get("num_workers", 1),
        channels=opt.get("channels", 3),
        dataset_frac=opt.get("dataset_frac", 1.0),
        multi_label=opt.get('target_multi_label', False),
        seed=seed,
    )

    # --------------------
    # Part 1. GAN Training
    # --------------------
    gan_type = opt.get("gan_type", "ACGAN")
    logging.info(f"Initializing GAN: {gan_type}")

    if gan_type == "ACGAN":
        gan = ACGAN(opt)
    elif gan_type == "WGAN_GP":
        gan = WGAN_GP(opt)
    else:
        raise NotImplementedError

    # get trainer for gan training
    logging.info("Initializing Pytorch Lightning Trainer")
    addn_callbacks = [
        SaveGeneratedImages(
            output_dir=opt.output_dir,
            every_k_epochs=100,
            number_of_images=len(dm.indices_dict["mia_train"]),
            batch_size=100,
        )
    ]
    if opt.get("enable_fid", False):
        addn_callbacks.append(FID(opt.get("fid_feature", 64)))

    trainer = get_trainer(
        opt,
        network_type=gan_type,
        max_epochs=opt.get("n_epochs", 200),
        addn_callbacks=addn_callbacks,
        every_n_epochs=100,
    )

    # get dataloader for gan
    dataloader = dm.get_dataloader_from_indices(net_type="gan")

    # fit the gan
    logging.info("Fitting the GAN")
    trainer.fit(gan, dataloader)


if __name__ == "__main__":
    # parameter settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        help="Path to experiment config",
    )
    opt = parser.parse_args()

    # read from yaml
    opt = yaml.safe_load(Path(opt.path).read_text())
    # check config
    check_config(opt)

    # create output directory
    opt["output_dir"] = join(
        opt.get("output_dir"),
        datetime.now().strftime("%d_%m_%Y-%H_%M"),
    )
    os.makedirs(opt["output_dir"], exist_ok=False)
    
    # setup logging directory
    logging.basicConfig(
        filename=join(opt.get("output_dir"), opt.get("log_dir", "output.log")),
        filemode="a",
        level=logging.INFO,
    )

    main(AttrDict(opt))
