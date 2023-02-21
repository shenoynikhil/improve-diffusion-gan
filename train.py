"""Script to run GAN training.
To run this script, choose a suitable experiment config from `experiments/`

```python
python src/train.py --path <path to config>
```
"""
# Authors: Nikhil Shenoy, Matthew Tang

import argparse
import logging
import os
from datetime import datetime
from os.path import join
from pathlib import Path

import pytorch_lightning as pl
import yaml
from attrdict import AttrDict
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.commons import check_config  # noqa: E402

# Set so as to pick checkpoints from the right place
os.environ["TORCH_HOME"] = "/scratch/st-jiaruid-1/shenoy/projects/cpsc533r-project/data/checkpoints"


def main(config: dict):
    # set seed in order to ensure reproducibility
    seed = config.get("seed", 42)
    pl.seed_everything(seed=seed)
    logging.info(f"Starting Experiment with seed: {seed}")

    # get dataloader
    logging.info("Creating Datamodule")
    datamodule = instantiate(config.get("datamodule"))

    # load model
    logging.info("Creating GAN")
    gan = instantiate(config.get("model"))

    # get trainer for gan training
    logging.info("Creating Trainer")
    trainer = instantiate(config.get("trainer"), default_root_dir=config.get("output_dir"))

    # fit the gan
    logging.info("Fitting the GAN")
    trainer.fit(gan, datamodule)


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
    opt = OmegaConf.create(yaml.safe_load(Path(opt.path).read_text()))
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
