"""Script to run GAN training.
To run this script, choose a suitable experiment config from `experiments/`

```python
python train.py experiment=<path to experiment config>
```
"""
# Authors: Nikhil Shenoy, Matthew Tang
from typing import Optional

import hydra
import pyrootutils
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.pylogger import get_pylogger  # noqa: E402

log = get_pylogger(__name__)


def train(config: dict):
    # set seed in order to ensure reproducibility
    seed = config.get("seed", 42)
    pl.seed_everything(seed=seed)
    log.info(f"Starting Experiment with seed: {seed}")

    # get dataloader
    log.info("Creating Datamodule")
    datamodule = instantiate(config.get("datamodule"))

    # load model
    log.info("Creating GAN")
    gan = instantiate(config.get("model"))

    # get trainer for gan training
    log.info("Creating Trainer")
    trainer = instantiate(config.get("trainer"), default_root_dir=config.get("output_dir"))

    # fit the gan
    log.info("Fitting the GAN")
    trainer.fit(gan, datamodule)


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # Perform training and testing
    train(cfg)


if __name__ == "__main__":
    main()
