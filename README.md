# Improved Training Techniques for Diffusion-GANs

[![python](https://img.shields.io/badge/-Python_3.9-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_1.9.4-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3.1-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) <br>

As part of Course Project in [CPSC533R: Computer Graphics and Computer Vision](https://www.cs.ubc.ca/~rhodin/2022_2023_CPSC_533R/).

Our techniques build upon the original [Diffusion-GAN](https://openreview.net/forum?id=HZf7UbpWHuA) paper.

The following GANs are supported,
* VanillaGAN (DCGAN)
* Wasserstein GAN with Gradient Penalty (WGAN_GP)
* Spectral Normalization GAN (SN_GAN)

Our modifications are,
* Multiple Time Step Noise (mts)
* Reverse Strategy (rs): More noise at the beginning of the diffusion process and gradually decrease it.
* Top-K Training (tk): Only use the top-k samples to train the discriminator. [Paper](https://arxiv.org/abs/2002.06224)

### Setup
```bash
# Create a new conda environment
conda create -n gan-stable python=3.9
conda activate gan-stable

# install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# install other dependencies
pip install -r requirements.txt
```

### Running An Experiment
Select an experiment config from `configs/experiments/` and run it with the following command:
```bash
python train.py experiment=<path to experiment config without the configs/experiment prefix>
```

### Pre-Commit Hooks
We use [pre-commit](https://pre-commit.com/) to run a set of checks on the code before committing. To install the hooks, run:
```bash
pip install pre-commit
pre-commit install
```

#### Contributors
- [Nikhil Shenoy](https://shenoynikhil.com/about)

