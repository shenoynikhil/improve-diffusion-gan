# Iterative Noise Addition to Improve GAN Training
As part of Course Project in [CPSC533R: Computer Graphics and Computer Vision](https://www.cs.ubc.ca/~rhodin/2022_2023_CPSC_533R/).

### Setup
```bash
# Create a new conda environment
conda create -n gan-stable python=3.6
conda activate gan-stable

# install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch-nightly -c nvidia

# install other dependencies
pip install -r requirements.txt
```

### Running An Experiment
Select a experiment config from `experiments/` and run it with the following command:
```bash
python train.py --path experiments/<experiment_name>.yaml
```

### Pre-Commit Hooks
We use [pre-commit](https://pre-commit.com/) to run a set of checks on the code before committing. To install the hooks, run:
```bash
pip install pre-commit
pre-commit install
```
