"""Datamodule for all datasets
"""
import logging
import os
from os.path import join

import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

from models import generate_and_save_images


class BaseDataModule(pl.LightningDataModule):
    """BaseDataModule
    Intializes dataloaders, datasets for all the experiments
    """

    def __init__(
        self,
        data_type: str,
        transforms: list,
        output_dir: str,
        batch_size: int = 100,
        num_workers: int = 1,
        channels: int = 1,
    ):
        """Initializes

        Parameters
        ----------
        data_type: str
            Contains the type of dataset to be used
            One of ["mnist", "cifar10", "celeba"]
        transforms: list
            Contains a list of transforms to be applied to the dataset while calling
            `__getitem__()`
        output_dir: str
            Contains a path where artifacts related to this datamodule gets dumped
        batch_size: int
            Contains batch_size information for dataloaders
        num_worker: int
            Contains num_workers to be used in the dataloader
        channels: int
            Contains channel information of the images present in the Dataset
        """
        super().__init__()
        self.data_type = data_type
        self.transforms = transforms
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.channels = channels

        # perform setup in init
        self.setup()

    def setup(self):
        """Sets up the main dataset"""
        if self.data_type == "mnist":
            self.dataset = datasets.MNIST(
                root="data/mnist/train",
                train=True,
                download=False,  # cannot download on sockeye jobs
                transform=self.transforms,
            )
            self.test_dataset = datasets.MNIST(
                root="data/mnist/test",
                train=False,
                download=False,  # cannot download on sockeye jobs
                transform=self.transforms,
            )
        elif self.data_type == "cifar10":
            self.dataset = datasets.CIFAR10(
                root="data/cifar10/train",
                train=True,
                download=False,  # cannot download on sockeye jobs
                transform=self.transforms,
            )
            self.test_dataset = datasets.CIFAR10(
                root="data/cifar10/test",
                train=False,
                download=False,  # cannot download on sockeye jobs
                transform=self.transforms,
            )
        else:
            raise NotImplementedError

    def train_dataloader(self):
        """Returns train dataloader"""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Returns test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def get_generated_datasets(
        self,
        gan=None,
        gen_images_dir: str = None,
        gen_test_images_dir: str = None,
    ):
        """Get generated datasets for MIA Experiment
        Two choices,
        1. If `gen_images_dir` and `gen_test_images_dir` is None, generate images with gan
        2. If `gen_images_dir` and `gen_test_images_dir` is not None, load them as dataset

        Parameters
        ----------
        gan: pl.LightningModule
            GAN that is compatible with the `models.gan_utils.generate_and_save_images()`
        gen_images_dir: str
            Path to a directory containing generated images to be used for training.
            Size of this is equal to `datamodule.indices_dict["mia_train"]`
        gen_test_images_dir: str
            Path to a directory containing generated images to be used for testing
            Size of this is equal to `datamodule.indices_dict["mia_test"]`
        """
        # Use the generator to create a synthetic dataset containing 1k points
        if gen_images_dir is None:
            logging.info("Creating a synthetic dataset of 10k points")
            gen_images_dir = join(self.output_dir, "gen_images/")
            os.makedirs(gen_images_dir, exist_ok=True)
            generate_and_save_images(
                gan=gan,
                number_of_images=len(self.indices_dict["mia_train"]),
                output_dir=join(gen_images_dir, "0"),
                batch_size=100,
            )

        # Use ImageFolder to create dataset(s)
        transforms_list = [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
        if self.channels == 1:
            transforms_list.insert(0, transforms.Grayscale(num_output_channels=1))
        gen_dataset = datasets.ImageFolder(
            root=gen_images_dir,
            transform=transforms.Compose(transforms_list),
        )
        gen_dataset_indices = list(np.arange(len(gen_dataset)))

        # set label
        gen_dataset.targets = [0.0 for _ in range(len(gen_dataset))]

        if gen_test_images_dir is None:
            gen_test_images_dir = join(self.output_dir, "gen_test_images/")
            os.makedirs(gen_test_images_dir, exist_ok=True)
            generate_and_save_images(
                gan=gan,
                number_of_images=len(self.indices_dict["mia_test"]),
                output_dir=join(gen_test_images_dir, "0"),
                batch_size=100,
            )

        # get test datasets
        test_gen_dataset = datasets.ImageFolder(
            root=gen_test_images_dir,  # target folder of images
            transform=transforms.Compose(transforms_list),
        )
        test_gen_dataset_indices = list(np.arange(len(test_gen_dataset)))
        test_gen_dataset.targets = [0.0 for _ in range(len(test_gen_dataset))]

        return (
            gen_dataset,
            gen_dataset_indices,
            test_gen_dataset,
            test_gen_dataset_indices,
        )
