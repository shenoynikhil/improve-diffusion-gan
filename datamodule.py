"""
Class to get dataloaders for MIA experiment
-------------------------------------------
We need to use the datapoints carefully, so we curate the datasets that would be used for training
specific classifiers. We create the following dataloaders,
- GAN Training: use `dataloader = get_dataloader(type='gan')`
- net_f Training
- net_g Training
- mia training
- mia testing
"""
import copy
import logging
import os
import random
from os.path import join

import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from celeba_dataset import CelebADataset
from models import generate_and_save_images


class MIAExperimentDataModule(pl.LightningDataModule):
    """MIAExperimentDataModule
    Intializes dataloaders, datasets for all the experiments
    """

    def __init__(
        self,
        data_path: str,
        transforms: list,
        output_dir: str,
        batch_size: int = 100,
        num_workers: int = 1,
        channels: int = 1,
        dataset_frac: float = 1.0,
        multi_label: bool = False,
        seed: int = 42,
    ):
        """Initializes

        Parameters
        ----------
        data_path: str
            Contains path information of where the dataset is stored
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
        dataset_frac: float
            Fraction of dataset will be used, to be used for debugging
        multi_label: bool
            If turned on, multi-label targets will be returned
        seed: int
            Random seed to be used, wherever `random` calls
        """
        super().__init__()
        self.data_path = data_path
        self.transforms = transforms
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.channels = channels
        self.dataset_frac = dataset_frac
        self.multi_label = multi_label
        self.seed = seed

        # perform setup in init
        self.setup()

    def setup(self):
        """Sets up the main dataset"""
        if "mnist" in self.data_path:
            self.dataset = datasets.MNIST(
                self.data_path,
                train=True,
                download=False,  # cannot download on sockeye jobs
                transform=self.transforms,
            )
        elif "cifar10" in self.data_path:
            self.dataset = datasets.CIFAR10(
                self.data_path,
                train=True,
                download=False,  # cannot download on sockeye jobs
                transform=self.transforms,
            )
        elif "celeba" in self.data_path:
            self.dataset = CelebADataset(
                folder_dir=self.data_path,
                transforms=self.transforms,
                multi_label=self.multi_label,
            )
        else:
            raise NotImplementedError

        # TODO: Remove later, if self.dataset_frac < 1.0 then subset the dataset
        if self.dataset_frac < 1.0:
            self.dataset = self._get_subset(self.dataset, self.dataset_frac)

        # all indices
        self.all_indices = list(range(len(self.dataset)))
        random.seed
        random.shuffle(self.all_indices)

        # for getting dataset subsets
        n = len(self.all_indices)
        self.indices_dict = {
            "net_f": self.all_indices[: int(n / 2)],
            "net_g": self.all_indices[int(n / 2) :],
            "gan": self.all_indices[int(n / 3) : int(n / 2)],
            "mia_train": self.all_indices[int(n / 6) : int(n / 3)],
            "mia_test": self.all_indices[: int(n / 6)],
        }

    def _get_subset(self, dataset, dataset_frac):
        """Constructs a subset based on a dataset_frac"""
        # compute number of points to consider
        n = int(len(dataset) * dataset_frac)
        indices = list(range(len(dataset)))
        return Subset(dataset=dataset, indices=indices[:n])

    def get_mia_real_dataset(self):
        """Returns a copy of the dataset initialized for MIA task.
        Changes the targets to all `1.0` to indicate these are real images
        """
        # when celeba, things are slightly different since we have written
        # the dataset class
        if "celeba" in self.data_path:
            return CelebADataset(
                folder_dir=self.data_path,
                transforms=self.transforms,
                mia_labels=True,
            )

        # create gan dataset copy as well set targets to 1 (as these are real images)
        mia_real_dataset = copy.deepcopy(self.dataset)
        # set all targets to 1 (representing real images)
        mia_real_dataset.targets = [1.0 for _ in mia_real_dataset.targets]
        return mia_real_dataset

    def get_dataloader_from_indices(
        self, net_type: str, batch_size: int = None
    ) -> DataLoader:
        """Returns DataLoader based on net_type specified.
        Basically this retrieves indices from `net_type` using `datamodule.indices_dict: dict`.
        Make sure this is one of,

        Parameters
        ----------
        net_type: str
            One of ["net_f", "net_g", "gan", "mia_train", "mia_test"]
        """
        # get indices based on net_type
        indices = self.indices_dict[net_type]
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(
            dataset=Subset(dataset=self.dataset, indices=indices),
            batch_size=batch_size,
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
