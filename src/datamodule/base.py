"""Datamodule for all datasets
"""
import pytorch_lightning as pl
import torchvision.transforms as tv_transforms
from torch.utils.data import DataLoader
from torchvision import datasets


class BaseDataModule(pl.LightningDataModule):
    """BaseDataModule
    Intializes dataloaders, datasets
    """

    def __init__(
        self,
        data_type: str,
        root_data_dir: str = "data/",
        transforms: list = None,
        batch_size: int = 100,
        num_workers: int = 1,
    ):
        """Initializes

        Parameters
        ----------
        data_type: str
            Contains the type of dataset to be used
            One of ["mnist", "cifar10", "celeba"]
        transforms_list: list
            Contains a list of transforms to be applied to the dataset while calling
            `__getitem__()`
        batch_size: int
            Contains batch_size information for dataloaders
        num_worker: int
            Contains num_workers to be used in the dataloader
        """
        super().__init__()
        self.data_type = data_type
        self.root_data_dir = root_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # setup transforms
        img_size_default = 28 if data_type == "mnist" else 32
        self.transforms = (
            tv_transforms.Compose(
                [
                    tv_transforms.Resize((img_size_default, img_size_default)),
                    tv_transforms.ToTensor(),
                    tv_transforms.Normalize([0.5], [0.5]),
                ]
            )
            if transforms is None
            else transforms
        )

        # perform setup in init
        self.setup()

    def setup(self, stage: str = "fit"):
        """Sets up the main dataset"""
        if self.data_type == "mnist":
            if stage == "fit":
                self.dataset = datasets.MNIST(
                    root=f"{self.root_data_dir}/mnist/train",
                    train=True,
                    download=True,
                    transform=self.transforms,
                )
            elif stage == "test":
                self.test_dataset = datasets.MNIST(
                    root=f"{self.root_data_dir}/mnist/test",
                    train=False,
                    download=True,
                    transform=self.transforms,
                )
        elif self.data_type == "cifar10":
            if stage == "fit":
                self.dataset = datasets.CIFAR10(
                    root=f"{self.root_data_dir}/cifar10/train",
                    train=True,
                    download=True,
                    transform=self.transforms,
                )
            elif stage == "test":
                self.test_dataset = datasets.CIFAR10(
                    root=f"{self.root_data_dir}/cifar10/test",
                    train=False,
                    download=True,
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
