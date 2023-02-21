"""
TODO: Setup for this project
Dataset class for CelebA"""
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CelebADataset(Dataset):
    """Dataset class for CelebA Dataset
    - Source: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
    - If not multi_label: Gender (Male/Female) is the Target Variable
    - else: MultiLabel (all 40 binary attributes)

    Each data point has 40 binary attributes in the attributes csv file.
    We use the `male` column to get the binary target variable.
    """

    def __init__(
        self,
        folder_dir: str = "data/img_align_celeba",
        transforms=None,
        mia_labels: bool = False,
        multi_label: bool = False,
    ):
        """Initialize CelebA dataset

        Parameters
        ----------
        folder_dir: str
            Path to where the CelebA dataset is stored
        transforms: list
            List of transforms applied with get item
        mia_labels: bool
            If turned on, labels are `torch.tensor(1.)` for all `__getitem__()` calls
        multi_label: bool
            If turned on, labels are all 40 attributes
        """
        super().__init__()
        attr_file_path = os.path.join(folder_dir, "list_attr_celeba.csv")
        assert os.path.exists(attr_file_path), "Attribute file does not exist"
        self.files = pd.read_csv(attr_file_path)

        self.img_dir = os.path.join(folder_dir, "img_align_celeba")
        assert os.path.exists(self.img_dir)

        # setup transforms
        assert transforms is not None, "Pass transforms while setting up datamodule"
        self.transforms = transforms

        # will be used during the `__getitem__()` call
        self.mia_labels = mia_labels

        # used during `__getitem__` call
        self.multi_label = multi_label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = self.transforms(Image.open(os.path.join(self.img_dir, self.files.iloc[index, 0])))
        # if we want mia_labels (all torch.Tensor([1.0]))
        if self.mia_labels:
            return (img, 1.0)

        # in the multi label case return all 40 binary attributes
        if self.multi_label:
            label = torch.Tensor(
                [1 if x == 1.0 else 0 for x in self.files.iloc[index, 1:].to_list()]
            )
            return img, label

        # get gender label from "Male" column
        label = self.files.loc[index, "Male"]
        # since labels are either 1. or -1.
        label = 1.0 if label == 1.0 else 0.0

        return (img, label)
