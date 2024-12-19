from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

class LocalDataset(Dataset):
    """
    General local dataset representation for federation 
    """
    def __init__(
            self, 
            split,
            file_name,
            root =os.getcwd(),
            size = None,
            transform=None,
            target_transform=None,
            mmap_mode=None,
            download=False):
        """
        Init local dataset

        Parameters:
        -----------
        split : "train", "val", or "test"
        file_name : Name of the dataset (str). Does NOT include file extension
        root : root directory of the dataset (str)
        size : size of the images (e.g. 28x28, 224x224, etc)
        transform : transform to apply to data
        target_transform : transform to apply to targets
        mmap_mode : memory map mode for loading npz
        download : WORK IN PROGRESS - currently only supports on-disk datasets
        """

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.size = size
        self.data_path = os.join(root, f"{file_name}.npz")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"File {self.data_path} does not exist")

        if self.split in ["train", "val", "test"]:
            npz_data = np.load(self.data_path, mmap_mode=mmap_mode)
            self.imgs = npz_data[f"{self.split}_images"]
            self.labels = npz_data[f"{self.split}_labels"]
        else:
            raise ValueError()

    def __len__(self):
        return self.imgs.shape[0]

    def __repr__(self):
        return f"LocalDataset: {self.data_path}"
    
    def __get_item__(self, idx):
        img, target = self.imgs[idx], self.labels[idx].astype(int)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def download(self):
        raise NotImplementedError("Download functionality not yet implemented")
        



