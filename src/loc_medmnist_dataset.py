import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader
# from pathlib import Path
# import pandas as pd
# from skimage import io, transform
# import numpy as np
# from torchvision import transforms, utils

class LocalMedMNISTDataset():
  """
  Custom dataset representation that houses a local (external) partition 
  of MedMNIST data.
  """
  def __init__(self, data_path, medmnist_dataset):
    """
    Init medmnist data

    Parameters:
    -----------
    data_path : Path object
    medmnist_dataset : medmnist dataset used (str)
    """
    self.data_path = data_path
    self.mmn_name = medmnist_dataset
    info = INFO[self.mmn_name]
    self.task = info['task']
    self.dc = getattr(medmnist, info['python_class'])
  
  def get_data_loader(self, transforms, bs, split, download=False, shuffle=True):
    """
    Returns a torch DataLoader of the external dataset

    Parameters:
    -----------
    transforms : data transform to apply to data
    bs : batch size (int)
    split : get data loader for "test" or "train" split
    download : whether or not to download the data to disk.
               Default is False, in most use cases data is already on disk
    shuffle : whether or not to shuffle data, default is True. 
              Note: shuffle is not recommended for test or validation splits
    """
    dataset = self.dc(root=str(self.data_path), split=split, transform=transforms, download=download)
    # TODO: randomness introduced here -> do we need to ensure deterministic for experiments?
    #print(dataset)
    loader = DataLoader(dataset=dataset, batch_size=bs, shuffle=shuffle)
    return loader
    



