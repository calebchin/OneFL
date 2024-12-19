import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# implementations from https://github.com/myeongkyunkang/FedISCA/blob/main/utils.py
class DeepInversionHook():
  """
  Forward hook to track BN statistics (mean and loss), computes L2 loss
  """
  def __init__(self, module):
    self.hook = module.register_forward_hook(self.hook_fn)

  def hook_fn(self, module, input, output):
    num_channels = input[0].shape[1]
    mean = input[0].mean([0, 2, 3])
    var = input[0].permute(1, 0, 2, 3).contiguous().view([num_channels, -1]).var(1, unbiased=False)
    reg_loss = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(module.running_mean.data.type(var.type()) - mean, 2)
    self.r_feature = reg_loss
  
  def close(self):
    self.hook.remove()

class Ensemble(torch.nn.Module):
  """
  Create an ensemble of models
  """
  def __init__(self, models):
    """
    Parameters:
    -----------
    models (list)
      List of models to ensemble
    """
    super(Ensemble, self).__init__()
    self.models = nn.ModuleList(models)

  def forward(self, x):
    logits_total = 0
    for i in range(len(self.models)):
      logit = self.models[i](x)
      logits_total += logit
    avg_logit = logits_total / len(self.models)
    return avg_logit

class KLDiv(nn.Module):
  """
  Module that computes the Kullback-Leibler divergence between two logits
  """
  def __init__(self, T=1.0, reduction='batchmean'):
    """
    Parameters:
    -----------
    T : temperature (smoothness) of the softmax distribution
        Default is 1.0
    reduction : Reduction method for the KL div
    """
    super().__init__()
    self.T = T
    self.reduction = reduction
  
  def forward(self, logits, targets):
    q = F.log_softmax(logits / self.T, dim=1)
    p = F.softmax(targets / self.T, dim=1)
    return F.kl_div(q, p, reduction=self.reduction) * (self.T * self.T)


  def kldiv(self, logits, targets, ):
    q = F.log_softmax(logits / self.T, dim=1)
    p = F.softmax(targets / self.T, dim=1)
    return F.kl_div(q, p, reduction=self.reduction) * (self.T * self.T)

# implementation inspired by https://github.com/myeongkyunkang/FedISCA/blob/main/utils.py
# and do_fl_partitioning in 
# https://github.com/jopasserat/federated-learning-tutorial/blob/master/fl_demo/fl_utils.py
def fl_partition(dataset, d_name, num_clients, num_labels, iid=True, beta=0.4, min_size=100, val_split=0.2, seed=None):
  """
  Simulates a federated learning scenario by partitioning a dataset
  on disk. Requires that dataset is already downloaded.

  Parameters:
  -----------
  dataset : a DataClass instance of medmnist or custom data class object
  num_clients : the number of partitions to create
  iid : whether or not to use iid partitioning or non-iid (dirichlet) partitions
  min_size : minimum samples for one client using Dirichlet non-iid partition

  """
  rng = np.random.default_rng(seed=seed)
  all_images = dataset.imgs
  data_size = all_images.shape[0]
  all_labels = np.array(dataset.labels)

  if iid:
    permute_ids = rng.permutation(len(all_images))
    split_ids = np.array_split(permute_ids, num_clients)
    part_type = "iid"
  else:
    # use dirichlet non-iid partition
    curr_min_size = 0
    while curr_min_size < min_size:
      split_ids = [[]] * num_clients
      for k in range(num_labels):
        id_k = np.where(all_labels == k)[0]
        rng.shuffle(id_k)
        props = rng.dirichlet(np.repeat(beta, num_clients))
        #print(props)
        # assign 0 probability to the clients who have more than average number of samples
        props = np.array([p * (len(idx_j) < data_size / num_clients) for p, idx_j in zip(props, split_ids)])
        # normalize
        props = props / np.sum(props)
        # scale to indice partition, remove last element
        props = (np.cumsum(props) * len(id_k)).astype(int)[:-1]
        #print(props)
        split_ids = [curr_ids + list(new_ids) for curr_ids, new_ids in zip(split_ids, np.split(id_k, props))]
        curr_min_size = min([len(ids) for ids in split_ids])
    # shuffle ids for each client
    for c in range(num_clients):
      rng.shuffle(split_ids[c])
    part_type = "dirichlet"
    
  # now partition data on disk
  partition_dir = Path(dataset.root) / Path((d_name) + "_expr")  / ("sim_partitions_" + part_type)
  if partition_dir.exists():
    shutil.rmtree(partition_dir)
  Path.mkdir(partition_dir, parents=True, exist_ok=True)

  list_name = []
  # iterate through the clients
  for cli in range(num_clients):
    # make a directory for client
    Path.mkdir(partition_dir / f"client_{cli}", exist_ok=True)
    # create train / val splits
    # inds are alreafy shuffled, just take val_percentage of the inds
    if val_split > 0.0:
      num_val = int(val_split * len(split_ids[cli])) 
      train_ids, val_ids = split_ids[cli][num_val:], split_ids[cli][:num_val]
      train_imgs = all_images[train_ids]
      train_labels = all_labels[train_ids]

      val_imgs = all_images[val_ids]
      val_labels = all_labels[val_ids]

      # open new file and save the train val split
      with open(partition_dir / f"client_{cli}" / f"{dataset.flag}{dataset.size_flag}.npz", 'wb') as f:
        d = {
          "train_images" : train_imgs,
          "train_labels" : train_labels,
          "val_images" : val_imgs,
          "val_labels" : val_labels
        }
        list_name.append(f.name)
        np.savez(f, **d)

  return partition_dir, list_name, split_ids

def get_data_loader(data_class, data_path, filename, bs, split, transforms=None, target_transforms=None, download=False, shuffle=True):
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
    dataset = data_class(root=str(data_path), filename=filename, split=split, transform=transforms, target_transforms=target_transforms, download=download)
    # TODO: randomness introduced here -> do we need to ensure deterministic for experiments?
    #print(dataset)
    loader = DataLoader(dataset=dataset, batch_size=bs, shuffle=shuffle)
    return loader
