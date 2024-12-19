######
# Citations: 
# The Local Trainer class is a reimplementation of:
#  https://github.com/myeongkyunkang/FedISCA/blob/main/train_classifier.py
# The class format and code is inspired by:
#   Pytorch documentation (https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)
# As well as suggestions made from OpenAI's gpt4o when given the prompt:
#   "Walk me through an implementation of a simple training loop in pytorch as a class"
# The examples from Pytorch and OpenAI were adapted for our use case, and the
# existing FedISCA code was transformed into a class and allows for general datasets 
######
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_sch
import torch.backends.cudnn as cudnn

from sklearn import metrics
from sklearn.utils import class_weight
#from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from datetime import datetime

from resnet import ResNet50


class LocalTrainer:
  def __init__(self,
               model,
               dataset,
               dir_save_model_weights,
               num_classes, 
               root=None,
               seed = None,
               epochs = 100, 
               loss_fn = "cross entropy",
               optimizer = "Adam",
               aug = [], 
               preprocess = [transforms.ToTensor()],
               batch_size = 128, 
               lr = 0.001,
               lr_adj = [50, 75],
               data_to_gpu = False,
               ):
    """
    Initializes a LocalTrainer object
    
    Parameters:
    -----------
    model (PyTorch module subclass):
        The model to train (ResNet, VGG, etc.)

    dataset (PyTorch Dataset subclass):
        The training data

    dir_save_model_weights (str):
        Path to the directory to save the best model weights
    
    num_classes (int): 
        Number of output classes
    
    seed (int, optional): 
        Random seed. If random seed is provided, then deterministic flags are
        set. Note that some operations may not have a deterministic implementation,
        so behavior is not guarenteed.

    epochs (int):
        The number of epochs to train for

    loss_fn (str, optional):
        The loss function to use in NN training. 
        Currently supported: ["cross entropy"]
        Default: "cross entropy"

    optimizer (str, optional):
        The optimizer to use in NN training. Currently supported: "Adam"
        Default: "Adam"
    
    aug (list, optional): 
        List containing data augmentations (resize crop, color jitter, etc.)
        Default is None, where no augmentation is applied

    preprocess (list, optional):
        List containing preprocessing steps
        Default is a transformation to tensor. Every preproces step
        MUST contain transforms.ToTensor() step
    
    batch_size (int, optional): 
        Number of samples per batch. Default is 128.
    
    lr (float, optional): 
        Initial learning rate. Default is 0.001.
    
    lr_adj (list, optional): 
        Epoch milestones for lr adjustment. Default is [50, 75].

    data_to_gpu (bool, optional):
        Whether or not to directly load the train/val data to the GPU
        Default is False
    """
    if seed is not None:
      torch.manual_seed(seed)
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      cudnn.deterministic = True
      cudnn.benchmark = False
    supported_optimizers = ["Adam"]
    supported_lossfn = ["cross entropy"]

    # self.train_data_path = train_data_path
    # self.val_data_path = val_data_path
    self.dir_save_model_weights = dir_save_model_weights
    os.makedirs(self.dir_save_model_weights, exist_ok=True)

    self.num_classes = num_classes
    self.aug = aug
    self.preprocess = preprocess
    self.seed = seed
    self.epochs = epochs
    self.batch_size = batch_size
    #self.lr_init = lr
    self.lr_curr = lr
    self.lr_adj = lr_adj
    self.aug = aug

    self.model = model
    # change last linear layer to match number of output classes
    features = self.model.fc.in_features
    self.model.fc = nn.Linear(features, num_classes)

    # data transforms + preprocess, val set should NOT be augmented
    train_transforms = transforms.Compose(aug + preprocess)
    val_transforms = transforms.Compose(preprocess)
    self.dataset = dataset
    # set up datasets
    if root is not None:
      self.train_data = self.dataset(split="train", root=root, transform=train_transforms, download=False)
      self.val_data = self.dataset(split="val", root=root, transform=val_transforms, download=False)
    else:
      self.train_data = self.dataset(split="train", transform=train_transforms, download=False)
      self.val_data = self.dataset(split="val", transform=val_transforms, download=False)
   
    # use GPU if available
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = self.model.to(self.device)

    if data_to_gpu:
      self.train_data.data.to(self.device)
      self.train_data.target.to(self.device)
    
    self.train_loader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=0)
    self.val_loader = DataLoader(dataset=self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=0)

    if loss_fn == "cross entropy":
      self.criterion = nn.CrossEntropyLoss()
    else:
      raise ValueError(f"Loss function '{loss_fn}' is not supported. Choose from {supported_lossfn}")
    if optimizer == "Adam":
      self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    else:
      raise ValueError(f"Optimizer '{optimizer}' is not supported. Choose from {supported_optimizers}")
    # later have this as a parameter
    self.scheduler = lr_sch.ReduceLROnPlateau(self.optimizer, 'min')

  def train_epoch(self):
    """
    Train a single epoch of self.model
    """
    self.model.train()
    tot_loss = 0.0
    for i, (inputs, labels) in enumerate(self.train_loader):
      labels = labels.flatten().long() if len(labels.shape) == 2 else labels.long()
      inputs, labels = inputs.to(self.device), labels.to(self.device)
      self.optimizer.zero_grad()
      out = self.model(inputs)
      loss = self.criterion(out, labels)
      # backprop
      loss.backward()
      # Adjust learning weights
      self.optimizer.step()
      # sum up batch loss
      tot_loss += loss.item()
      # for debugging can print batch loss
      #print("Batch %d loss: %f" % (i, loss.item()))

    epoch_loss = tot_loss / self.batch_size
    return epoch_loss
  
  def load_model(self, path):
    self.model.state_dict = torch.load(path)

  def validate(self, loader):
    """
    Validate a single epoch of training
    """
    self.model.eval()
    val_loss = 0.0
    corr, num_img = 0, 0
    truth_lbls = []
    pred_lbls = []
    with torch.no_grad():
      for i, (inputs, labels) in enumerate(loader):
        labels = labels.flatten().long() if len(labels.shape) == 2 else labels.long()
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        out = self.model(inputs)
        loss = self.criterion(out, labels).sum()
        val_loss += loss.item()
        
        pred = out.data.max(1)[1]
        corr += pred.eq(labels.data.view_as(pred)).sum()
        num_img += inputs.shape[0] # sum over all batches

        truth_lbls.extend(labels.cpu().tolist())
        pred_lbls.extend(pred.cpu().tolist())
        # for debugging can print batch loss
        #print("Val batch %d loss: %f" % (i, loss.item()))
    acc = corr / num_img
    b_acc = metrics.balanced_accuracy_score(truth_lbls, pred_lbls)
    val_epoch_loss = val_loss / num_img
    return val_epoch_loss, acc.item(), b_acc
  def train(self):
    """
    Train model for self.epochs epochs
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_vloss = 0
    for e in range(self.epochs):
      train_loss = self.train_epoch()
      val_loss, acc, b_acc = self.validate(self.val_loader)
      self.scheduler.step(val_loss)

      print('Epoch num: %d \nTrain loss: %f \nVal loss: %f \nVal acc: %f\nVal balanced acc: %f' % (e + 1, train_loss, val_loss, acc, b_acc))
      if val_loss < best_vloss or e == 0:
        best_vloss = val_loss
        best_state = self.model.state_dict()
        # save model state for epochs that improve val loss
        
    #model_path = os.path.join(self.dir_save_model_weights, '{}_model_date_{}_ep_{}'.format(self.dataset.flag, timestamp, e + 1))
    model_path = os.path.join(self.dir_save_model_weights, 'best.pth')
    torch.save(best_state, model_path)
    print("All done!")