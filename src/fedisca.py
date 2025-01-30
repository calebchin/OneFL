# A reimplementation of the FedISCA (https://github.com/myeongkyunkang/FedISCA) framework
import collections
import copy
import gc
import os
import random

from loc_medmnist_dataset import LocalMedMNISTDataset

from pathlib import Path

import medmnist
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import resnet18
from sklearn import metrics
from resnet import ResNet50, ResNet18
from utils import KLDiv, DeepInversionHook, Ensemble

  
class FedISCA:
  """
  Class to perform federated learning using FedISCA framework
  """
  def __init__(self, 
               test_data_dir,
               model_weights_dir, # directory containing models to federate
               exper_dir,
               input_size=24,
               in_channels=3, # might be a list in the future
               num_classes=10,
               batch_size=256, # maybe make a list for diff batch size per loc
               epochs=50,
               iter_mi=500, 
               di_lr=0.05,
               glb_lr=0.001, 
               lr_steps=[80, 120],
               di_var_scale=2.5e-5, 
               di_l2_scale=0.0,
               r_feature_weight=10,
               log_freq=200,
               optimizer="Adam",
               kldiv_T=20,
               is_medmnist=True,
               medmnist="pathmnist",
               seed =None):
    
    supported_optimizers = ["Adam"]
    self.models_dir = model_weights_dir
    self.exper_dir = exper_dir
    self.exper_img_dir = os.path.join(exper_dir, 'images')
    os.makedirs(self.exper_img_dir, exist_ok=True)
    self.in_channels = in_channels
    self.num_classes = num_classes
    self.test_data_dir = test_data_dir
    self.batch_size = batch_size
    self.epochs = epochs
    self.iter_mi = iter_mi
    self.glb_lr = glb_lr
    self.di_lr = di_lr
    self.lr_steps = lr_steps
    self.di_var_scale = di_var_scale
    self.di_l2_scale = di_l2_scale
    self.r_feature_weight = r_feature_weight
    self.seed = seed
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.input_size = input_size
    self.log_freq = log_freq

    # global test set: in practice this does not exist, but useful for experiments 
    if is_medmnist:
      self.test_data = LocalMedMNISTDataset(Path(self.test_data_dir), medmnist)


    if seed is not None:
      torch.manual_seed(seed)
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      cudnn.deterministic = True
      cudnn.benchmark = False

    if optimizer not in supported_optimizers:
      raise ValueError(f"Optimizer '{optimizer}' is not supported. Choose from {supported_optimizers}")
    
    self.optimizer_name = optimizer
    self.optimizer_img = None
    self.optimizer_glb = None
    #self.scheduler = lr_sch.ReduceLROnPlateau(self.optimizer_glb, 'min')
    self.kldiv_T = kldiv_T

  def _adj_lr(self, optimizer, epoch):
    if epoch not in self.lr_steps:
      return
    lr_dec = 0.1**(self.lr_steps.index(epoch) + 1)
    for param_group in optimizer.param_groups:
      param_group['lr'] = self.glb_lr * lr_dec
    
    
  def load_models(self, base_models=[]):
    """
    Returns an ensemble of the stored models for federation

    Parameters:
    base_model (object):
      Base model load weights into, initialized with num_classes and in_channels
    """
    models = []
    for i, loc_dir in enumerate(sorted(os.listdir(self.models_dir))):
      weight_path = os.path.join(self.models_dir, loc_dir, 'best.pth')
      model_weights = torch.load(weight_path, weights_only=True)
      # in the future, there will be different models, but for now, we retrieve a single model type
      # base_mod = ResNet18(in_channels=self.in_channels, num_classes=self.num_classes)
      if len(base_models) == []:
        base_mod = ResNet18(in_channels=self.in_channels, num_classes=self.num_classes)
        base_mod.load_state_dict(model_weights)
      else:
        base_mod = base_models[i]
        base_mod.load_state_dict(model_weights)
      base_mod.to(self.device)

      base_mod.eval() # set to evaluation
      models.append(base_mod)

    return Ensemble(models)

  def test_model(self, net, data_loader, criterion):
    """
    Test nn `net` on data from `data_loader` with criterion `criterion` 
    """
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    gt_list, pred_list = [], []
    with torch.no_grad():
      for batch_id, (inputs, targets) in enumerate(data_loader):
        # why do we need to flatten here?
        inputs, targets = inputs.to(self.device), targets.flatten().long().to(self.device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        gt_list.extend(targets.tolist())
        pred_list.extend(predicted.tolist())
    acc = correct / total
    b_acc = metrics.balanced_accuracy_score(gt_list, pred_list)
    # TODO: add more metrics here, fairness related?
    print('Loss: %.3f | Acc: %.3f%% (%d/%d), B. Acc: %.3f%%' % (test_loss / (batch_id + 1), 100. * acc, correct, total, 100. * b_acc))
    return test_loss, acc, b_acc

  def model_inversion(self, models, base_model=None, large_jtr = 30, small_jtr = 2):
    """
    Perform model inversion on ensemble models

    Parameters:
    -----------
    models (Ensemble)
      Models to federate
    base_model (object)
      Base model to use for classifier, initialized with num_classes and in_channels
    large_jtr (int) 
      Large jitter for model inversion, default is 30
    small_jtr (int)
      Small jitter for model inversion, default is 2
    """
    # random noise to generate images from
    noise_input = torch.randn((self.batch_size, self.in_channels, self.input_size, self.input_size),
                              requires_grad=True,
                              device=self.device,
                              dtype=torch.float)
    # target labels
    targets = torch.LongTensor(list(range(0, self.num_classes)) * 
                              (self.batch_size // self.num_classes) + 
                              list(range(0, self.batch_size % self.num_classes))).to(self.device)
    # later should be able to load in transfer learning model
    # TODO: generalize main model (maybe load in existing weights)
    # if base_model == "rn18":
    #   main_model = ResNet18(in_channels=self.in_channels, num_classes=self.num_classes)
    # elif base_model == "rn50":
    if base_model is None:
      base_model = ResNet18(in_channels=self.in_channels, num_classes=self.num_classes)
      main_model = base_model
    else:
      main_model = base_model
    main_model = main_model.to(self.device)
    
    self.optimizer_glb = optim.SGD(main_model.parameters(), lr = self.glb_lr, momentum=0.9, weight_decay=5e-4)
    self.optimizer_img = optim.Adam([noise_input], lr = self.di_lr)
    
    # hooks for feature stats
    hooks = []
    for module in models.modules():
      if isinstance(module, nn.BatchNorm2d):
        hooks.append(DeepInversionHook(module))
    
    criterion_cls = KLDiv(T=self.kldiv_T)
    criterion = nn.CrossEntropyLoss()

    preprocess_list = [transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])]
    prp = transforms.Compose(preprocess_list)
    test_data_loader = self.test_data.get_data_loader(transforms=prp, bs=self.batch_size, split="test")

    #models_noiseadapt = self.load_models()
    #models_noiseadapt.state_dict = models.state_dict
    models_noiseadapt = copy.deepcopy(models)
    models_noiseadapt = models_noiseadapt.to(self.device)
    models_noiseadapt.train()
    best_acc = 0
    bst_model = None
    bst_noiseadapt = None
    for epoch in range(self.epochs):
      # new input image (gaussian noise) for each input
      noise_input.data = torch.randn((self.batch_size,
                                      self.in_channels, 
                                      self.input_size, 
                                      self.input_size),
                                      requires_grad=True,
                                      device=self.device,
                                      dtype=torch.float
                                      )
      # collect inverted images
      best_cost = np.inf
      n_classes = targets.max().item() + 1
      # restart optimizer
      self.optimizer_img.state = collections.defaultdict(dict)
      images = []

      # empty cache
      torch.cuda.empty_cache()
      gc.collect()

      if noise_input.shape[-1] > 128:
          lim_0, lim_1 = large_jtr, large_jtr
      else:
          lim_0, lim_1 = small_jtr, small_jtr 
      print("Generating images...")
      # iters_mi images to generate
      for i in range(self.iter_mi):
        # jitter gaussian noise image
        jtr_1 = random.randint(-lim_0, lim_0)
        jtr_2 = random.randint(-lim_1, lim_1)

        aug_input = torch.roll(noise_input, shifts=(jtr_1, jtr_2), dims=(2, 3))

        self.optimizer_img.zero_grad()
        models.zero_grad()
        outputs = models(aug_input)
        loss_ce = criterion(outputs, targets)
        loss_target = loss_ce.item()

        diff1 = aug_input[:, :, :, :-1] - aug_input[:, :, :, 1:]
        diff2 = aug_input[:, :, :-1, :] - aug_input[:, :, 1:, :]
        diff3 = aug_input[:, :, 1:, :-1] - aug_input[:, :, :-1, 1:]
        diff4 = aug_input[:, :, :-1, :-1] - aug_input[:, :, 1:, 1:]
        loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
        loss_tv = self.di_var_scale * loss_var 
        # loss_var is L_TV

        # R_feature -> this is BN loss
        loss_bn = self.r_feature_weight * sum([m.r_feature for m in hooks])
        #torch.sum(torch.Tensor([m.r_feature for m in hooks]))

        loss_l2 = self.di_l2_scale * torch.norm(aug_input, 2)
        
        # total loss:
        loss = loss_ce + loss_tv + loss_bn + loss_l2
        if i % self.log_freq == 0:
            print(f"It {i}\t Losses: total: {loss.item():3.3f},\ttarget: {loss_target:3.3f} \tR_feature_loss scaled:\t {loss_bn.item():3.3f}")
            vutils.save_image(noise_input.data.clone(), '{}/output_{}_{}.png'.format(self.exper_img_dir, epoch, i), normalize=True, scale_each=True, nrow=n_classes)

        # check which aug_input has best loss
        if best_cost > loss.item():
          best_cost = loss.item()
          best_inputs = noise_input.data
        loss.backward()
        self.optimizer_img.step()

        images.append(noise_input.detach().cpu().data)
        
      # log performance on last image
      print(f"It {self.iter_mi}\t Losses: total: {loss.item():3.3f},\ttarget: {loss_target:3.3f} \tR_feature_loss scaled:\t {loss_bn.item():3.3f}")
      vutils.save_image(noise_input.data.clone(), '{}/output_{}_{}.png'.format(self.exper_img_dir, epoch, self.iter_mi), normalize=True, scale_each=True, nrow=n_classes)
      #-------------------------------
      # eval teacher on best images
      os.makedirs('{}/best_imgs'.format(self.exper_img_dir), exist_ok=True)
      vutils.save_image(best_inputs.clone(), '{}/best_imgs/output_{}.png'.format(self.exper_img_dir, epoch), normalize=True, scale_each=True, nrow=n_classes)

      # outputs = models(best_inputs)
      # _, preds = outputs.max(1)
      # print('Teacher correct out of {}: {}, loss at {}'.format(self.batch_size, preds.eq(targets).sum().item(), criterion(outputs, targets).item()))

      # Train central model on generated images
      print("Training classifier...")
      # models_noiseadapt = self.load_models()
      # models_noiseadapt.state_dict = models.state_dict
      # #models_noiseadapt = copy.deepcopy(models)
      # models_noiseadapt = models_noiseadapt.to(self.device)
      main_model.train()
      models_noiseadapt.train()
      self._adj_lr(self.optimizer_glb, epoch)

      for img_ind in range(len(images) - 1, -1, -1):
        with torch.no_grad():
          models_noiseadapt(images[img_ind].to(self.device))
      for img_ind in range(len(images)):
        input_img = images[img_ind].to(self.device)
        self.optimizer_glb.zero_grad()
        alpha = img_ind / len(images)
        with torch.no_grad():
          outputs_noise = models_noiseadapt(input_img)
          outputs = models(input_img)

        outputs_glb = main_model(input_img)
        real_loss = criterion_cls(outputs_glb, outputs.detach())
        noise_loss = criterion_cls(outputs_glb, outputs_noise.detach())
        # merge loss with alpha weights
        all_loss = alpha * real_loss + (1.0 - alpha) * noise_loss
        all_loss.backward()
        self.optimizer_glb.step()
      # validation
      print('Test classifier')
      val_loss, acc, b_acc = self.test_model(main_model, test_data_loader, criterion)
      with open(os.path.join(self.exper_dir, 'test_metrics.csv'), 'at') as wf:
        wf.write('{},{:.4f},{:.4f}\n'.format(epoch, acc, b_acc))

      if acc > best_acc:
        best_acc = acc
        bst_model = main_model
        bst_noiseadapt = models_noiseadapt
        torch.save(main_model, os.path.join(self.exper_dir, 'best.pth'))
        torch.save(models_noiseadapt, os.path.join(self.exper_dir, 'best_noise.pth'))  
    
    print('Test ensemble...')
    _, acc_ens, b_acc_ens = self.test_model(models, test_data_loader, criterion)
    with open(os.path.join(self.exper_dir, 'test_teacher.csv'), 'wt') as wf:
        wf.write('{:.4f}\n'.format(acc_ens))
    print('Test classifier')
    _, acc_ens, b_acc_ens = self.test_model(main_model, test_data_loader, criterion)
    with open(os.path.join(self.exper_dir, 'test_teacher.csv'), 'wt') as wf:
        wf.write('{:.4f}\n'.format(acc_ens))
    
    torch.save(main_model, os.path.join(self.exper_dir, 'glb_model_last.pth'))
    torch.save(models_noiseadapt, os.path.join(self.exper_dir, 'noise_adapt_ensemble_last.pth'))
    return bst_model, bst_noiseadapt