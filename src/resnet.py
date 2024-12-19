# implementation inspired from https://github.com/myeongkyunkang/FedISCA/blob/main/models/resnet_cifar.py
# and pytorch https://pytorch.org/vision/0.9/_modules/torchvision/models/resnet.html
import torch
import torch.nn as nn
from torch import Tensor


class BaseBlock(nn.Module):
    """
    Base block of Resnet. 
    
    Consists of 2 convolutional layers with downsampling
    in the first layer
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(BaseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                      out_channels, 
                      stride = stride, 
                      kernel_size = 3, 
                      padding = 1, 
                      bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, 
                      out_channels,
                      kernel_size = 3,
                      padding = 1,
                      bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * out_channels))
        self.relu = nn.ReLU()

    def forward(self, x) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        x = self.downsample(x)
        out += x
        out = self.relu(out)
        return out
        
class BottleNeck(nn.Module):
    """
    Alternate base block for ResNet

    Three convolution layers with downsampling in the second layer.
    """
    # default expansion
    expansion: int = 4
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 stride = 1, 
                 groups = 1, 
                 base_width = 64, 
                 dilation = 1):
        super(BottleNeck, self).__init__()
        width = int(out_channels * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(in_channels,
                      width,
                      kernel_size = 1,
                      bias = False)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(width,
                      width,
                      stride = stride,
                      kernel_size = 3,
                      padding = 1,
                      groups = groups,  
                      bias = False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width,
                      out_channels * self.expansion,
                      kernel_size = 1,
                      bias = False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * out_channels))
        self.stride = stride

    def forward(self, x) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #print(out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        #print(out.shape)
        out = self.conv3(out)
        out = self.bn3(out)
        #print(out.shape)
        x = self.downsample(x)
        #print(x.shape)
        out += x
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    """
    ResNet arch using any block (Basic or Bottleneck) 
    """
    def __init__(
            self,
            block,
            layers,
            num_classes = 10,
            in_channels = 3 
        ):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(in_channels, 
                               out_channels = 64, 
                               kernel_size = 3, 
                               stride = 1,
                               padding = 1,
                               bias = False)
         
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, planes = 64, num_blocks = layers[0], stride = 1)
        self.layer2 = self._make_layer(block, planes = 128, num_blocks = layers[1], stride = 2)
        self.layer3 = self._make_layer(block, planes = 256, num_blocks = layers[2], stride = 2)
        self.layer4 = self._make_layer(block, planes = 512, num_blocks = layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, ret_feature = False):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        feature = out.view(out.size(0), -1)
       
        out = self.fc(feature)
        if ret_feature == False:
            return out
        else:
            return out, feature

def ResNet18(in_channels=3, num_classes=10):
    return ResNet(BaseBlock, [2, 2, 2, 2], num_classes, in_channels=in_channels)

def ResNet50(in_channels=3, num_classes=10):
  return ResNet(BottleNeck, [3, 4, 6, 3], num_classes, in_channels=in_channels)

