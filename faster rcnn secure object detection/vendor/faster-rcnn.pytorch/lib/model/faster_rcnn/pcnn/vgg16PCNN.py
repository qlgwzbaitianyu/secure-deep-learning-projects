# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb
from smc_function_number import smc_functions

class vgg16PCNN(_fasterRCNN):
  def __init__(self, kernel):
    super(MyModel, self).__init__()
    self.conv1 = nn.Conv2d(1, 1, 3, 1, 1, bias=False)

    # self.relu1 = nn.ReLU()
    self.relu1 = pcnnReluA()
    self.conv2 = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
    self.maxpool1 = nn.MaxPool2d();
    # Add other layers here

    # Initialize conv1 with custom kernel
    self.conv1.weight = nn.Parameter(kernel)

  def forward(self, x):
    x = self.conv1(x)
    x =self.relu1(out)
    x = self.conv2(x)
    x = self.maxpool1(x)
    # pass x to other modules
    return x

  def pcnnReluA(self, bottom, top):
    smc = smc_functions()
    bn = bottom[0].num
    bc = bottom[0].channels
    bh = bottom[0].height
    bw = bottom[0].width
    for n in range(bn):
      for c in range(bc):
        for h in range(bh):
          for w in range(bw):
            bit_1, bit_2 = smc.SecBitExtra(bottom[0].data[n, c, h, w], bottom[1].data[n, c, h, w])
            bit = bit_1 ^ bit_2
            if bit == 1:
              bottom[0].data[n, c, h, w] = 0
    top[0].data[...] = bottom[0].data

    def pcnnReluB(self, bottom, top):
      smc = smc_functions()
      bn = bottom[0].num
      bc = bottom[0].channels
      bh = bottom[0].height
      bw = bottom[0].width
      for n in range(bn):
        for c in range(bc):
          for h in range(bh):
            for w in range(bw):
              bit_1, bit_2 = smc.SecBitExtra(bottom[0].data[n, c, h, w], bottom[1].data[n, c, h, w])
              bit = bit_1 ^ bit_2
              if bit == 1:
                bottom[0].data[n, c, h, w] = 0
      top[0].data[...] = bottom[0].data


