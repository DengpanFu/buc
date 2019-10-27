#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Dengpan Fu (v-defu@microsoft.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys; sys.path.insert(0, '../..')
import numpy as np
import collections, itertools

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from .base_model import BaseModel

class BackboneModel(BaseModel):
    def __init__(self, name='resnet50', model_zoo_init=True, 
        model_zoo_path=None, pre_trained=False, pre_model=None, 
        cut_at_pooling=True, last_conv_stride1=True):
        super(BackboneModel, self).__init__(name=name)
        if not 'resnet' in self.name:
            raise ValueError("Current only exp with resnet")
        self.model_zoo_init = model_zoo_init
        self.model_zoo_path = model_zoo_path
        self.pre_trained = pre_trained
        self.pre_model = pre_model
        self.cut_at_pooling = cut_at_pooling
        self.last_conv_stride1 = last_conv_stride1
        if self.pre_trained: self.model_zoo_init = False
        
        self.model = getattr(torchvision.models, self.name, None)
        if self.model is None:
            raise TypeError("Base model not in trochvision")
        self.model = self.model()
        if self.model_zoo_init:
            self.load_pre_model(self.model, self.model_zoo_path)
        
        if self.cut_at_pooling:
            self.model = nn.Sequential(collections.OrderedDict(
                list(self.model._modules.items())[:-2]))
        
        if self.pre_trained:
            self.load_pre_model(self.model, self.pre_model)

        if self.last_conv_stride1:
            for module in self.model.layer4[0].modules():
                if isinstance(module, nn.Conv2d):
                    module.stride = (1, 1)
        # try:
        #     self.out_channels = self.model.layer4[-1].conv3.out_channels
        # except:
        #     self.out_channels = self.model.layer4[-1].conv2.out_channels
        self.out_channels = self.model.layer4[-1].expansion * 512

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    net = BackboneModel()
    net.print_params()
