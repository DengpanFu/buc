#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Dengpan Fu (v-defu@microsoft.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys; sys.path.insert(0, '../..')
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from .base_model import BaseModel
from .backbone_model import BackboneModel
from .utils import set_bn_fix, set_bn_release, fix_layer_params

class IDEModel(BaseModel):
    def __init__(self, name='ide', backbone='resnet50', num_classes=0, embed_dim=0, 
        dropout=0, has_bn=False, fix_base_bn=True, out_base_feat=False, 
        fix_part_layers=False, fixed_layers=None, backbone_model_zoo_init=True, 
        backbone_model_zoo_path=None, lmp=0):
        super(IDEModel, self).__init__(name=name)
        self.base = BackboneModel(name=backbone, model_zoo_init=backbone_model_zoo_init,
                                  model_zoo_path=backbone_model_zoo_path)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.has_bn = has_bn if self.embed_dim > 0 else False
        self.fix_base_bn = fix_base_bn
        self.fix_part_layers = fix_part_layers
        if fixed_layers is None or len(fixed_layers) < 1:
            # by default, we try to fix the first 3 layers of resnet
            self.fixed_layers = ['conv1', 'bn1', 'layer1', 'layer2']
        else:
            self.fixed_layers = fixed_layers
        self.out_base_feat = out_base_feat
        self.lmp = lmp
        self.num_features = self.base.out_channels
        if self.embed_dim > 0:
            self.embed_fc = nn.Linear(self.num_features, self.embed_dim)
            init.kaiming_normal_(self.embed_fc.weight, mode='fan_out')
            init.constant_(self.embed_fc.bias, 0)
            if self.has_bn:
                self.embed_bn = nn.BatchNorm1d(self.embed_dim)
                init.constant_(self.embed_bn.weight, 1)
                init.constant_(self.embed_bn.bias, 0)
            self.num_features = self.embed_dim
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.num_classes > 0:
            self.classifier = nn.Linear(self.num_features, self.num_classes)
            init.normal_(self.classifier.weight, std=0.001)
            init.constant_(self.classifier.bias, 0)
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)

    def forward(self, x, is_oim=False):
        x = self.base(x)

        if not self.training and self.out_base_feat:
            if self.lmp == 0:
                out = self.avgpool(x).view(x.size(0), -1)
            elif self.lmp >= 1:
                out = F.adaptive_max_pool2d(x, output_size=(self.lmp, 1)).view(x.size(0), -1)
            else:
                out = F.adaptive_avg_pool2d(x, output_size=(-self.lmp, 1)).view(x.size(0), -1)
            return out

        x = self.avgpool(x).view(x.size(0), -1)

        if self.embed_dim > 0:
            x = self.embed_fc(x)
            if self.has_bn: 
                x = self.embed_bn(x)

        feat = F.normalize(x)
        if self.dropout > 0:
            feat = self.drop(feat)
        if is_oim:
            return feat

        if self.embed_dim > 0:
            x = F.relu(x)
        if self.dropout > 0: 
            x = self.drop(x)

        if not self.training:
            return x

        if self.num_classes > 0:
            pred = self.classifier(x)
            return feat, pred
        else:
            return x

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode:
            if self.fix_base_bn:
                self.base.apply(set_bn_fix)
                # print("All BatchNorm layer fixed in base net")
            if self.fix_part_layers:
                fixed_layers = []
                for layer in self.fixed_layers:
                    module = getattr(self.base.model, layer, None)
                    if module is not None:
                        fixed_layers.append(layer)
                        module.apply(fix_layer_params)
                # if len(fixed_layers) > 0:
                #     print("Fixed layers: '{:s}' in base net".format(", ".join(fixed_layers)))

    def param_groups(self, split=False):
        if split:
            base_params = self.base.parameters()
            new_params = []
            if self.embed_dim > 0:
                new_params += list(self.embed_fc.parameters())
                if self.has_bn:
                    new_params += list(self.embed_bn.parameters())
            if self.num_classes > 0:
                new_params += list(self.classifier.parameters())
            return base_params, new_params
        else:
            return self.parameters()

    def print_params(self, concise=True):
        for name, v in self.named_parameters():
            if concise:
                print("{:<36}  mean:{:<12.4e}  std:{:<12.4e} grad: {}".format(
                    name, v.mean().item(), v.std().item(), v.requires_grad))
            else:
                print("{:<40}  min:{:<12.4e}  max:{:<12.4e}  " \
                      "mean:{:<12.4e}  std:{:<12.4e} grad: {}".format(
                        name, v.min().item(), v.max().item(), 
                        v.mean().item(), v.std().item(), v.requires_grad))


class BUCModel(IDEModel):
    def __init__(self, name='ide', backbone='resnet50', num_classes=0, embed_dim=0, 
        dropout=0, has_bn=True, fix_base_bn=False, out_base_feat=False, 
        fix_part_layers=False, fixed_layers=None, backbone_model_zoo_init=True, 
        backbone_model_zoo_path='/home/dengpanfu/.torch/models/resnet50-19c8e357.pth', lmp=0):
        super(BUCModel, self).__init__(name, backbone, num_classes, 
                                       embed_dim, dropout, has_bn, 
                                       fix_base_bn, out_base_feat, 
                                       fix_part_layers, fixed_layers, 
                                       backbone_model_zoo_init, 
                                       backbone_model_zoo_path, lmp)
        # self.num_features = self.base.out_channels

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        x = self.base(x)

        if not self.training and self.out_base_feat:
            if self.lmp == 0:
                out = self.avgpool(x).view(x.size(0), -1)
            elif self.lmp >= 1:
                out = F.adaptive_max_pool2d(x, output_size=(self.lmp, 1)).view(x.size(0), -1)
            else:
                out = F.adaptive_avg_pool2d(x, output_size=(-self.lmp, 1)).view(x.size(0), -1)
            return out

        x = self.avgpool(x).view(x.size(0), -1)
        feat = F.normalize(x)

        if self.embed_dim > 0:
            x = self.embed_fc(x)
            if self.has_bn: 
                x = self.embed_bn(x)

        x = F.normalize(x)

        if self.dropout > 0:
            x = self.drop(x)

        # return feat, x
        return x, feat
