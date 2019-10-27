#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Dengpan Fu (v-defu@microsoft.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import torch
from torch import nn

class BaseModel(nn.Module):
    """"""
    def __init__(self, name=''):
        super(BaseModel, self).__init__()
        self.name = name

    def forward(self, x):
        raise NotImplementedError

    def load_pre_model(self, net, fpath, strict=False):
        if fpath is not None:
            if os.path.isfile(fpath):
                state_dict = torch.load(fpath)
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                net.load_state_dict(state_dict, strict=strict)
                print('Pre-model: {} loaded!'.format(fpath))
            else:
                raise IOError("pre-model: {} is non-exists".format(fpath))
        else:
            print("pre-model is None, skip load parameters for {}".format(self.name))

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

    def param_groups(self, split=False):
        raise NotImplementedError
