#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Dengpan Fu (v-defu@microsoft.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np

import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name=''):
        self.name = name
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_checkpoint(net, fpath, epoch=None, optimizer=None, 
    replace=True):
    if not replace and os.path.exists(fpath):
        print("snapshot: {} is already exist".format(fpath))
        return
    if hasattr(net, 'module'):
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()
    mkdir_if_missing(os.path.dirname(fpath))
    dict_to_save = {'state_dict': state_dict}
    if not epoch is None: dict_to_save['epoch'] = epoch
    if not optimizer is None:
        dict_to_save['optim_dict'] = optimizer.state_dict()
    torch.save(dict_to_save, fpath)
    print("checkpoint: {} saved.".format(fpath))

def resume_params(net, fpath, optimizer=None):
    if not os.path.exists(fpath):
        print("resume model file {} is non-exist".format(fpath))
        return
    checkpoint = torch.load(fpath)
    epoch = checkpoint.get('epoch', 0)
    param_dict = checkpoint.get('state_dict', None)
    if not param_dict is None:
        net.load_state_dict(param_dict)
        print("network params resumed from: {}".format(fpath))
    if not optimizer is None:
        optim_dict = checkpoint.get('optim_dict', None)
        if not optim_dict is None:
            optim_dict.load_state_dict(optim_dict)
            print("optimizer params resumed from: {}".format(fpath))
    return epoch

def set_bn_fix(m):
    """ Set BN layer params fixed """
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False
        m.eval()

def set_bn_release(m):
    """ Release BN layer params """
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=True
        m.train()

def fix_layer_params(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False
        m.eval()
    else:
        for p in m.parameters():
            p.requires_grad = False

def tensor2im(tensor):
    pass