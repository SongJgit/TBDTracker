"""set gpus and random seeds."""

import os
import random
import numpy as np
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn


def select_device(device):
    """ set device
    Args:
        device: str, 'cpu' or '0' or '1,2,3'-like

    Return:
        torch.device

    """

    if device == 'cpu':
        logger.info('Use CPU for training')

    elif ',' in device:  # multi-gpu
        logger.error('Multi-GPU currently not supported')

    else:
        logger.info(f'set gpu {device}')
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available()

    cuda = device != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    return device


class BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):

    def _check_input_dim(self, input):
        # The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
        # is this method that is overwritten by the sub-class
        # This original goal of this method was for tensor sanity checks
        # If you're ok bypassing those sanity checks (eg. if you trust your inference
        # to provide the right dimensional inputs), then you can just use this method
        # for easy conversion from SyncBatchNorm
        # (unfortunately, SyncBatchNorm does not store the original class - if it did
        #  we could return the one that was originally created)
        return


def revert_sync_batchnorm(module):
    # this is very similar to the function that it is trying to revert:
    # https://github.com/pytorch/pytorch/blob/c8b3686a3e4ba63dc59e5dcfe5db3430df256833/torch/nn/modules/batchnorm.py#L679
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        # new_cls = BatchNormXd
        module_output = BatchNormXd(module.num_features, module.eps, module.momentum, module.affine,
                                    module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, 'qconfig'):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output


class TracedModel(nn.Module):

    def __init__(self, model=None, device=None, img_size=(640, 640)):
        super(TracedModel, self).__init__()

        print(' Convert model to Traced-model... ')
        self.stride = model.stride
        self.names = model.names
        self.model = model

        self.model = revert_sync_batchnorm(self.model)
        self.model.to('cpu')
        self.model.eval()

        self.detect_layer = self.model.model[-1]
        self.model.traced = True

        rand_example = torch.rand(1, 3, img_size, img_size)

        traced_script_module = torch.jit.trace(self.model, rand_example, strict=False)
        # traced_script_module = torch.jit.script(self.model)
        # traced_script_module.save("traced_model.pt")
        print(' traced_script_module saved! ')
        self.model = traced_script_module
        self.model.to(device)
        self.detect_layer.to(device)
        print(' model is traced! \n')

    def forward(self, x, augment=False, profile=False):
        out = self.model(x)
        out = self.detect_layer(out)
        return out
