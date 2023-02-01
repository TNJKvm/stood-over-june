import os.path
import random
import ntpath
from typing import Tuple, Any

import numpy as np
import torch
from torch import nn

import src.models

module_map = {
    'CNN': src.models.SmallCNN,
    'RESNET18': src.models.ResNet18,
    'RESNET18_CIFAR100': src.models.ResNet18_Imagenet,
    'ResNet50': src.models.ResNet50_Imagenet,
    'ResNet34': src.models.ResNet34_Imagenet,
    'vgg11': src.models.vgg11,
    'mobilenet': src.models.mobilenet_v2,
    'googlenet': src.models.googlenet,
    'inception': src.models.inception_v3,
    'densenet121': src.models.densenet121
}


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model(path: str) -> tuple[nn.Module, str, int, str]:
    if not os.path.exists(path):
        raise ValueError('Module path does not exists')

    properties = ntpath.basename(path).split(sep='.')[0].split(sep='_')
    model_type = properties[0]
    dataset = properties[1]
    seed = properties[-1]
    model = properties[2]
    if dataset == 'CIFAR100' and model == 'RESNET18':
        model = module_map['RESNET18_CIFAR100'](classes=100)
    elif dataset == 'CIFAR100' and model != 'RESNET18':
        model = module_map[model](classes=100)
    else:
        model = module_map[model]()

    model.load_state_dict(torch.load(path, map_location='cpu'))

    return model, model_type, int(seed), dataset
