# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from .cifar import get_cifar_dataset
from .imagenet import get_imagenet_dataset

from .transform import build_transform


default_preprocess = {
    'input_size': 32,
    'color_jitter': 0.4,
    'auto_augment': 'rand-m9-mstd0.5-inc1',
    'interpolation': 'bicubic',
    're_prob': 0.25,
    're_mode': 'pixel',
    're_count': 1,
}

info = {
    'cifar10': {
        'num_classes': 10,
        'process': default_preprocess,
    },
    'cifar100': {
        'num_classes': 100,
        'process': default_preprocess,
    },
    'imagenet': {
        'num_classes': 1000,
        'process': default_preprocess,
    },
}


def build_dataset(dataset_name, path, train):
    preprocess_params = info[dataset_name]['process']
    transform = build_transform(train, **preprocess_params)

    if dataset_name in ['cifar10', 'cifar100']:
        dataset = get_cifar_dataset(
            dataset_name, path, train, transform)
    elif dataset_name in ['imagenet']:
        dataset = get_imagenet_dataset(
            dataset_name, path, train, transform)
    return dataset, info[dataset_name]

