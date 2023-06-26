# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from torchvision.transforms import InterpolationMode

from .cifar import get_cifar_dataset
from .imagenet import get_imagenet_dataset
from .transform import (
    build_cifar10_transform,
    build_cifar100_transform,
    build_imagenet_transform,
)

info = {
    "cifar10": {
        "num_classes": 10,
        "image_size": (3, 32, 32),
    },
    "cifar100": {
        "num_classes": 100,
        "image_size": (3, 32, 32),
    },
    "imagenet": {
        "num_classes": 1000,
        "image_size": (3, 224, 224),
    },
}


def build_dataset(dataset_name, model_name, path, train):
    if dataset_name == "cifar10":
        transform = build_cifar10_transform(train)
    elif dataset_name == "cifar100":
        transform = build_cifar100_transform(train)
    elif dataset_name in ["imagenet"]:
        transform = build_imagenet_transform(model_name, train)
    else:
        raise RuntimeError(f"Unsupported dataset_name {dataset_name}")

    if dataset_name in ["cifar10", "cifar100"]:
        dataset = get_cifar_dataset(dataset_name, path, train, transform)
    elif dataset_name in ["imagenet"]:
        dataset = get_imagenet_dataset(dataset_name, path, train, transform)
    return dataset, info[dataset_name]
