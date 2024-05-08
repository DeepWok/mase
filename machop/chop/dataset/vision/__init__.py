import os
from pathlib import Path

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

from .mnist import get_mnist_dataset, MNISTMase
from .cifar import get_cifar_dataset, Cifar10Mase, Cifar100Mase
from .imagenet import get_imagenet_dataset, ImageNetMase
from .transforms import get_vision_dataset_transform


def get_vision_dataset(name: str, path: os.PathLike, split: str, model_name: str):
    """
    Args:
        name (str): name of the dataset
        path (str): path to the dataset
        train (bool): whether the dataset is used for training
        model_name (Optional[str, None]): name of the model. Some pretrained models have
        model-dependent transforms for training and evaluation.
    Returns:
        dataset (torch.utils.data.Dataset): dataset (with transforms)
    """
    assert split in [
        "train",
        "validation",
        "test",
        "pred",
    ], f"Unknown split {split}, should be one of train, validation, test, pred"

    train = split == "train"
    transform = get_vision_dataset_transform(name, train, model_name)

    match name:
        case "mnist":
            dataset = get_mnist_dataset(name, path, train, transform)
        case "cifar10" | "cifar100":
            dataset = get_cifar_dataset(name, path, train, transform)
        case "cifar10_subset":
            name = name.replace("_subset", "")
            path = Path(str(path).replace("_subset", ""))
            dataset = get_cifar_dataset(name, path, train, transform, subset=True)
        case "imagenet":
            dataset = get_imagenet_dataset(name, path, train, transform)
        case "imagenet_subset":
            # NOTE: We just repurpose the routine for ImageNet. You'll find that the
            # subset dataset is created in the ImageNetMase class constructor. :)
            name = name.replace("_subset", "")
            path = Path(str(path).replace("_subset", ""))
            dataset = get_imagenet_dataset(name, path, train, transform, subset=True)

    return dataset


VISION_DATASET_MAPPING = {
    "mnist": MNISTMase,
    "cifar10": Cifar10Mase,
    "cifar10_subset": Cifar10Mase,
    "cifar100": Cifar100Mase,
    "imagenet": ImageNetMase,
    # A subset of ImageNet w/ 100 train and 20 val images per class (1000 classes)
    "imagenet_subset": ImageNetMase,
}


def get_vision_dataset_cls(name: str):
    assert name in VISION_DATASET_MAPPING, f"Unknown dataset {name}"
    return VISION_DATASET_MAPPING[name.lower()]
