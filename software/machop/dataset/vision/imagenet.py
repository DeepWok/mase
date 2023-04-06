import os

import torchvision as tv


def get_imagenet_dataset(name, path, train, transform):
    if name in ["imagenet", "IMAGENET"]:
        root = os.path.join(path, "train" if train else "val")
        dataset = tv.datasets.ImageFolder(root, transform=transform)
    else:
        raise ValueError(f"Unknown dataset {name}")
    return dataset
