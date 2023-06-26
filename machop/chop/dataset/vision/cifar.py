from torchvision import datasets


def get_cifar_dataset(name, path, train, transform):
    if name in ["cifar10", "CIFAR10"]:
        dataset = datasets.CIFAR10(
            path, train=train, transform=transform, download=True
        )
    elif name in ["cifar100", "CIFAR100"]:
        dataset = datasets.CIFAR100(
            path, train=train, transform=transform, download=True
        )
    else:
        raise ValueError(f"Unknown dataset {name}")
    return dataset
