from .cifar import get_cifar100_transform, get_cifar10_transform
from .imagenet import get_imagenet_transform


def get_vision_dataset_transform(name: str, train: bool, model_name: str):
    """
    Args:
        name (str): name of the dataset
        train (bool): whether the dataset is used for training
        model_name (Optional[str, None]): name of the model. Some pretrained models have model-dependent transforms.
    Returns:
        transform (callable): transform function
    """
    match name.lower():
        case "cifar10":
            return get_cifar10_transform(train, model_name)
        case "cifar100":
            return get_cifar100_transform(train, model_name)
        case "imagenet" | "imagenet_subset":
            return get_imagenet_transform(train, model_name)
        case _:
            raise ValueError(f"Unknown dataset {name}")
