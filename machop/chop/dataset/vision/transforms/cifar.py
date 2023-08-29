from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms as tv_transforms

# CIFAR10 and CIFAR100
# -----------------------------------------

DEFAULT_CIFAR_PREPROCESS_ARGS = {
    "input_size": 32,
    "color_jitter": 0.4,
    "auto_augment": "rand-m9-mstd0.5-inc1",
    "interpolation": "bicubic",
    "re_prob": 0.25,
    "re_mode": "pixel",
    "re_count": 1,
}


# FIXME: Should we use (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD) for testing transform?
# We can find CIFAR10 and CIFAR mean and std from:
# https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151?permalink_comment_id=2851662#gistcomment-2851662
#
# CIFAR10:
#   mean: tensor([0.4914, 0.4822, 0.4465])
#   std: tensor([0.2470, 0.2435, 0.2616])
# CIFAR100:
#   mean: tensor([0.5071, 0.4865, 0.4409])
#   std: tensor([0.2673, 0.2564, 0.2762])
#
# CIFAR10_DEFAULT_MEAN = (0.4914, 0.4822, 0.4465)
# CIFAR10_DEFAULT_STD = (0.2470, 0.2435, 0.2616)
# CIFAR100_DEFAULT_MEAN = (0.5071, 0.4865, 0.4409)
# CIFAR100_DEFAULT_STD = (0.2673, 0.2564, 0.2762)


def _get_cifar_default_transform(train: bool, mean: tuple[float], std: tuple[float]):
    if train:
        transform = create_transform(
            **DEFAULT_CIFAR_PREPROCESS_ARGS,
            is_training=True,
        )
        transform.transforms[0] = tv_transforms.RandomCrop(
            DEFAULT_CIFAR_PREPROCESS_ARGS["input_size"], padding=4
        )
    else:
        transform_list = [
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean, std),
        ]
        transform = tv_transforms.Compose(transform_list)

    return transform


def get_cifar10_default_transform(train: bool) -> tv_transforms.Compose:
    return _get_cifar_default_transform(
        train, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    )


def get_cifar100_default_transform(train: bool) -> tv_transforms.Compose:
    return _get_cifar_default_transform(
        train, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    )


def get_cifar10_transform(train: bool, model: str = None):
    if model is None:
        return get_cifar10_default_transform(train)
    else:
        # Currently no model-dependent transform for CIFAR10 is supported.
        return get_cifar10_default_transform(train)


def get_cifar100_transform(train: bool, model: str = None):
    if model is None:
        return get_cifar100_default_transform(train)
    else:
        # Currently no model-dependent transform for CIFAR100 is supported.
        return get_cifar100_default_transform(train)
