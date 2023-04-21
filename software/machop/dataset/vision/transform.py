from functools import partial
from typing import Optional, Tuple, Union

import torch
import torchvision as tv
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tv_F

# -----------------------------------------
# CIFAR10 and CIFAR100
# -----------------------------------------

cifar_preprocess = {
    "input_size": 32,
    "color_jitter": 0.4,
    "auto_augment": "rand-m9-mstd0.5-inc1",
    "interpolation": "bicubic",
    "re_prob": 0.25,
    "re_mode": "pixel",
    "re_count": 1,
}


imagenet_preprocess = {
    "input_size": 224,
    "color_jitter": 0.4,
    "auto_augment": "rand-m9-mstd0.5-inc1",
    "interpolation": "bilinear",
    "re_prob": 0.25,
    "re_mode": "pixel",
    "re_count": 1,
}


def _generic_build_transform(
    train,
    input_size,
    color_jitter,
    auto_augment,
    interpolation,
    re_prob,
    re_mode,
    re_count,
):
    resize_im = input_size > 32
    if train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=color_jitter,
            auto_augment=auto_augment,
            interpolation=interpolation,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(input_size, padding=4)
        return transform

    transform_list = []
    if resize_im:
        size = int((256 / 224) * input_size)
        transform_list.append(
            transforms.Resize(
                size, interpolation=3
            ),  # to maintain same ratio w.r.t. 224 images
        )
        transform_list.append(transforms.CenterCrop(input_size))

    transform_list.append(transforms.ToTensor())
    transform_list.append(
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    )
    return transforms.Compose(transform_list)


build_cifar10_transform = partial(_generic_build_transform, **cifar_preprocess)

build_cifar100_transform = partial(_generic_build_transform, **cifar_preprocess)

build_default_imagenet_transform = partial(
    _generic_build_transform, **imagenet_preprocess
)

# -----------------------------------------
# IMAGENET's transform depends on the model if pretrained weights are used

imagenet_tv_preprocess_cls_mapping = {
    "resnet18": tv.models.resnet.ResNet18_Weights.IMAGENET1K_V1.transforms,
    "resnet34": tv.models.resnet.ResNet34_Weights.IMAGENET1K_V1.transforms,
    "resnet50": tv.models.resnet.ResNet50_Weights.IMAGENET1K_V2.transforms,
    "resnet101": tv.models.resnet.ResNet101_Weights.IMAGENET1K_V2.transforms,
    "wideresnet50_2": tv.models.resnet.Wide_ResNet50_2_Weights.IMAGENET1K_V2.transforms,
    "mobilenetv3_small": tv.models.mobilenet.MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms,
    "mobilenetv3_large": tv.models.mobilenet.MobileNet_V3_Large_Weights.IMAGENET1K_V2.transforms,
    "efficientnet_v2_s": tv.models.efficientnet.EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms,
    "efficientnet_v2_m": tv.models.efficientnet.EfficientNet_V2_M_Weights.IMAGENET1K_V1.transforms,
    "efficientnet_v2_l": tv.models.efficientnet.EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms,
}


class PreprocessorForTVPretrainedModel:
    def __init__(self, tv_preprocess: torch.nn.Module) -> None:
        self.tv_preprocess = tv_preprocess()

    def __call__(self, image: Tensor):
        return self.tv_preprocess(image)


def build_imagenet_transform(model_name, train):
    if train:
        return build_default_imagenet_transform(train)
    else:
        if model_name in imagenet_tv_preprocess_cls_mapping:
            return PreprocessorForTVPretrainedModel(
                imagenet_tv_preprocess_cls_mapping[model_name]
            )
        else:
            # return PreprocessorForTVPretrainedModel(
            #     imagenet_tv_preprocess_cls_mapping[model_name]
            # )
            return build_default_imagenet_transform(train=False)
