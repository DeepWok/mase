from functools import partial
from typing import Optional, Tuple, Union

import torch
import torchvision as tv
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import InterpolationMode, autoaugment
from torchvision.transforms import functional as tv_F
from torchvision.transforms import transforms
from torchvision.transforms._presets import ImageClassification

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

val_imagenet_tv_preprocess_cls_mapping = {
    "resnet18": tv.models.resnet.ResNet18_Weights.IMAGENET1K_V1.transforms,
    "resnet34": tv.models.resnet.ResNet34_Weights.IMAGENET1K_V1.transforms,
    "resnet50": tv.models.resnet.ResNet50_Weights.IMAGENET1K_V2.transforms,
    "resnet101": tv.models.resnet.ResNet101_Weights.IMAGENET1K_V2.transforms,
    "wideresnet50_2": tv.models.resnet.Wide_ResNet50_2_Weights.IMAGENET1K_V2.transforms,
    "mobilenetv2": tv.models.mobilenet.MobileNet_V2_Weights.IMAGENET1K_V2.transforms,
    "mobilenetv3_small": tv.models.mobilenet.MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms,
    "mobilenetv3_large": tv.models.mobilenet.MobileNet_V3_Large_Weights.IMAGENET1K_V2.transforms,
    "efficientnet_b0": tv.models.efficientnet.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms,
    "efficientnet_v2_s": tv.models.efficientnet.EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms,
    "efficientnet_v2_m": tv.models.efficientnet.EfficientNet_V2_M_Weights.IMAGENET1K_V1.transforms,
    "efficientnet_v2_l": tv.models.efficientnet.EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms,
}


class ClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
    ):
        trans = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(
                    autoaugment.RandAugment(
                        interpolation=interpolation, magnitude=ra_magnitude
                    )
                )
            elif auto_augment_policy == "ta_wide":
                trans.append(
                    autoaugment.TrivialAugmentWide(interpolation=interpolation)
                )
            elif auto_augment_policy == "augmix":
                trans.append(
                    autoaugment.AugMix(
                        interpolation=interpolation, severity=augmix_severity
                    )
                )
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(
                    autoaugment.AutoAugment(
                        policy=aa_policy, interpolation=interpolation
                    )
                )
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


train_imagenet_tv_preprocess_cls_mapping = {
    "efficientnet_v2_s": partial(
        ClassificationPresetTrain,
        crop_size=300,
        auto_augment_policy="ta_wide",
        random_erase_prob=0.1,
    ),
    "efficientnet_v2_m": partial(
        ClassificationPresetTrain,
        crop_size=300,
        auto_augment_policy="ta_wide",
        random_erase_prob=0.1,
    ),
    "efficientnet_v2_l": partial(
        ClassificationPresetTrain,
        crop_size=300,
        auto_augment_policy="ta_wide",
        random_erase_prob=0.1,
    ),
}


class PreprocessorForTVPretrainedModel:
    def __init__(self, tv_preprocess: torch.nn.Module) -> None:
        self.tv_preprocess = tv_preprocess()

    def __call__(self, image: Tensor):
        return self.tv_preprocess(image)


def build_imagenet_transform(model_name, train):
    if train:
        if model_name in train_imagenet_tv_preprocess_cls_mapping:
            return train_imagenet_tv_preprocess_cls_mapping[model_name]()
        else:
            return build_default_imagenet_transform(train)
    else:
        if model_name in val_imagenet_tv_preprocess_cls_mapping:
            return PreprocessorForTVPretrainedModel(
                val_imagenet_tv_preprocess_cls_mapping[model_name]
            )
        else:
            # return PreprocessorForTVPretrainedModel(
            #     imagenet_tv_preprocess_cls_mapping[model_name]
            # )
            return build_default_imagenet_transform(train=False)
