import logging
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms as tv_transforms
from torchvision.transforms import InterpolationMode
from .presets import ClassificationPresetTrain, ClassificationPresetEval

logger = logging.getLogger(__name__)

DEFAULT_IMAGENET_PREPROCESS_ARGS = {
    "input_size": 224,
    "color_jitter": 0.4,
    "auto_augment": "rand-m9-mstd0.5-inc1",
    "interpolation": "bilinear",
    "re_prob": 0.25,
    "re_mode": "pixel",
    "re_count": 1,
}


def get_imagenet_default_transform(train: bool) -> tv_transforms.Compose:
    if train:
        transform = create_transform(
            **DEFAULT_IMAGENET_PREPROCESS_ARGS,
            is_training=True,
        )
    else:
        transform_list = [
            tv_transforms.Resize(size=256, interpolation=3),
            tv_transforms.CenterCrop(DEFAULT_IMAGENET_PREPROCESS_ARGS["input_size"]),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
        transform = tv_transforms.Compose(transform_list)
    return transform


# ===================================
# Checkpoint-dependent transforms
#
# To achieve the accuracy reported in torchvision, we need to use the same transforms,
# especially for the validation/test set.
# Refer to Torchvision recipe: https://github.com/pytorch/vision/blob/main/references/classification/presets.py
# ===================================

DEFAULT_PRESET_ARGS = {
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    "val_resize_size": 256,
    "val_crop_size": 224,
    "train_crop_size": 224,
    "interpolation": InterpolationMode("bilinear"),
    "auto_augment_policy": None,
    "random_erase_prob": 0.0,
    "ra_magnitude": 9,
    "augmix_severity": 3,
    "backend": "pil",
    "use_v2": False,
}


def get_model_dependent_transform_args(model_name: str) -> dict:
    match model_name.lower():
        case "resnet18" | "resnet34" | "resnet50" | "resnet101" | "resnet152":
            return DEFAULT_PRESET_ARGS | {}
        case "mobilenetv2":
            return DEFAULT_PRESET_ARGS | {}
        case "mobilenetv3_small" | "mobilenetv3_large":
            return DEFAULT_PRESET_ARGS | {
                "auto_augment_policy": "imagenet",
                "random_erase_prob": 0.2,
            }
        case "efficientnet_b0" | "efficientnet_b3":
            # https://github.com/pytorch/vision/tree/main/references/classification#efficientnet-v1
            logger.info(
                f"model {model_name} is ported from timm, but the corresponding transform is ported yet. Use the torchvision's default one instead. "
                "Refer to https://github.com/pytorch/vision/tree/main/references/classification#efficientnet-v1"
            )
            return DEFAULT_PRESET_ARGS | {
                "interpolation": InterpolationMode("bicubic"),
            }
        case "efficientnet_v2_s":
            return DEFAULT_PRESET_ARGS | {
                "auto_augment_policy": "ta_wide",
                "random_erase_prob": 0.1,
                "train_crop_size": 300,
                "eval_crop_size": 384,
            }
        case "efficientnet_v2_m":
            return DEFAULT_PRESET_ARGS | {
                "auto_augment_policy": "ta_wide",
                "random_erase_prob": 0.1,
                "train_crop_size": 384,
                "eval_crop_size": 480,
            }
        case "efficientnet_v2_l":
            # refer to: https://github.com/pytorch/vision/tree/main/references/classification#efficientnet-v2
            logger.info(
                f"model {model_name} is ported from original paper, but the corresponding transform is ported yet. Use the torchvision's default one instead. "
                "Refer to https://github.com/pytorch/vision/tree/main/references/classification#efficientnet-v2"
            )
            return DEFAULT_PRESET_ARGS | {
                "interpolation": InterpolationMode("bicubic"),
                "train_crop_size": 480,
                "eval_crop_size": 480,
                "mean": (0.5, 0.5, 0.5),
                "std": (0.5, 0.5, 0.5),
            }
        case _:
            raise NotImplementedError(
                f"model-dependent transform is not implemented for `{model_name}`."
            )


def get_imagenet_model_dependent_transform(train: bool, model_name: str):
    model_transform_args = get_model_dependent_transform_args(model_name)
    if train:
        transform = ClassificationPresetTrain(
            mean=model_transform_args["mean"],
            std=model_transform_args["std"],
            crop_size=model_transform_args["train_crop_size"],
            interpolation=model_transform_args["interpolation"],
            auto_augment_policy=model_transform_args["auto_augment_policy"],
            random_erase_prob=model_transform_args["random_erase_prob"],
            ra_magnitude=model_transform_args["ra_magnitude"],
            augmix_severity=model_transform_args["augmix_severity"],
            backend=model_transform_args["backend"],
            use_v2=model_transform_args["use_v2"],
        )
    else:
        transform = ClassificationPresetEval(
            mean=model_transform_args["mean"],
            std=model_transform_args["std"],
            crop_size=model_transform_args["val_crop_size"],
            resize_size=model_transform_args["val_resize_size"],
            interpolation=model_transform_args["interpolation"],
            backend=model_transform_args["backend"],
            use_v2=model_transform_args["use_v2"],
        )
    return transform


def get_imagenet_transform(train: bool, model_name: str):
    if model_name is None:
        transform = get_imagenet_default_transform(train)
    else:
        try:
            transform = get_imagenet_model_dependent_transform(train, model_name)
            logger.info(
                "Model-dependent imagenet transform is available. Use model-dependent transform."
            )
        except NotImplementedError:
            transform = get_imagenet_default_transform(train)
    return transform
