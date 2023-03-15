from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms


def build_transform(
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
