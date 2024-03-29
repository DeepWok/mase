from torchvision import transforms as tv_transforms

# MNIST
# -----------------------------------------


def lutnet_transform(img):
    """
    MNIST provided by PyTorch is a list of images with pixel values in the range [0, 1]

    MNIST provided by TensorFlow is a list of images with pixel values in the range [0, 255]

    This function converts the PyTorch representation to the TensorFlow representation, in order to recreate the results from the LUTNet paper
    """
    img = img * 255
    return img


def convert_to_3_channels(img):
    """
    Converts a single-channel image to a 3-channel image by replicating the single channel
    across all three channels.

    The input 'img' is expected to be a PyTorch tensor.
    """
    img = tv_transforms.ToPILImage()(img)

    img = img.convert("RGB")

    img = tv_transforms.ToTensor()(img)
    return img


def _get_mnist_default_transform():
    transform_list = [
        tv_transforms.ToTensor(), 
        tv_transforms.Lambda(lutnet_transform),
        tv_transforms.Lambda(convert_to_3_channels),
    ]
    transform = tv_transforms.Compose(transform_list)
    # transform_list = [tv_transforms.ToTensor(), tv_transforms.Lambda(lutnet_transform)]
    # transform = tv_transforms.Compose(transform_list)
    return transform


def get_mnist_default_transform(train: bool) -> tv_transforms.Compose:
    return _get_mnist_default_transform()


def get_mnist_transform(train: bool, model: str = None):
    if model is None:
        return get_mnist_default_transform(train)
    else:
        # Currently no model-dependent transform for mnist is supported.
        return get_mnist_default_transform(train)
