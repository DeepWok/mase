import os
from pathlib import Path
import torchvision as tv


class ImageNetMase(tv.datasets.ImageFolder):
    info = {
        "num_classes": 1000,
        "image_size": (3, 224, 224),
    }

    test_dataset_available: bool = False
    pred_dataset_available: bool = False

    def __init__(self, root: os.PathLike, transform: callable) -> None:
        super().__init__(root, transform=transform)

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        pass


def get_imagenet_dataset(
    name: str, path: os.PathLike, train: bool, transform: callable
) -> tv.datasets.ImageFolder:
    match name.lower():
        case "imagenet":
            imagenet_dir = Path(path)
            root = imagenet_dir / ("train" if train else "val")
            if not root.exists():
                raise RuntimeError(
                    f"ImageNet dataset not found at {root}, please download it manually first."
                )
            dataset = ImageNetMase(root, transform=transform)
        case _:
            raise ValueError(f"Unknown dataset {name}")
    return dataset
