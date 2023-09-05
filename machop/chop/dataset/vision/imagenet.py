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

    def __init__(self, root: os.PathLike, transform: callable, subset=False) -> None:
        if subset:
            root = self._create_subset_dataset(root)
        super().__init__(root, transform=transform)

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        pass

    # If the user requests a subset version of ImageNet, then sample only 10 images
    # from each class.
    def _create_subset_dataset(self, root: os.PathLike) -> os.PathLike:
        # Check if the root directory points to the validation or train folder
        root = Path(root)
        subset_root = root.parent.parent / "imagenet_subset"
        if subset_root.exists():
            return subset_root / root.name

        # Create a tiny dataset with only 100 samples per class for each split
        sizes = [100, 20]
        dataset_dir = root.parent
        for i, split in enumerate(["train", "val"]):
            # Get all the class directories from the original dataset
            class_dirs = [d for d in (dataset_dir / split).iterdir() if d.is_dir()]

            for class_dir in class_dirs:
                subset_class_dir = subset_root / split / class_dir.name
                subset_class_dir.mkdir(parents=True, exist_ok=True)

                # NOTE: We don't pick randomly. We just go in the order they're listed.
                for j, img in enumerate(class_dir.iterdir()):
                    if j >= sizes[i]:
                        break
                    os.symlink(img, subset_class_dir / img.name)

        return subset_root / root.name


def get_imagenet_dataset(
    name: str, path: os.PathLike, train: bool, transform: callable, subset=False
) -> tv.datasets.ImageFolder:
    match name.lower():
        case "imagenet":
            imagenet_dir = Path(path)
            root = imagenet_dir / ("train" if train else "val")
            if not root.exists():
                raise RuntimeError(
                    f"ImageNet dataset not found at {root}, please download it manually first."
                )
            dataset = ImageNetMase(root, transform=transform, subset=subset)
        case _:
            raise ValueError(f"Unknown dataset {name}")
    return dataset
