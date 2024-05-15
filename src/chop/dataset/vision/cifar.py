from torch.utils.data import Dataset
from torchvision import datasets
import os
import numpy as np
from ..utils import add_dataset_info


@add_dataset_info(
    name="cifar10",
    dataset_source="torchvision",
    available_splits=("train", "test"),
    image_classification=True,
    num_classes=10,
    image_size=(3, 32, 32),
)
class Cifar10Mase(datasets.CIFAR10):
    def __init__(
        self,
        root: os.PathLike,
        train: bool,
        transform: callable,
        download: bool,
        subset=False,
        subset_size_per_class=10,
    ) -> None:
        self.subset = subset
        self.subset_size_per_class = subset_size_per_class
        super().__init__(root, train=train, transform=transform, download=download)
        if subset:
            self.data, self.targets = self._create_subset()

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        pass

    def _create_subset(self):
        num_classes = 10
        data, targets = [], []
        target_array = np.array(self.targets)

        for i in range(num_classes):
            indices = np.where(target_array == i)[0]
            np.random.shuffle(indices)
            indices = indices[: self.subset_size_per_class]
            data.append(self.data[indices])
            targets.extend([i] * self.subset_size_per_class)

        data = np.concatenate(data, axis=0)
        return data, targets


@add_dataset_info(
    name="cifar100",
    dataset_source="torchvision",
    available_splits=("train", "test"),
    image_classification=True,
    num_classes=100,
    image_size=(3, 32, 32),
)
class Cifar100Mase(datasets.CIFAR100):
    test_dataset_available: bool = True
    pred_dataset_available: bool = False

    info = {
        "num_classes": 100,
        "image_size": (3, 32, 32),
    }

    def __init__(
        self, root: os.PathLike, train: bool, transform: callable, download: bool
    ) -> None:
        super().__init__(root, train=train, transform=transform, download=download)

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        pass


def get_cifar_dataset(
    name: str, path: os.PathLike, train: bool, transform: callable, subset=False
) -> Dataset:
    match name.lower():
        case "cifar10":
            dataset = Cifar10Mase(
                path, train=train, transform=transform, download=True, subset=subset
            )
        case "cifar100":
            dataset = Cifar100Mase(
                path, train=train, transform=transform, download=True
            )
        case _:
            raise ValueError(f"Unknown dataset {name}")
    return dataset
