from torch.utils.data import Dataset
from torchvision import datasets
import os


class MNISTMase(datasets.MNIST):
    test_dataset_available: bool = True
    pred_dataset_available: bool = False
    info = {
        "num_classes": 10,
        "image_size": (1, 28, 28),
    }

    def __init__(
        self, root: os.PathLike, train: bool, transform: callable, download: bool
    ) -> None:
        super().__init__(root, train=train, transform=transform, download=download)

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        pass


def get_mnist_dataset(
    name: str, path: os.PathLike, train: bool, transform: callable
) -> Dataset:
    match name.lower():
        case "mnist":
            dataset = MNISTMase(path, train=train, transform=transform, download=True)
        case _:
            raise ValueError(f"Unknown dataset {name}")
    return dataset
