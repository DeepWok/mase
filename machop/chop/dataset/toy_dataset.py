import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import MaseDatasetInfo, add_dataset_info


@add_dataset_info(
    name="toy_tiny",
    dataset_source="manual",
    available_splits=("train", "validation", "test", "pred"),
    image_classification=True,
    num_classes=2,
    image_size=(1, 2, 2),
)
class ToyTinyDataset(Dataset):
    def __init__(self, split="train", num_samples: int = 10240) -> None:
        super().__init__()
        self.num_samples = num_samples

        if split == "train":
            rng = np.random.RandomState(0)
        elif split == "validation":
            rng = np.random.RandomState(1)
        elif split == "test":
            rng = np.random.RandomState(2)
        elif split == "pred":
            rng = np.random.RandomState(3)
        else:
            raise RuntimeError(
                f"split must be `train`, `test`, `validation`, or `split`, but got {split}"
            )

        self.data = (rng.rand(num_samples, 4) - 0.5) * 2
        self.labels = (np.sum(self.data, axis=1) > 0).astype(np.int64)

        self.data = self.data.reshape(num_samples, 1, 2, 2)

    def __getitem__(self, index):
        data_i = torch.tensor(self.data[index, ...], dtype=torch.float32)
        label_i = torch.tensor(self.labels[index, ...], dtype=torch.long)
        return data_i, label_i

    def __len__(self):
        return self.num_samples

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        pass


def get_toy_dataset(name: str, split: str, num_samples: int = 10240):
    assert split in ["train", "validation", "test", "pred"]

    match name:
        case "toy_tiny":
            dataset = ToyTinyDataset(split=split, num_samples=num_samples)
        case _:
            raise ValueError(f"Unknown dataset {name}")

    return dataset


TOY_DATASET_MAPPING = {
    "toy_tiny": ToyTinyDataset,
}


def get_toy_dataset_cls(name: str):
    assert name in TOY_DATASET_MAPPING, f"Unknown dataset {name}"
    return TOY_DATASET_MAPPING[name]
