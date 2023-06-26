import numpy as np
import torch
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    def __init__(self, split="train", num_samples=512) -> None:
        super().__init__()
        self.num_samples = num_samples

        if split == "train":
            rng = np.random.RandomState(0)
        elif split == "validation":
            rng = np.random.RandomState(1)
        elif split == "test":
            rng = np.random.RandomState(2)
        else:
            raise RuntimeError(
                f"split must be `train`, `test`, or `validation`, but got {split}"
            )

        self.data = (rng.rand(num_samples, 4) - 0.5) * 10
        self.labels = np.zeros((num_samples, 1))
        for i in range(num_samples):
            self.labels[i, :] = np.sum(self.data[i, ...]) > 0

    def __getitem__(self, index):
        data_i = torch.tensor(self.data[index, ...], dtype=torch.float32).reshape(
            1, 2, 2
        )
        label_i = torch.tensor(self.labels[index, ...], dtype=torch.long).squeeze()
        return data_i, label_i

    def __len__(self):
        return self.num_samples
