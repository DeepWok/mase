import os
from pathlib import Path

from .blender import LegoNeRFDataset


def get_nerf_dataset(name: str, path: os.PathLike, split: str):
    assert split in [
        "train",
        "validation",
        "test",
        "pred",
    ], f"Unknown split {split}, should be one of train, validation, test, pred"

    match name:
        case "nerf-lego":
            path = path.joinpath("nerf_synthetic/lego")
            dataset = LegoNeRFDataset(path, split)
        case _:
            raise ValueError(f"Unknown dataset {name}")

    return dataset


NERF_DATASET_MAPPING = {
    "nerf-lego": LegoNeRFDataset,
}


def get_nerf_dataset_cls(name: str):
    assert name in NERF_DATASET_MAPPING, f"Unknown dataset {name}"
    return NERF_DATASET_MAPPING[name.lower()]
