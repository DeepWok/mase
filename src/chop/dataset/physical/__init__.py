from pathlib import Path
from .jsc import JetSubstructureDataset


def get_physical_dataset(name: str, path: Path, split: str):
    assert split in ["train", "validation", "test", "pred"]

    match name:
        case "jsc":
            # h5 file path
            path = path.joinpath(
                "processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z"
            )
            dataset = JetSubstructureDataset(path, split=split)
        case _:
            raise ValueError(f"Unknown dataset {name}")

    return dataset


PHYSICAL_DATASET_MAPPING = {
    "jsc": JetSubstructureDataset,
}


def get_physical_dataset_cls(name: str):
    assert name in PHYSICAL_DATASET_MAPPING, f"Unknown dataset {name}"
    return PHYSICAL_DATASET_MAPPING[name.lower()]
