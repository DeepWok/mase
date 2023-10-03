import os
from .jsc import JetSubstructureDataset, get_jsc_dataset


def get_physical_dataset(name: str, path: os.PathLike, split: str):
    assert split in ["train", "validation", "test", "pred"]

    match name:
        case "jsc":
            path = os.path.join(
                path,
                "processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z",
            )
            get_jsc_dataset(path)
            config_file = os.path.abspath(
                f"./chop/dataset/physical/jsc/yaml_IP_OP_config.yml"
            )
            # Dataset config
            dataset = JetSubstructureDataset(path, config_file, split=split)
        case _:
            raise ValueError(f"Unknown dataset {name}")

    return dataset


PHYSICAL_DATASET_MAPPING = {
    "jsc": JetSubstructureDataset,
}


def get_physical_dataset_cls(name: str):
    assert name in PHYSICAL_DATASET_MAPPING, f"Unknown dataset {name}"
    return PHYSICAL_DATASET_MAPPING[name.lower()]
