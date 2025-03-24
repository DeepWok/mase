from pathlib import Path

from .speech_recognition import CondensedLibrispeechASRDataset

def get_audio_dataset(name: str, path: Path, split: str, **kwargs):
    """
    Args:
        name (str): name of the dataset
        path (os.PathLike): path to the dataset
        split (str): dataset split ("train", "validation", "test", or "pred")
        **kwargs: additional dataset-specific arguments
    Returns:
        dataset (torch.utils.data.Dataset): The requested dataset
    """
    assert split in [
        "train",
        "validation",
        "test",
        "pred",
    ], f"Unknown split {split}, should be one of train, validation, test, pred"

    match name:
        case "librispeech_asr" | "nyalpatel/condensed_librispeech_asr":
            dataset = CondensedLibrispeechASRDataset(path, split=split, **kwargs)
        case _:
            raise ValueError(f"Unknown dataset {name}")

    return dataset


AUDIO_DATASET_MAPPING = {
    "librispeech_asr": CondensedLibrispeechASRDataset,
    "nyalpatel/condensed_librispeech_asr": CondensedLibrispeechASRDataset,
}


def get_audio_dataset_cls(name: str):
    assert name in AUDIO_DATASET_MAPPING, f"Unknown dataset {name}"
    return AUDIO_DATASET_MAPPING[name.lower()]