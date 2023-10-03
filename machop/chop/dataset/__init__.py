import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..tools.registry import MACHOP_CACHE_DIR
from .nlp import NLP_DATASET_MAPPING, get_nlp_dataset, get_nlp_dataset_cls
from .toy_dataset import TOY_DATASET_MAPPING, get_toy_dataset, get_toy_dataset_cls
from .vision import VISION_DATASET_MAPPING, get_vision_dataset, get_vision_dataset_cls
from .physical import (
    PHYSICAL_DATASET_MAPPING,
    get_physical_dataset,
    get_physical_dataset_cls,
)

DATASET_CACHE_DIR = MACHOP_CACHE_DIR / "dataset"


def get_dataset_info(name: str):
    """
    Args:
        name (str): name of the dataset
    Returns:
        info (dict): information about the dataset.
        For vision datasets, keys are ["num_classes", "image_size"].
        For NLP datasets, keys are ["num_classes"].
    """
    name = name.lower()
    if name in TOY_DATASET_MAPPING:
        return get_toy_dataset_cls(name).info
    elif name in VISION_DATASET_MAPPING:
        return get_vision_dataset_cls(name).info
    elif name in NLP_DATASET_MAPPING:
        return get_nlp_dataset_cls(name).info
    elif name in PHYSICAL_DATASET_MAPPING:
        return get_physical_dataset_cls(name).info
    else:
        raise ValueError(f"Dataset {name} is not supported")


def get_dataset(
    name: str,
    split: bool,
    tokenizer=None,
    max_token_len: int = None,
    num_workers: int = os.cpu_count(),
    load_from_cache_file: bool = True,
    auto_setup: bool = True,
    model_name: str = None,
):
    """
    Args:
        name (str): name of the dataset
        path (str): path to the dataset
        train (bool): whether the dataset is used for training
        model_name (Optional[str, None]): name of the model. Some pretrained models have model-dependent transforms for training and evaluation.
    Returns:
        dataset (torch.utils.data.Dataset): dataset (with transforms)
    """
    global DATASET_CACHE_DIR
    MACHOP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    assert split in [
        "train",
        "validation",
        "test",
        "pred",
    ], f"Unknown split {split}, should be one of train, validation, test, pred"

    name = name.lower()
    if name in TOY_DATASET_MAPPING:
        dataset = get_toy_dataset(name, split)
    elif name in PHYSICAL_DATASET_MAPPING:
        path = DATASET_CACHE_DIR / name
        dataset = get_physical_dataset(name, path, split)
    elif name in VISION_DATASET_MAPPING:
        path = DATASET_CACHE_DIR / name
        dataset = get_vision_dataset(name, path, split, model_name)
    elif name in NLP_DATASET_MAPPING:
        path = DATASET_CACHE_DIR / name
        dataset = get_nlp_dataset(
            name,
            split=split,
            tokenizer=tokenizer,
            max_token_len=max_token_len,
            num_workers=num_workers,
            load_from_cache_file=load_from_cache_file,
            auto_setup=auto_setup,
        )
    else:
        raise ValueError(f"Dataset {name} is not supported")
    return dataset


AVAILABLE_DATASETS = (
    list(VISION_DATASET_MAPPING.keys())
    + list(NLP_DATASET_MAPPING.keys())
    + list(TOY_DATASET_MAPPING.keys())
    + list(PHYSICAL_DATASET_MAPPING.keys())
)


class MaseDataModule(pl.LightningDataModule):
    """
    MaseDataModule is a PyTorch Lightning DataModule that provides a unified interface to load datasets.

    Note than MaseDataModule requires .prepare_data() and .setup() to be called before .train_dataloader(), .val_dataloader(), .test_dataloader(), and .pred_dataloader()
    if the data module will not be passed to a PyTorch Lightning Trainer.
    """

    def __init__(
        self,
        name: str,
        batch_size: int,
        num_workers: int,
        max_token_len: int = None,
        tokenizer=None,
        load_from_cache_file: bool = True,
        model_name: str = None,
    ) -> None:
        super().__init__()

        self.name = name
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.num_workers = num_workers
        self.load_from_cache_file = load_from_cache_file
        self.model_name = model_name

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.pred_dataset = None
        self.dataset_info = get_dataset_info(name)

    def prepare_data(self) -> None:
        train_dataset = get_dataset(
            self.name,
            split="train",
            tokenizer=self.tokenizer,
            max_token_len=self.max_token_len,
            num_workers=self.num_workers,
            load_from_cache_file=self.load_from_cache_file,
            auto_setup=False,
            model_name=self.model_name,
        )
        val_dataset = get_dataset(
            self.name,
            split="validation",
            tokenizer=self.tokenizer,
            max_token_len=self.max_token_len,
            num_workers=self.num_workers,
            load_from_cache_file=self.load_from_cache_file,
            auto_setup=False,
            model_name=self.model_name,
        )
        test_dataset = get_dataset(
            self.name,
            split="test",
            tokenizer=self.tokenizer,
            max_token_len=self.max_token_len,
            num_workers=self.num_workers,
            load_from_cache_file=self.load_from_cache_file,
            auto_setup=False,
            model_name=self.model_name,
        )
        pred_dataset = get_dataset(
            self.name,
            split="pred",
            tokenizer=self.tokenizer,
            max_token_len=self.max_token_len,
            num_workers=self.num_workers,
            load_from_cache_file=self.load_from_cache_file,
            auto_setup=False,
            model_name=self.model_name,
        )

        if self.dataset_info.requires_preprocessing:
            train_dataset.prepare_data()
            if not self.dataset_info.preprocess_one_split_for_all:
                val_dataset.prepare_data()
                if test_dataset is not None:
                    test_dataset.prepare_data()
                if pred_dataset is not None:
                    pred_dataset.prepare_data()

    def setup(self, stage: str = None) -> None:
        if stage in ["fit", None]:
            self.train_dataset = get_dataset(
                self.name,
                split="train",
                tokenizer=self.tokenizer,
                max_token_len=self.max_token_len,
                num_workers=self.num_workers,
                load_from_cache_file=self.load_from_cache_file,
                auto_setup=True,
                model_name=self.model_name,
            )
            self.train_dataset.setup()
        if stage in ["fit", "validate", None]:
            self.val_dataset = get_dataset(
                self.name,
                split="validation",
                tokenizer=self.tokenizer,
                max_token_len=self.max_token_len,
                num_workers=self.num_workers,
                load_from_cache_file=True,
                auto_setup=True,
                model_name=self.model_name,
            )
            self.val_dataset.setup()
        if stage in ["test", None]:
            self.test_dataset = get_dataset(
                self.name,
                split="test",
                tokenizer=self.tokenizer,
                max_token_len=self.max_token_len,
                num_workers=self.num_workers,
                load_from_cache_file=True,
                auto_setup=True,
                model_name=self.model_name,
            )
            if self.test_dataset is not None:
                self.test_dataset.setup()
        if stage in ["predict", None]:
            self.pred_dataset = get_dataset(
                self.name,
                split="pred",
                tokenizer=self.tokenizer,
                max_token_len=self.max_token_len,
                num_workers=self.num_workers,
                load_from_cache_file=True,
                auto_setup=True,
                model_name=self.model_name,
            )
            if self.pred_dataset is not None:
                self.pred_dataset.setup()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError(
                "The test dataset is not available"
                "probably because the test set does not have ground truth labels, "
                "or the test dataset does not exist. For the former case, try predict_dataloader"
            )
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def pred_dataloader(self) -> DataLoader:
        if self.pred_dataset is None:
            raise RuntimeError("The pred dataset is not available.")
        return DataLoader(
            self.pred_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
