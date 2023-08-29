import math
import os
import re
import string
from typing import Dict

import pytorch_lightning as pl
import torch

# from datasets import load_dataset, load_from_disk
import datasets as hf_datasets
from torch.utils.data import Dataset


class TextEntailmentDatasetBase(Dataset):
    info = None

    test_dataset_available: bool = False
    pred_dataset_available: bool = False

    # The mapping to update tokenizer's special token mapping
    # Some dataset contains special tokens like <unk> in the text
    # Keys should be in the list of predefined special attributes: [`bos_token`, `eos_token`, `unk_token`,
    # `sep_token`, `pad_token`, `cls_token`, `mask_token`, `additional_special_tokens`].
    special_token_mapping: Dict[str, str] = None

    sent1_col_name = None
    sent2_col_name = None
    label_col_name = None

    def __init__(
        self,
        split: str,
        tokenizer,
        max_token_len: int,
        num_workers: int,
        load_from_cache_file: bool = True,
        auto_setup: bool = True,
    ) -> None:
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.num_workers = num_workers
        self.load_from_cache_file = load_from_cache_file
        self.data = None

        if self.special_token_mapping is not None:
            self.tokenizer.add_special_tokens(self.special_token_mapping)

        if auto_setup:
            self.prepare_data()
            self.setup()

    def _download_dataset(self) -> hf_datasets.DatasetDict:
        raise NotImplementedError

    def prepare_data(self):
        self._download_dataset()

    def setup(self):
        self.data = self._download_dataset()[self.split]

    def __len__(self):
        if self.data is None:
            raise ValueError(
                "Dataset is not setup. Please call `dataset.prepare_data()` + `dataset.setup()` or pass `auto_setup=True` before using the dataset."
            )
        return len(self.data)

    def __getitem__(self, index):
        if self.data is None:
            raise ValueError(
                "Dataset is not setup. Please call `dataset.prepare_data()` + `dataset.setup()` or pass `auto_setup=True` before using the dataset."
            )
        data_row = self.data[index]
        question = data_row[self.sent1_col_name]
        answer = data_row[self.sent2_col_name]
        labels = data_row[self.label_col_name]
        encoding = self.tokenizer(
            question,
            answer,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()
        input_dict = dict(
            question=question,
            answer=answer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=torch.tensor([labels]),
        )
        if "token_type_ids" in self.tokenizer.model_input_names:
            input_dict["token_type_ids"] = encoding["token_type_ids"].flatten()

        return input_dict


class TextEntailmentDatasetQNLI(TextEntailmentDatasetBase):
    info = {"num_classes": 2}

    test_dataset_available = False
    pred_dataset_available = True

    sent1_col_name = "question"
    sent2_col_name = "sentence"
    label_col_name = "label"

    def __init__(
        self,
        split: str,
        tokenizer,
        max_token_len: int,
        num_workers: int,
        load_from_cache_file: bool = True,
        auto_setup: bool = True,
    ) -> None:
        super().__init__(
            split,
            tokenizer,
            max_token_len,
            num_workers,
            load_from_cache_file,
            auto_setup,
        )

    def _download_dataset(self) -> hf_datasets.DatasetDict:
        dataset_dict = hf_datasets.load_dataset("glue", "qnli")
        return dataset_dict


class TextEntailmentDatasetMNLI(TextEntailmentDatasetBase):
    info = {"num_classes": 3}

    test_dataset_available = False
    pred_dataset_available = True

    sent1_col_name = "premise"
    sent2_col_name = "hypothesis"
    label_col_name = "label"

    def __init__(
        self,
        split: str,
        tokenizer,
        max_token_len: int,
        num_workers: int,
        load_from_cache_file: bool = True,
        auto_setup: bool = True,
    ) -> None:
        super().__init__(
            split,
            tokenizer,
            max_token_len,
            num_workers,
            load_from_cache_file,
            auto_setup,
        )

    def _download_dataset(self) -> hf_datasets.DatasetDict:
        dataset_dict = hf_datasets.load_dataset("glue", "mnli")
        return dataset_dict


class TextEntailmentDatasetBoolQ(TextEntailmentDatasetBase):
    """
    Subset of SuperGLUE
    """

    info = {"num_classes": 2}

    test_dataset_available = True
    pred_dataset_available = False

    sent1_col_name = "passage"
    sent2_col_name = "question"
    label_col_name = "label"

    def _download_dataset(self) -> hf_datasets.DatasetDict:
        dataset_dict = hf_datasets.load_dataset("super_glue", "boolq")
        return dataset_dict
