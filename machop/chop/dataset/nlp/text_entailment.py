import math
import os
import re
import string
from typing import Dict

import pytorch_lightning as pl
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset


class TextEntailDataset(Dataset):
    path = None
    num_classes = None

    # The mapping to update tokenizer's special token mapping
    # Some dataset contains special tokens like <unk> in the text
    # Keys should be in the list of predefined special attributes: [`bos_token`, `eos_token`, `unk_token`,
    # `sep_token`, `pad_token`, `cls_token`, `mask_token`, `additional_special_tokens`].
    special_token_mapping: Dict[str, str] = None

    sent1_col_name = None
    sent2_col_name = None
    label_col_name = None

    def __init__(self, split, tokenizer, max_token_len, num_workers=None) -> None:
        self.split = split
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.num_workers = num_workers
        self.data = None
        super().__init__()

    def _set_tokenizer(self, tokenizer, max_token_len):
        if tokenizer is None:
            tokenizer = self.tokenizer
        if max_token_len is None:
            max_token_len = self.max_token_len
        assert tokenizer is not None
        assert max_token_len is not None
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

        # some dataset contains special tokens in the text, which may be different from the tokenizer's
        # for example, HuggingFace ptb_text dataset contains '<unk>' as unknown token,
        # but bert-base-uncased uses '[UNK]' as unknown token.
        if self.special_token_mapping is not None:
            self.tokenizer.add_special_tokens(self.special_token_mapping)

    def prepare_data(self, tokenizer, max_token_len, num_workers=None):
        self._set_tokenizer(tokenizer, max_token_len)
        if os.path.isdir(self.path):
            dataset = load_from_disk(self.path)
            print(f"Local dataset is found on disk, at {self.path}")
        else:
            print("Downloading dataset...")
            dataset = self._download_or_load_raw_dataset()
            dataset.save_to_disk(self.path)
            print("Dataset is downloaded and saved to disk")

    def setup(self, tokenizer, max_token_len):
        self._set_tokenizer(tokenizer, max_token_len)
        assert os.path.isdir(self.path), f"The dataset dir {self.path} does not exist"
        self.data = load_from_disk(self.path)[self.split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
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

    def _download_or_load_raw_dataset(self):
        raise NotImplementedError


class TextEntailDatasetQNLI(TextEntailDataset):
    path = "./data/qnli"
    num_classes = 2

    sent1_col_name = "question"
    sent2_col_name = "sentence"
    label_col_name = "label"

    def __init__(self, split, tokenizer=None, max_token_len=None, auto_setup=False):
        super().__init__(
            split=split,
            tokenizer=tokenizer,
            max_token_len=max_token_len,
            num_workers=None,
        )

        if auto_setup:
            assert self.tokenizer is not None
            self.prepare_data(self.tokenizer, self.max_token_len)
            self.setup(self.tokenizer, self.max_token_len)
            print("Dataset is auto-setup")

    def _download_or_load_raw_dataset(self):
        if not os.path.isdir(self.path):
            print("Downloading dataset...")
            dataset = load_dataset(
                "glue", "qnli", cache_dir=os.path.abspath("./cache/dataset_cache_dir")
            )
            dataset.save_to_disk(self.path)
        else:
            print("Dataset is already downloaded")
            dataset = load_from_disk(self.path)
        print("Dataset loaded")
        return dataset


class TextEntailDatasetMNLI(TextEntailDataset):
    path = "./data/mnli"
    num_classes = 3
    sent1_col_name = "premise"
    sent2_col_name = "hypothesis"
    label_col_name = "label"

    def __init__(self, split, tokenizer=None, max_token_len=None, auto_setup=False):
        super().__init__(
            split=split,
            tokenizer=tokenizer,
            max_token_len=max_token_len,
            num_workers=None,
        )

        if auto_setup:
            assert self.tokenizer is not None
            self.prepare_data(self.tokenizer, self.max_token_len)
            self.setup(self.tokenizer, self.max_token_len)
            print("Dataset is auto-setup")

    def _download_or_load_raw_dataset(self):
        if not os.path.isdir(self.path):
            print("Downloading dataset...")
            dataset = load_dataset(
                "glue", "mnli", cache_dir=os.path.abspath("./cache/dataset_cache_dir")
            )
            dataset.save_to_disk(self.path)
        else:
            print("Dataset is already downloaded")
            dataset = load_from_disk(self.path)
        print("Dataset loaded")
        return dataset


class TextEntailDatasetBoolQ(TextEntailDataset):
    """
    Subset of SuperGLUE

    """

    path = "./data/boolq"
    num_classes = 2
    sent1_col_name = "passage"
    sent2_col_name = "question"
    label_col_name = "label"

    def __init__(self, split, tokenizer=None, max_token_len=None, auto_setup=False):
        super().__init__(
            split=split,
            tokenizer=tokenizer,
            max_token_len=max_token_len,
            num_workers=None,
        )

        if auto_setup:
            assert self.tokenizer is not None
            self.prepare_data(self.tokenizer, self.max_token_len)
            self.setup(self.tokenizer, self.max_token_len)
            print("Dataset is auto-setup")

    def _download_or_load_raw_dataset(self):
        if not os.path.isdir(self.path):
            print("Downloading dataset...")
            dataset = load_dataset(
                "super_glue",
                "boolq",
                cache_dir=os.path.abspath("./cache/dataset_cache_dir"),
            )
            dataset.save_to_disk(self.path)
        else:
            print("Dataset is already downloaded")
            dataset = load_from_disk(self.path)
        print("Dataset loaded")
        return dataset
