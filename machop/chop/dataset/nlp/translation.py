import os
from typing import Dict

import datasets as hf_datasets
from torch.utils.data import Dataset
from torchnlp.datasets import multi30k_dataset


class TranslationDatasetBase(Dataset):
    info = {"num_classes": None}

    test_dataset_available: bool = False
    pred_dataset_available: bool = False

    # The mapping to update tokenizer's special token mapping
    # Some dataset contains special tokens like <unk> in the text
    # Keys should be in the list of predefined special attributes: [`bos_token`, `eos_token`, `unk_token`,
    # `sep_token`, `pad_token`, `cls_token`, `mask_token`, `additional_special_tokens`].
    special_token_mapping: Dict[str, str] = None

    src_col_name = None
    trg_col_name = None

    def __init__(
        self,
        split: str,
        tokenizer,
        max_token_len: int,
        num_workers: int,
        load_from_cache_file: bool = True,
        auto_setup: bool = True,
    ):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.num_workers = num_workers
        self.data = None
        self.load_from_cache_file = load_from_cache_file

        if self.special_token_mapping is not None:
            self.tokenizer.add_special_tokens(self.special_token_mapping)

        if auto_setup:
            self.prepare_data()
            self.setup()

    def _download_dataset(self):
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
        data_row = self.data[index]["translation"]
        source = data_row[self.src_col_name]
        target = data_row[self.trg_col_name]
        encoding = self.tokenizer(
            source,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()
        label_encoding = self.tokenizer(
            source,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        decoder_input_ids = label_encoding["input_ids"].flatten()
        decoder_attention_mask = label_encoding["attention_mask"].flatten()
        return dict(
            source=source,
            target=target,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )


# It seems like the download link of this is broken ..
class TranslationDatasetIWSLT2017_EN_DE(TranslationDatasetBase):
    test_dataset_available = True
    pred_dataset_available = False

    src_col_name = "en"
    trg_col_name = "de"

    def _download_dataset(self):
        dataset_dict = hf_datasets.load_dataset("iwslt2017", "iwslt2017-en-de")
        return dataset_dict


class TranslationDatasetIWSLT2017_DE_EN(TranslationDatasetBase):
    test_dataset_available = True
    pred_dataset_available = False

    src_col_name = "de"
    trg_col_name = "en"

    def _download_dataset(self):
        dataset_dict = hf_datasets.load_dataset("iwslt2017", "iwslt2017-de-en")
        return dataset_dict


class TranslationDatasetIWSLT2017_EN_FR(TranslationDatasetBase):
    test_dataset_available = True
    pred_dataset_available = False

    src_col_name = "en"
    trg_col_name = "fr"

    def _download_dataset(self):
        dataset_dict = hf_datasets.load_dataset("iwslt2017", "iwslt2017-en-fr")
        return dataset_dict


class TranslationDatasetIWSLT2017_EN_CH(TranslationDatasetBase):
    test_dataset_available = True
    pred_dataset_available = False

    src_col_name = "en"
    trg_col_name = "ch"

    def _download_dataset(self):
        dataset_dict = hf_datasets.load_dataset("iwslt2017", "iwslt2017-en-ch")
        return dataset_dict


class TranslationDatasetWMT19_DE_EN(TranslationDatasetBase):
    test_dataset_available = True
    pred_dataset_available = False

    src_col_name = "de"
    trg_col_name = "en"

    def _download_dataset(self):
        dataset_dict = hf_datasets.load_dataset("wmt19", "de-en")
        return dataset_dict


class TranslationDatasetWMT19_ZH_EN(TranslationDatasetBase):
    test_dataset_available = True
    pred_dataset_available = False

    src_col_name = "zh"
    trg_col_name = "en"

    def _download_dataset(self):
        dataset_dict = hf_datasets.load_dataset("wmt19", "zh-en")
        return dataset_dict


class TranslationDatasetWMT16_RO_EN(TranslationDatasetBase):
    test_dataset_available = True
    pred_dataset_available = False

    src_col_name = "ro"
    trg_col_name = "en"

    def _download_dataset(self):
        dataset_dict = hf_datasets.load_dataset("wmt16", "ro-en")
        return dataset_dict
