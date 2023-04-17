import logging
import os
from itertools import chain
from typing import Dict

import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def preprocess_datasets(
    raw_dataset,
    tokenizer,
    block_size=256,
    overwrite_cache=False,
    preprocessing_num_workers=4,
):
    column_names = raw_dataset["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if block_size is None:
        block_size = tokenizer.model_max_length

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not overwrite_cache,
        desc="Running tokenizer on dataset",
        keep_in_memory=True,
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
        keep_in_memory=True,
    )
    return dataset


class LanguageModeling(Dataset):
    num_classes = None
    raw_path = None

    # The mapping to update tokenizer's special token mapping
    # Some dataset contains special tokens like <unk> in the text
    # Keys should be in the list of predefined special attributes: [`bos_token`, `eos_token`, `unk_token`,
    # `sep_token`, `pad_token`, `cls_token`, `mask_token`, `additional_special_tokens`].
    special_token_mapping: Dict[str, str] = None

    def __init__(self, split, tokenizer, block_size, num_workers):
        self.split = split
        self.tokenizer = tokenizer
        self.max_token_len = block_size
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
        # set tokenizer, download and save raw dataset, preprocess raw dataset and save to disk
        self._set_tokenizer(tokenizer, max_token_len)
        processed_dataset_dir = os.path.join(
            self.raw_path,
            "processed_{}".format(self.tokenizer.name_or_path.replace("/", "_")),
        )
        if os.path.isdir(processed_dataset_dir):
            processed_dataset = load_from_disk(processed_dataset_dir)
            print(f"Processed dataset is found on disk, at {processed_dataset_dir}")
        else:
            raw_dataset = self._download_or_load_raw_dataset()
            print("Processing raw dataset...")
            processed_dataset = preprocess_datasets(
                raw_dataset,
                self.tokenizer,
                block_size=self.max_token_len,
                preprocessing_num_workers=self.num_workers
                if num_workers is None
                else num_workers,
            )
            processed_dataset.save_to_disk(processed_dataset_dir)
            print(f"Saved processed dataset to disk, at {processed_dataset_dir}")
        self.processed_dataset_dir = processed_dataset_dir

    def setup(self, tokenizer, max_token_len):
        if tokenizer is None:
            tokenizer = self.tokenizer
        if max_token_len is None:
            max_token_len = self.max_token_len
        self._set_tokenizer(tokenizer, max_token_len)
        # load processed dataset from disk
        # this will happen on every process in DataModule
        self.processed_dataset_dir = os.path.join(
            self.raw_path,
            "processed_{}".format(self.tokenizer.name_or_path.replace("/", "_")),
        )
        assert os.path.isdir(
            self.processed_dataset_dir
        ), f"The processed dataset dir does not exist: {self.processed_dataset_dir}"
        self.data = load_from_disk(self.processed_dataset_dir)[self.split]

    def __len__(self):
        return self.data.num_rows

    def __getitem__(self, index):
        data_row = self.data[index]
        return dict(
            input_ids=torch.tensor(data_row["input_ids"]),
            attention_mask=torch.tensor(data_row["attention_mask"]),
            labels=torch.tensor(data_row["labels"]),
        )

    def _download_or_load_raw_dataset(self):
        raise NotImplementedError


class LanguageModelingDatasetWikitext2(LanguageModeling):
    raw_path = "./data/wikitext2"

    def __init__(
        self,
        split="train",
        tokenizer=None,
        max_token_len=None,
        num_workers=4,
        auto_setup=False,
    ):
        super().__init__(
            split=split,
            tokenizer=tokenizer,
            block_size=max_token_len,
            num_workers=num_workers,
        )
        self.processed_dataset_dir = None
        if auto_setup:
            assert self.tokenizer is not None
            self.prepare_data(self.tokenizer, self.max_token_len)
            self.setup(self.tokenizer, self.max_token_len)
            print("Dataset is auto-setup")

    def _download_or_load_raw_dataset(self):
        if not os.path.isdir(self.raw_path):
            print("Downloading and processing raw dataset...")
            raw_dataset = load_dataset(
                "wikitext",
                "wikitext-2-raw-v1",
                cache_dir=os.path.abspath("./cache/dataset_cache_dir"),
            )
            raw_dataset.save_to_disk(self.raw_path)
        else:
            print("Raw dataset is already downloaded")
            raw_dataset = load_from_disk(self.raw_path)
        print("Raw dataset loaded")
        return raw_dataset


class LanguageModelingDatasetWikiText103(LanguageModeling):
    raw_path = "./data/wikitext103"

    def __init__(
        self,
        split="train",
        tokenizer=None,
        max_token_len=None,
        num_workers=4,
        auto_setup=False,
    ):
        super().__init__(
            split=split,
            tokenizer=tokenizer,
            block_size=max_token_len,
            num_workers=num_workers,
        )
        self.processed_dataset_dir = None
        if auto_setup:
            assert self.tokenizer is not None
            self.prepare_data(self.tokenizer, self.max_token_len)
            self.setup(self.tokenizer, self.max_token_len)
            print("Dataset is auto-setup")

    def _download_or_load_raw_dataset(self):
        if not os.path.isdir(self.raw_path):
            print("Downloading and processing raw dataset...")
            raw_dataset = load_dataset(
                "wikitext",
                "wikitext-103-raw-v1",
                cache_dir=os.path.abspath("./cache/dataset_cache_dir"),
            )
            raw_dataset.save_to_disk(self.raw_path)
        else:
            print("Raw dataset already downloaded")
            raw_dataset = load_from_disk(self.raw_path)
        print("Raw dataset loaded")
        return raw_dataset


class LanguageModelingDatasetC4(LanguageModeling):
    raw_path = "./data/c4"

    def __init__(
        self, split, tokenizer=None, max_token_len=None, num_workers=4, auto_setup=False
    ):
        super().__init__(
            split=split,
            tokenizer=tokenizer,
            block_size=max_token_len,
            num_workers=num_workers,
        )
        self.processed_dataset_dir = None
        if auto_setup:
            assert self.tokenizer is not None
            self.prepare_data(self.tokenizer, self.max_token_len)
            self.setup(self.tokenizer, self.max_token_len)
            print("Dataset is auto-setup")

    def _download_or_load_raw_dataset(self):
        if not os.path.isdir(self.raw_path):
            print("Downloading and processing raw dataset...")
            raw_dataset = load_dataset(
                "c4",
                "en",
                cache_dir=os.path.abspath("./cache/dataset_cache_dir"),
            )
            raw_dataset.save_to_disk(self.raw_path)
        else:
            print("Raw dataset is already downloaded")
            raw_dataset = load_from_disk(self.raw_path)
        print("Raw dataset loaded")
        return raw_dataset


class LanguageModelingDatasetPTB(LanguageModeling):
    raw_path = "./data/ptb"

    special_token_mapping = {"unk_token": "<unk>"}

    def __init__(
        self, split, tokenizer=None, max_token_len=None, num_workers=4, auto_setup=False
    ):
        super().__init__(
            split=split,
            tokenizer=tokenizer,
            block_size=max_token_len,
            num_workers=num_workers,
        )

        self.processed_dataset_dir = None
        if auto_setup:
            assert self.tokenizer is not None
            self.prepare_data(self.tokenizer, self.max_token_len)
            self.setup(self.tokenizer, self.max_token_len)
            print("Dataset is auto-setup")

    def _download_or_load_raw_dataset(self):
        if not os.path.isdir(self.raw_path):
            print("Downloading and processing raw dataset...")
            raw_dataset = load_dataset(
                "ptb_text_only",
                cache_dir=os.path.abspath("./cache/dataset_cache_dir"),
            )
            raw_dataset.save_to_disk(self.raw_path)
        else:
            print("Raw dataset is already downloaded")
            raw_dataset = load_from_disk(self.raw_path)
        print("Raw dataset loaded")
        return raw_dataset


# class LanuguageModelingDatasetBookCorpus(LanguageModeling):
#     def __init__(self, split="train", block_size=256):
#         self.block_size = block_size
#         self.split = split
#         super().__init__(dataset=None, split=split)

#     def setup_tokenizer(self, tokenizer, max_token_count):
#         self.tokenizer = tokenizer
#         self.max_token_count = max_token_count
#         raw_dataset = self._prepare_data()
#         name = f"./data/bookcorpus/block_size{self.block_size}.pkl"
#         if os.path.isfile(name):
#             with open(name, "rb") as f:
#                 dataset = pickle.load(f)
#         else:
#             dataset = preprocess_datasets(
#                 raw_dataset, tokenizer, block_size=self.block_size
#             )
#             with open(name, "wb") as f:
#                 pickle.dump(dataset, f)
#             print("Saved preprocessed dataset to disk, at {name}")
#         self.dataset = dataset[self.split]

#     def _prepare_data(self, path="./data/bookcorpus"):
#         if (path is not None) and (not os.path.isdir(path)):
#             print("Downloading and processing dataset...")
#             dataset = load_dataset(
#                 "bookcorpus", cache_dir=os.path.abspath("./cache/dataset_cache_dir")
#             )
#             dataset.save_to_disk(path)
#         else:
#             print("Dataset already downloaded and processed")
#             dataset = load_from_disk(path)
#         return dataset
