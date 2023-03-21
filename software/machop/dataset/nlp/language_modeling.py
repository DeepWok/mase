import logging
import os
import pickle
from itertools import chain

import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def preprocess_datasets(
    raw_dataset,
    tokenizer,
    block_size=64,
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

    def __init__(self, dataset, split, tokenizer=None):
        if dataset is not None:
            self.dataset = dataset[split]
        super().__init__()

    def setup_tokenizer(self, tokenizer, max_token_count):
        self.tokenizer = tokenizer
        self.max_token_count = max_token_count

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        data_row = self.dataset[index]
        return dict(
            input_ids=torch.tensor(data_row["input_ids"]),
            attention_mask=torch.tensor(data_row["attention_mask"]),
            labels=torch.tensor(data_row["labels"]),
        )


class LanguageModelingDatasetWikitext2(LanguageModeling):
    def __init__(self, split="train", block_size=256):
        self.block_size = block_size
        self.split = split
        super().__init__(dataset=None, split=split)

    def setup_tokenizer(self, tokenizer, max_token_count):
        self.tokenizer = tokenizer
        self.max_token_count = max_token_count
        raw_dataset = self._prepare_data()
        name = f"./data/wikitext2/block_size{self.block_size}.pkl"
        if os.path.isfile(name):
            with open(name, "rb") as f:
                dataset = pickle.load(f)
        else:
            dataset = preprocess_datasets(
                raw_dataset, tokenizer, block_size=self.block_size
            )
            with open(name, "wb") as f:
                pickle.dump(dataset, f)
            print(f"Saved preprocessed dataset to disk, at {name}")
        self.dataset = dataset[self.split]

    def _prepare_data(self, path="./data/wikitext2"):
        if (path is not None) and (not os.path.isdir(path)):
            print("Downloading and processing dataset...")
            dataset = load_dataset(
                "wikitext",
                "wikitext-2-raw-v1",
                cache_dir=os.path.abspath("./cache/dataset_cache_dir"),
            )
            dataset.save_to_disk(path)
        else:
            print("Dataset already downloaded and processed")
            dataset = load_from_disk(path)
        return dataset


class LanguageModelingDatasetWikiText103(LanguageModeling):
    def __init__(self, split="train", block_size=256):
        self.block_size = block_size
        self.split = split
        super().__init__(dataset=None, split=split)

    def setup_tokenizer(self, tokenizer, max_token_count):
        self.tokenizer = tokenizer
        self.max_token_count = max_token_count
        raw_dataset = self._prepare_data()
        name = f"./data/wikitext103/block_size{self.block_size}.pkl"
        if os.path.isfile(name):
            with open(name, "rb") as f:
                dataset = pickle.load(f)
        else:
            dataset = preprocess_datasets(
                raw_dataset, tokenizer, block_size=self.block_size
            )
            with open(name, "wb") as f:
                pickle.dump(dataset, f)
            print(f"Saved preprocessed dataset to disk, at {name}")
        self.dataset = dataset[self.split]

    def _prepare_data(self, path="./data/wikitext103"):
        if (path is not None) and (not os.path.isdir(path)):
            print("Downloading and processing dataset...")
            dataset = load_dataset(
                "wikitext",
                "wikitext-103-raw-v1",
                cache_dir=os.path.abspath("./cache/dataset_cache_dir"),
            )
            dataset.save_to_disk(path)
        else:
            print("Dataset already downloaded and processed")
            dataset = load_from_disk(path)
        return dataset
