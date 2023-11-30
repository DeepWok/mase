import copy
from dataclasses import dataclass
from functools import partial
import logging
from itertools import chain
from typing import Dict, Sequence

import torch
import datasets as hf_datasets
from huggingface_hub import snapshot_download, hf_hub_download
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import transformers

from ..utils import add_dataset_info


logger = logging.getLogger(__name__)


def preprocess_datasets_causal_lm(
    dataset_dict: hf_datasets.DatasetDict,
    tokenizer,
    block_size: int,
    num_workers: int,
    load_from_cache_file: bool = True,
    text_column_name="text",
):
    try:
        column_names = dataset_dict["train"].column_names
    except KeyError:
        splits = list(dataset_dict.keys())
        column_names = dataset_dict[splits[0]].column_names
        logger.info(
            f"Dataset doesn't have a 'train' split. Use the first split {splits[0]} instead."
        )
    assert (
        text_column_name in column_names
    ), f"Need a column named '{text_column_name}' to run the tokenizer on the dataset"

    if block_size is None:
        block_size = tokenizer.model_max_length

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    dataset_dict = dataset_dict.map(
        tokenize_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
        # keep_in_memory=True,
        load_from_cache_file=load_from_cache_file,
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

    dataset_dict = dataset_dict.map(
        group_texts,
        batched=True,
        num_proc=num_workers,
        desc=f"Grouping texts in chunks of {block_size}",
        # keep_in_memory=True,
        load_from_cache_file=load_from_cache_file,
    )
    return dataset_dict


class LanguageModelingBase(Dataset):
    info = None  # MaseDatasetInfo
    # The mapping to update tokenizer's special token mapping
    # Some dataset contains special tokens like <unk> in the text
    # Keys should be in the list of predefined special attributes: [`bos_token`, `eos_token`, `unk_token`,
    # `sep_token`, `pad_token`, `cls_token`, `mask_token`, `additional_special_tokens`].
    special_token_mapping: Dict[str, str] = None
    text_column_name = "text"

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
        dataset_dict = self._download_dataset()
        preprocess_datasets_causal_lm(
            dataset_dict,
            tokenizer=self.tokenizer,
            block_size=self.max_token_len,
            num_workers=self.num_workers,
            load_from_cache_file=self.load_from_cache_file,
            text_column_name=self.text_column_name,
        )

    def setup(self):
        dataset_dict = self._download_dataset()
        dataset_dict = preprocess_datasets_causal_lm(
            dataset_dict,
            tokenizer=self.tokenizer,
            block_size=self.max_token_len,
            num_workers=self.num_workers,
            load_from_cache_file=True,
            text_column_name=self.text_column_name,
        )
        self.data = dataset_dict[self.split]

    def __len__(self):
        if self.data is None:
            raise ValueError(
                "Dataset is not setup. Please call `dataset.prepare_data()` + `dataset.setup()` or pass `auto_setup=True` before using the dataset."
            )
        return self.data.num_rows

    def __getitem__(self, index):
        if self.data is None:
            raise ValueError(
                "Dataset is not setup. Please call `dataset.prepare_data()` + `dataset.setup()` or pass `auto_setup=True` before using the dataset."
            )
        data_row = self.data[index]
        return dict(
            input_ids=torch.tensor(data_row["input_ids"]),
            attention_mask=torch.tensor(data_row["attention_mask"]),
            labels=torch.tensor(data_row["labels"]),
        )


@add_dataset_info(
    name="wikitext2",
    dataset_source="hf_datasets",
    available_splits=("train", "validation", "test"),
    causal_LM=True,
)
class LanguageModelingDatasetWikitext2(LanguageModelingBase):
    def _download_dataset(self) -> hf_datasets.DatasetDict:
        dataset_dict = hf_datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
        return dataset_dict


@add_dataset_info(
    name="wikitext103",
    dataset_source="hf_datasets",
    available_splits=("train", "validation", "test"),
    causal_LM=True,
)
class LanguageModelingDatasetWikitext103(LanguageModelingBase):
    def _download_dataset(self) -> hf_datasets.DatasetDict:
        dataset_dict = hf_datasets.load_dataset("wikitext", "wikitext-103-raw-v1")
        return dataset_dict


@add_dataset_info(
    name="c4",
    dataset_source="hf_datasets",
    available_splits=("train", "validation"),
    causal_LM=True,
)
class LanguageModelingDatasetC4(LanguageModelingBase):
    def _download_dataset(self) -> hf_datasets.DatasetDict:
        dataset_dict = hf_datasets.load_dataset("c4", "en")
        return dataset_dict


@add_dataset_info(
    name="ptb",
    dataset_source="hf_datasets",
    available_splits=("train", "validation", "test"),
    causal_LM=True,
)
class LanguageModelingDatasetPTB(LanguageModelingBase):
    text_column_name = "sentence"

    def _download_dataset(self) -> hf_datasets.DatasetDict:
        dataset_dict = hf_datasets.load_dataset("ptb_text_only")
        return dataset_dict


# @add_dataset_info(
#     name="scienceqa",
#     dataset_source="hf_datasets",
#     available_splits=("train", "validation", "test"),
#     causal_LM=True,
# )
# class LanguageModelingDatasetScienceQA(LanguageModelingBase):
#     # *: prompting / preprocessing required
#     def _download_dataset(self) -> hf_datasets.DatasetDict:
#         dataset_dict = hf_datasets.load_dataset("metaeval/scienceqa_text_only")
#         return dataset_dict


@dataclass
class DataCollatorForCausalLMAlpaca:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    IGNORE_INDEX = -100

    def __call__(self, instances: list[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


@add_dataset_info(
    name="alpaca",
    dataset_source="hf_datasets",
    available_splits=("train", "validation"),
    causal_LM=True,
    data_collator_cls=DataCollatorForCausalLMAlpaca,
)
class LanguageModelingDatasetAlpaca(LanguageModelingBase):
    IGNORE_INDEX = -100

    def _download_dataset(self) -> hf_datasets.DatasetDict:
        dataset_dict = hf_datasets.load_dataset("tatsu-lab/alpaca")
        return dataset_dict

    @staticmethod
    def _preprocess(example, tokenizer, max_length, ignore_id):
        def _tokenize(text, tokenizer, max_length):
            return tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=max_length,
                truncation=True,
            )

        prompt = example["text"].removesuffix(example["output"])
        target = example["text"]

        prompt_tokenized = _tokenize(prompt, tokenizer, max_length)["input_ids"][0]
        target_tokenized = _tokenize(target, tokenizer, max_length)["input_ids"][0]
        input_ids = copy.deepcopy(target_tokenized)

        prompt_len = prompt_tokenized.ne(tokenizer.pad_token_id).sum().item()
        target_tokenized[:prompt_len] = ignore_id
        return dict(
            input_ids=input_ids,
            labels=target_tokenized,
        )

    def prepare_data(self):
        dataset_dict = self._download_dataset()

        # Add special tokens
        special_tokens_dict = {}
        if self.tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = "[PAD]"
        if self.tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = "</s>"
        if self.tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = "<s>"
        if self.tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = "<unk>"

        dataset_dict.map(
            function=partial(
                self._preprocess,
                tokenizer=self.tokenizer,
                max_length=self.max_token_len,
                ignore_id=self.IGNORE_INDEX,
            ),
            num_proc=self.num_workers,
            load_from_cache_file=self.load_from_cache_file,
            desc="Preprocessing dataset",
        )

    def setup(self):
        dataset_dict = self._download_dataset()
        dataset_dict = dataset_dict["train"].train_test_split(test_size=0.1, seed=42)
        dataset_dict = hf_datasets.DatasetDict(
            {
                "train": dataset_dict["train"],
                "validation": dataset_dict["test"],
            }
        )
        dataset_dict = dataset_dict.map(
            function=partial(
                self._preprocess,
                tokenizer=self.tokenizer,
                max_length=self.max_token_len,
                ignore_id=self.IGNORE_INDEX,
            ),
            num_proc=self.num_workers,
            load_from_cache_file=True,
            desc="Preprocessing dataset",
        )
        self.data = dataset_dict[self.split]

    def __getitem__(self, index):
        if self.data is None:
            raise ValueError(
                "Dataset is not setup. Please call `dataset.prepare_data()` + `dataset.setup()` or pass `auto_setup=True` before using the dataset."
            )
        data_row = self.data[index]
        return dict(
            input_ids=torch.tensor(data_row["input_ids"]),
            labels=torch.tensor(data_row["labels"]),
        )


@dataclass
class DataCollatorForCausalLMMMLU:
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int = 2048
    target_max_len: int = 128
    train_on_source: bool = False
    predict_with_generate: bool = False

    IGNORE_INDEX = -100

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [
            f"{self.tokenizer.bos_token}{example['input']}" for example in instances
        ]
        targets = [
            f"{example['output']}{self.tokenizer.eos_token}" for example in instances
        ]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt["input_ids"], tokenized_targets["input_ids"]
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor(
                            [self.IGNORE_INDEX for _ in range(len(tokenized_source))]
                            + copy.deepcopy(tokenized_target)
                        )
                    )
                else:
                    labels.append(
                        torch.tensor(copy.deepcopy(tokenized_source + tokenized_target))
                    )
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = (
            pad_sequence(labels, batch_first=True, padding_value=self.IGNORE_INDEX)
            if not self.predict_with_generate
            else None
        )
        data_dict = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict["labels"] = labels
        return data_dict


@add_dataset_info(
    name="mmlu-0-shot",
    dataset_source="hf_datasets",
    available_splits=("validation", "test"),
    causal_LM=True,
    data_collator_cls=DataCollatorForCausalLMMMLU,
)
class LanguageModelingDatasetMMLUZeroShot(LanguageModelingBase):
    IGNORE_INDEX = -100

    def _download_dataset(self) -> hf_datasets.DatasetDict:
        dataset_dict = hf_datasets.load_dataset("Cheng98/mmlu", "zero-shot")
        return dataset_dict

    def prepare_data(self):
        special_tokens_dict = {}
        if self.tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = "[PAD]"
        if self.tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = "</s>"
        if self.tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = "<s>"
        if self.tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = "<unk>"

        self.tokenizer.add_special_tokens(special_tokens_dict)

    def setup(self):
        dataset_dict = self._download_dataset()
        self.data = dataset_dict[self.split]

    def __getitem__(self, index):
        if self.data is None:
            raise ValueError(
                "Dataset is not setup. Please call `dataset.prepare_data()` + `dataset.setup()` or pass `auto_setup=True` before using the dataset."
            )
        return self.data[index]
