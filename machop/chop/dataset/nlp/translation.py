import os
from typing import Dict

from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from torchnlp.datasets import multi30k_dataset


class TranslationDataset(Dataset):
    path = None
    num_classes = None

    # The mapping to update tokenizer's special token mapping
    # Some dataset contains special tokens like <unk> in the text
    # Keys should be in the list of predefined special attributes: [`bos_token`, `eos_token`, `unk_token`,
    # `sep_token`, `pad_token`, `cls_token`, `mask_token`, `additional_special_tokens`].
    special_token_mapping: Dict[str, str] = None

    src_col_name = None
    trg_col_name = None

    def __init__(self, split, tokenizer, max_token_len, num_workers):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.num_workers = num_workers
        self.data = None

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

    def _download_or_load_raw_dataset(self):
        raise NotImplementedError


# It seems like the download link of this is broken ..
class TranslationDatasetIWSLT2017_EN_DE(TranslationDataset):
    path = "./data/iwslt2017-en-de"

    src_col_name = "en"
    trg_col_name = "de"

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

    def _download_or_load_raw_dataset(self):
        if not os.path.isdir(self.path):
            print("Downloading dataset...")
            dataset = load_dataset(
                "iwslt2017",
                "iwslt2017-en-de",
                cache_dir=os.path.abspath("./cache/dataset_cache_dir"),
            )
            dataset.save_to_disk(self.path)
        else:
            print("Dataset is already downloaded")
            dataset = load_from_disk(self.path)
        print("Dataset loaded")
        return dataset


class TranslationDatasetIWSLT2017_DE_EN(TranslationDataset):
    path = "./data/iwslt2017-de-en"

    def __init__(self, split="train"):
        super().__init__()
        self.src_col_name = "de"
        self.trg_col_name = "en"
        self.split = split

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

    def _download_or_load_raw_dataset(self):
        if not os.path.isdir(self.path):
            print("Downloading dataset...")
            dataset = load_dataset(
                "iwslt2017",
                "iwslt2017-de-en",
                cache_dir=os.path.abspath("./cache/dataset_cache_dir"),
            )
            dataset.save_to_disk(self.path)
        else:
            print("Dataset is already downloaded")
            dataset = load_from_disk(self.path)
        print("Dataset loaded")
        return dataset


class TranslationDatasetIWSLT2017_EN_FR(TranslationDataset):
    path = "./data/iwslt2017-en-fr"

    src_col_name = "en"
    trg_col_name = "fr"

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

    def _download_or_load_raw_dataset(self):
        if not os.path.isdir(self.path):
            print("Downloading dataset...")
            dataset = load_dataset(
                "iwslt2017",
                "iwslt2017-en-fr",
                cache_dir=os.path.abspath("./cache/dataset_cache_dir"),
            )
            dataset.save_to_disk(self.path)
        else:
            print("Dataset is already downloaded")
            dataset = load_from_disk(self.path)
        print("Dataset loaded")
        return dataset


class TranslationDatasetIWSLT2017_EN_CH(TranslationDataset):
    path = "./data/iwslt2017-en-ch"

    src_col_name = "en"
    trg_col_name = "ch"

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

    def _download_or_load_raw_dataset(self):
        if not os.path.isdir(self.path):
            print("Downloading dataset...")
            dataset = load_dataset(
                "iwslt2017",
                "iwslt2017-en-ch",
                cache_dir=os.path.abspath("./cache/dataset_cache_dir"),
            )
            dataset.save_to_disk(self.path)
        else:
            print("Dataset is already downloaded")
            dataset = load_from_disk(self.path)
        print("Dataset loaded")
        return dataset


# class TranslationDatasetOPUS_EN_FR(TranslationDataset):
#     path = './data/opus-en-fr'
#     num_labels = None

#     def __init__(self, tokenizer, max_token_count, split='train'):
#         super().__init__(tokenizer=tokenizer, max_token_count=max_token_count)
#         self.src_col_name = "en"
#         self.trg_col_name = "fr"
#         self.data = self.dataset[split]

#     def _download_and_process(self):
#         dataset = load_dataset('opus_euconst',
#                                'en-fr',
#                                cache_dir='./dataset_cache_dir')
#         dataset.save_to_disk(self.path)
#         self.dataset = dataset
#         import pdb
#         pdb.set_trace()


class TranslationDatasetWMT19_DE_EN(TranslationDataset):
    path = "./data/wmt19-de-en"

    src_col_name = "de"
    trg_col_name = "en"

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

    def _download_or_load_raw_dataset(self):
        if not os.path.isdir(self.path):
            print("Downloading dataset...")
            dataset = load_dataset(
                "iwslt2017",
                "iwslt2017-de-en",
                cache_dir=os.path.abspath("./cache/dataset_cache_dir"),
            )
            dataset.save_to_disk(self.path)
        else:
            print("Dataset is already downloaded")
            dataset = load_from_disk(self.path)
        print("Dataset loaded")
        return dataset


class TranslationDatasetWMT19_ZH_EN(TranslationDataset):
    path = "./data/wmt19-zh-en"

    src_col_name = "zh"
    trg_col_name = "en"

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

    def _download_or_load_raw_dataset(self):
        if not os.path.isdir(self.path):
            print("Downloading dataset...")
            dataset = load_dataset(
                "wmt19",
                "zh-en",
                cache_dir=os.path.abspath("./cache/dataset_cache_dir"),
            )
            dataset.save_to_disk(self.path)
        else:
            print("Dataset is already downloaded")
            dataset = load_from_disk(self.path)
        print("Dataset loaded")
        return dataset


class TranslationDatasetWMT16_RO_EN(TranslationDataset):
    path = "./data/wmt16-ro-en"

    src_col_name = "ro"
    trg_col_name = "en"

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

    def _download_or_load_raw_dataset(self):
        if not os.path.isdir(self.path):
            print("Downloading dataset...")
            dataset = load_dataset(
                "wmt16",
                "ro-en",
                cache_dir=os.path.abspath("./cache/dataset_cache_dir"),
            )
            dataset.save_to_disk(self.path)
        else:
            print("Dataset is already downloaded")
            dataset = load_from_disk(self.path)
        print("Dataset loaded")
        return dataset


class TranslationDatasetMulti30k(TranslationDataset):
    path = "./data/multi30k"

    src_col_name = "en"
    trg_col_name = "de"

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

    def __getitem__(self, index):
        data_row = self.data[index]
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

    def _download_and_process(self):
        train_dataset = multi30k_dataset(
            directory=os.path.abspath("./data/multi30k"), train=True
        )
        validation_dataset = multi30k_dataset(
            directory=os.path.abspath("./data/multi30k"), dev=True
        )
        test_dataset = multi30k_dataset(
            directory=os.path.abspath("./data/multi30k"), test=True
        )
        dataset = {
            "train": train_dataset,
            "validation": validation_dataset,
            "test": test_dataset,
        }
        return dataset

    def prepare_data(self, tokenizer, max_token_len, num_workers=None):
        self._set_tokenizer(tokenizer, max_token_len)
        self._download_and_process()

    def setup(self, tokenizer, max_token_len):
        self._set_tokenizer(tokenizer, max_token_len)
        assert os.path.isdir(self.path), f"The dataset dir {self.path} does not exist"
        self.data = self._download_and_process()[self.split]
