import os
import torch

from torchnlp.datasets import multi30k_dataset

from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk


class TranslationDataset(Dataset):
    path = None
    num_classes = None
    split = None

    def __init__(self):
        self.src_col_name = None
        self.trg_col_name = None

        if (self.path is not None) and (not os.path.isdir(self.path)):
            print("Downloading and processing dataset...")
            self._download_and_process()
        else:
            print("Dataset already downloaded and processed")
            self._load_from_path()

    def setup_tokenizer(self, tokenizer, max_token_count):
        self.tokenizer = tokenizer
        self.max_token_count = max_token_count
        self.dataset = self.dataset[self.split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_row = self.dataset[index]['translation']
        source = data_row[self.src_col_name]
        target = data_row[self.trg_col_name]
        encoding = self.tokenizer.encode_plus(source,
                                              add_special_tokens=True,
                                              max_length=self.max_token_count,
                                              padding="max_length",
                                              truncation=True,
                                              return_attention_mask=True,
                                              return_tensors="pt")
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()
        label_encoding = self.tokenizer.encode_plus(
            source,
            add_special_tokens=True,
            max_length=self.max_token_count,
            padding="max_length",
            truncation=True,
            return_attention_mask=True, 
            return_tensors="pt")
        decoder_input_ids = label_encoding["input_ids"].flatten()
        decoder_attention_mask = label_encoding["attention_mask"].flatten()
        return dict(source=source,
                    target=target,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask)

    def _download_and_process(self):
        raise NotImplementedError

    def _load_from_path(self):
        self.dataset = load_from_disk(self.path)


# It seems like the download link of this is broken ..
class TranslationDatasetIWSLT2017_EN_DE(TranslationDataset):
    path = './data/iwslt2017-en-de'

    def __init__(self, split='train'):
        super().__init__()
        self.src_col_name = "en"
        self.trg_col_name = "de"
        # self.data = self.dataset[split]
        self.split = split

    def _download_and_process(self):
        dataset = load_dataset('iwslt2017',
                               'iwslt2017-en-de',
                               cache_dir='./datasets_cache_dir')
        dataset.save_to_disk(self.path)
        self.dataset = dataset


class TranslationDatasetIWSLT2017_DE_EN(TranslationDataset):
    path = './data/iwslt2017-de-en'

    def __init__(self, split='train'):
        super().__init__()
        self.src_col_name = "de"
        self.trg_col_name = "en"
        # self.data = self.dataset[split]
        self.split = split

    def _download_and_process(self):
        dataset = load_dataset('iwslt2017',
                               'iwslt2017-de-en',
                               cache_dir='./datasets_cache_dir')
        dataset.save_to_disk(self.path)
        self.dataset = dataset


class TranslationDatasetIWSLT2017_EN_FR(TranslationDataset):
    path = './data/iwslt2017-en-fr'

    def __init__(self, split='train'):
        super().__init__()
        self.src_col_name = "en"
        self.trg_col_name = "fr"
        # self.data = self.dataset[split]
        self.split = split

    def _download_and_process(self):
        dataset = load_dataset('iwslt2017',
                               'iwslt2017-en-fr',
                               cache_dir='./datasets_cache_dir')
        dataset.save_to_disk(self.path)
        self.dataset = dataset


class TranslationDatasetIWSLT2017_EN_CH(TranslationDataset):
    path = './data/iwslt2017-en-ch'

    def __init__(self, split='train'):
        super().__init__()
        self.src_col_name = "en"
        self.trg_col_name = "ch"
        # self.data = self.dataset[split]
        self.split = split

    def _download_and_process(self):
        dataset = load_dataset('iwslt2017',
                               'iwslt2017-en-ch',
                               cache_dir='./datasets_cache_dir')
        dataset.save_to_disk(self.path)
        self.dataset = dataset

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
#                                cache_dir='./datasets_cache_dir')
#         dataset.save_to_disk(self.path)
#         self.dataset = dataset
#         import pdb
#         pdb.set_trace()


class TranslationDatasetWMT19_DE_EN(TranslationDataset):
    path = './data/wmt19-de-en'

    def __init__(self, split='train'):
        super().__init__()
        self.src_col_name = "de"
        self.trg_col_name = "en"
        # self.data = self.dataset[split]
        self.split = split

    def _download_and_process(self):
        dataset = load_dataset('wmt19',
                               'de-en',
                               cache_dir='./datasets_cache_dir')
        dataset.save_to_disk(self.path)
        self.dataset = dataset

class TranslationDatasetWMT19_ZH_EN(TranslationDataset):
    path = './data/wmt19-zh-en'

    def __init__(self, split='train'):
        super().__init__()
        self.src_col_name = "zh"
        self.trg_col_name = "en"
        # self.data = self.dataset[split]
        self.split = split

    def _download_and_process(self):
        dataset = load_dataset('wmt19',
                               'zh-en',
                               cache_dir='./datasets_cache_dir')
        dataset.save_to_disk(self.path)
        self.dataset = dataset

class TranslationDatasetWMT16_RO_EN(TranslationDataset):
    path = './data/wmt16-ro-en'
    num_labels = None

    def __init__(self, tokenizer, max_token_count, split='train'):
        super().__init__(tokenizer=tokenizer, max_token_count=max_token_count)
        self.src_col_name = "ro"
        self.trg_col_name = "en"
        self.data = self.dataset[split]

    def _download_and_process(self):
        dataset = load_dataset('wmt16',
                               'ro-en',
                               cache_dir='./datasets_cache_dir')
        dataset.save_to_disk(self.path)
        self.dataset = dataset

    def __getitem__(self, index):
        data_row = self.data[index]['translation']
        source = data_row[self.src_col_name]
        target = data_row[self.trg_col_name]
        encoding = self.tokenizer.encode_plus(source,
                                              add_special_tokens=True,
                                              max_length=self.max_token_count,
                                              padding="max_length",
                                              truncation=True,
                                              return_attention_mask=True,
                                              return_tensors="pt")
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()
        label_encoding = self.tokenizer.encode_plus(
            source,
            add_special_tokens=True,
            max_length=self.max_token_count,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt")
        labels = label_encoding["input_ids"].flatten()
        return dict(source=source,
                    target=target,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels)


class TranslationDatasetMulti30k(TranslationDataset):
    path = './data/multi30k'

    def __init__(self, tokenizer, max_token_count, split='train'):
        super().__init__(tokenizer, max_token_count)
        self.src_col_name = "en"
        self.trg_col_name = "de"
        if split == 'train':
            self.data = multi30k_dataset(train=True)
        elif split in ['val', 'eval', 'validation']:
            self.data = multi30k_dataset(dev=True)
        elif split in ['test']:
            self.data = multi30k_dataset(test=True)
        else:
            raise ValueError(f"split {split} not supported")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data[index]
        source = data_row[self.src_col_name]
        target = data_row[self.trg_col_name]
        encoding = self.tokenizer.encode_plus(source,
                                              add_special_tokens=True,
                                              max_length=self.max_token_count,
                                              padding="max_length",
                                              truncation=True,
                                              return_attention_mask=True,
                                              return_tensors="pt")
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()
        label_encoding = self.tokenizer.encode_plus(
            source,
            add_special_tokens=True,
            max_length=self.max_token_count,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt")
        labels = label_encoding["input_ids"].flatten()
        return dict(source=source,
                    target=target,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels)

    def _download_and_process(self):
        pass

    def _load_from_path(self):
        pass
