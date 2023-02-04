import re
import math
import string
import os
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk


class TextEntailDataset(Dataset):
    path = None
    def __init__(self):
        self.sent1_col_name = None
        self.sent2_col_name = None
        self.label_col_name = None

        if (self.path is not None) and (not os.path.isdir(self.path)):
            print("Downloading and processing dataset...")
            self._download_and_process()
        else:
            print("Dataset already downloaded and processed")
            self._load_from_path()

    def setup_tokenizer(self, tokenizer, max_token_count):
        self.tokenizer = tokenizer
        self.max_token_count = max_token_count

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_row = self.data[index]
        question = data_row[self.sent1_col_name]
        answer = data_row[self.sent2_col_name]
        labels = data_row[self.label_col_name]
        encoding = self.tokenizer.encode_plus(
            question,
            answer,
            add_special_tokens=True,
            max_length=self.max_token_count,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        input_ids=encoding["input_ids"].flatten()
        attention_mask=encoding["attention_mask"].flatten()
        
        return dict(
            question=question,
            answer=answer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=torch.tensor([labels]))
        
    def _download_and_process(self):
        raise NotImplementedError
    
    def _load_from_path(self):
        self.dataset = load_from_disk(self.path)


class TextEntailDatasetQNLI(TextEntailDataset):
    path = './data/qnli'
    num_labels = 2
    def __init__(self, split='train'):
        super().__init__()
        self.sent1_col_name = "question"
        self.sent2_col_name = "sentence"
        self.label_col_name = "label"
        self.data = self.dataset[split]

    def _download_and_process(self):
        dataset = load_dataset('glue', 'qnli')
        dataset.save_to_disk(self.path)
        self.dataset = dataset


class TextEntailDatasetMNLI(TextEntailDataset):
    path = './data/mnli'
    num_classes = 3
    def __init__(self, split='train'):
        super().__init__()
        self.sent1_col_name = "premise"
        self.sent2_col_name = "hypothesis"
        self.label_col_name = "label"
        self.data = self.dataset[split]

    def _download_and_process(self):
        dataset = load_dataset('glue', 'mnli')
        dataset.save_to_disk(self.path)
        self.dataset = dataset