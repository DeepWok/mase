import re
import math
import string
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset
from datasets import load_dataset


class SentAnalDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.sent_col_name = None
        self.label_col_name = None
    
    def setup_tokenizer(self, tokenizer, max_token_count):
        self.tokenizer = tokenizer
        self.max_token_count = max_token_count

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_row = self.data[index]
        main_text = data_row[self.sent_col_name]
        labels = data_row[self.label_col_name]
        encoding = self.tokenizer.encode_plus(
            main_text,
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
            sentence=main_text,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=torch.tensor([labels])
        )


class SentAnalDatasetSST2(SentAnalDataset):
    path = './data/sst2'
    num_labels = 2
    def __init__(self, data):
        super().__init__(data=data)
        self.sent_col_name = "sentence"
        self.label_col_name = "label"
    
    def _download_and_process(self):
        dataset = load_dataset('glue', 'sst2')
        dataset.save_to_disk(self.path)
        self.dataset = dataset