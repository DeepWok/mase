import datasets as hf_datasets
from torch.utils.data import Dataset
from ..utils import add_dataset_info

@add_dataset_info(
    name="librispeech_asr",
    dataset_source="hf_datasets",
    available_splits=("train", "validation", "test"),
    sequence_classification=True,  # Adjust if necessary
)
class LibrispeechASRDataset(Dataset):
    def __init__(self, split: str, tokenizer, max_token_len: int, num_workers: int, load_from_cache_file: bool = True, auto_setup: bool = True):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.num_workers = num_workers
        self.load_from_cache_file = load_from_cache_file
        self.data = None

        if auto_setup:
            self.prepare_data()
            self.setup()

    def _download_dataset(self) -> hf_datasets.DatasetDict:
        dataset_dict = hf_datasets.load_dataset("librispeech_asr", split=self.split)
        return dataset_dict

    def prepare_data(self):
        self._download_dataset()

    def setup(self):
        self.data = self._download_dataset()

    def __len__(self):
        if self.data is None:
            raise ValueError("Dataset is not setup. Please call `dataset.prepare_data()` + `dataset.setup()` or pass `auto_setup=True` before using the dataset.")
        return len(self.data)

    def __getitem__(self, index):
        if self.data is None:
            raise ValueError("Dataset is not setup. Please call `dataset.prepare_data()` + `dataset.setup()` or pass `auto_setup=True` before using the dataset.")
        data_row = self.data[index]
        # Implement data processing logic here
        return data_row