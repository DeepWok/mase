import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .nlp import dataset_factory as nlp_dataset_factory
from .utils import get_dataset


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        name: str,
        batch_size: int,
        workers: int,
        tokenizer,
        max_token_len,
    ):
        super().__init__()
        self.dataset_name = name
        self.batch_size = batch_size
        self.num_workers = workers
        self.tokenizer = tokenizer
        if max_token_len is None:
            self.max_token_len = tokenizer.model_max_length
        else:
            self.max_token_len = max_token_len
        # breakpoint()

    def prepare_data(self) -> None:
        train_dataset, val_dataset, test_dataset = get_dataset(self.dataset_name)
        # breakpoint()
        if self.dataset_name in nlp_dataset_factory:
            train_dataset.prepare_data(
                self.tokenizer, self.max_token_len, self.num_workers
            )
            test_dataset.prepare_data(
                self.tokenizer, self.max_token_len, self.num_workers
            )
            val_dataset.prepare_data(
                self.tokenizer, self.max_token_len, self.num_workers
            )

    def setup(self, stage: str = None) -> None:
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = get_dataset(self.dataset_name)

        if self.dataset_name in nlp_dataset_factory:
            self.train_dataset.setup(self.tokenizer, self.max_token_len)
            self.val_dataset.setup(self.tokenizer, self.max_token_len)
            self.test_dataset.setup(self.tokenizer, self.max_token_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
