import logging

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .nlp import dataset_factory as nlp_dataset_factory
from .utils import get_dataset

logger = logging.getLogger(__name__)


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        batch_size: int,
        workers: int,
        tokenizer,
        max_token_len,
    ):
        super().__init__()
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = workers
        self.tokenizer = tokenizer
        if max_token_len is None and tokenizer is not None:
            self.max_token_len = tokenizer.model_max_length
        elif max_token_len is not None:
            self.max_token_len = max_token_len

        self.train_dataset = (
            self.val_dataset
        ) = self.test_dataset = self.pred_dataset = None

    def prepare_data(self) -> None:
        train_dataset, val_dataset, test_dataset, pred_dataset = get_dataset(
            model_name=self.model_name, dataset_name=self.dataset_name
        )

        if self.dataset_name in nlp_dataset_factory:
            train_dataset.prepare_data(
                self.tokenizer, self.max_token_len, self.num_workers
            )
            val_dataset.prepare_data(
                self.tokenizer, self.max_token_len, self.num_workers
            )
            if test_dataset is not None:
                test_dataset.prepare_data(
                    self.tokenizer, self.max_token_len, self.num_workers
                )
            if pred_dataset is not None:
                pred_dataset.prepare_data(
                    self.tokenizer, self.max_token_len, self.num_workers
                )

    def setup(self, stage: str = None) -> None:
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            self.pred_dataset,
        ) = get_dataset(model_name=self.model_name, dataset_name=self.dataset_name)

        if self.dataset_name in nlp_dataset_factory:
            self.train_dataset.setup(self.tokenizer, self.max_token_len)
            self.val_dataset.setup(self.tokenizer, self.max_token_len)
            if self.test_dataset is not None:
                self.test_dataset.setup(self.tokenizer, self.max_token_len)
                logger.debug("self.test_dataset is not None")
            if self.pred_dataset is not None:
                self.pred_dataset.setup(self.tokenizer, self.max_token_len)
                logger.debug("self.pred_dataset is not None")

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
        if self.test_dataset is None:
            raise RuntimeError(
                "The test dataset is not available"
                "probably because the test set does not have ground truth labels, "
                "or does not exist. For the former case, try predict_dataloader"
            )

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        if self.pred_dataset is None:
            raise RuntimeError("The pred dataset is not available.")
        return DataLoader(
            self.pred_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
