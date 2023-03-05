import pytorch_lightning as pl

from torch.utils.data import DataLoader


class MyDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name,
        workers,
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.train_dataset, self.val_dataset, self.test_dataset = (
            train_dataset,
            val_dataset,
            test_dataset,
        )
        self.batch_size = batch_size
        self.num_workers = workers

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
