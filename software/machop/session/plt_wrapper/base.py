import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

# from torchmetrics.functional import accuracy
from torchmetrics import Accuracy, MeanMetric


class WrapperBase(pl.LightningModule):
    def __init__(
        self, model, learning_rate=5e-4, epochs=200, optimizer=None, info=None
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss = torch.nn.CrossEntropyLoss()
        self.epochs = epochs
        self.optimizer = optimizer

        self.num_classes = info["num_classes"]
        if self.num_classes is not None:
            self.acc_train = Accuracy("multiclass", num_classes=info["num_classes"])
            self.acc_val = Accuracy("multiclass", num_classes=info["num_classes"])
            self.acc_test = Accuracy("multiclass", num_classes=info["num_classes"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        acc = self.acc_train(y_hat, y)

        self.log(
            "train_acc", self.acc_train, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        acc = self.acc_val(y_hat, y)

        self.log("val_acc", self.acc_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        acc = self.acc_test(y_hat, y)

        self.log("test_acc", self.acc_test, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
            scheduler = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-6)
        elif self.optimizer == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            scheduler = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-6)
        elif self.optimizer in ["sgd_warmup", "sgd"]:
            opt = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=0.0005,
                nesterov=True,
            )
            if self.optimizer == "sgd":
                scheduler = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=0.0)
        return {"optimizer": opt, "lr_scheduler": scheduler}
