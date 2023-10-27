import pytorch_lightning as pl
import torch

# from deepspeed.ops.adam import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR

# from torchmetrics.functional import accuracy
from torchmetrics import Accuracy, MeanMetric


class WrapperBase(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate=5e-4,
        weight_decay=0.0,
        epochs=1,
        optimizer=None,
        dataset_info=None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.epochs = epochs
        self.optimizer = optimizer

        self.num_classes = dataset_info.num_classes
        if self.num_classes is not None:
            self.acc_train = Accuracy("multiclass", num_classes=self.num_classes)
            self.acc_val = Accuracy("multiclass", num_classes=self.num_classes)
            self.acc_test = Accuracy("multiclass", num_classes=self.num_classes)

        self.loss_val = MeanMetric()
        self.loss_test = MeanMetric()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        self.acc_train(y_hat, y)
        self.log("train_acc_step", self.acc_train, prog_bar=True)
        self.log("train_loss_step", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        self.acc_val(y_hat, y)
        self.loss_val(loss)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_acc_epoch", self.acc_val, prog_bar=True)
        self.log("val_loss_epoch", self.loss_val, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        self.acc_test(y_hat, y)
        self.loss_test(loss)

        return loss

    def on_test_epoch_end(self):
        self.log("test_acc_epoch", self.acc_test, prog_bar=True)
        self.log("test_loss_epoch", self.loss_test, prog_bar=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch[0], batch[1]
        pred_y = self.forward(x)
        return {"batch_idx": batch_idx, "pred_y": pred_y}

    def configure_optimizers(self):
        # Use self.trainer.model.parameters() instead of self.parameters() to support FullyShared (Model paralleled) training
        if self.optimizer == "adamw":
            opt = torch.optim.AdamW(
                self.trainer.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            scheduler = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-6)
        elif self.optimizer == "adam":
            opt = torch.optim.Adam(
                self.trainer.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            scheduler = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-6)
        elif self.optimizer in ["sgd_warmup", "sgd"]:
            opt = torch.optim.SGD(
                self.trainer.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=0.0005,
                nesterov=True,
            )
            if self.optimizer == "sgd":
                scheduler = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=0.0)
        # elif self.optimizer in ["fused_adam", "FusedAdam"]:
        #     # DeepSpeed strategy="deepspeed_stage_3"
        #     opt = FusedAdam(self.parameters(), lr=self.learning_rate)
        #     scheduler = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-6)
        else:
            raise ValueError(f"Unsupported optimizer name {self.optimizer}")
        return {"optimizer": opt, "lr_scheduler": scheduler}
