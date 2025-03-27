import torch
import torch.nn as nn

from ..base import WrapperBase
from .losses import v8DetectionLoss


class UltralyticsDetectionWrapper(WrapperBase):
    def __init__(
        self,
        model,
        dataset_info,
        learning_rate=1e-4,
        weight_decay=0,
        scheduler_args=None,
        epochs=3,
        criterion=v8DetectionLoss,
        optimizer=None,
    ):
        super().__init__(
            model=model,
            dataset_info=dataset_info,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            scheduler_args=scheduler_args,
            epochs=epochs,
            optimizer=optimizer,
        )
        self.model = model

        # Need to pass model.args to the model
        self.criterion = criterion(
            model=self.model,
            box=7.5,
            cls=0.5,
            dfl=1.5,
        )  # Overrides the WrapperBase loss_fn (BCE)
        # Uses base hyperparameters from YOLO Info

    def training_step(self, batch, batch_idx):
        inputs = batch["img"]
        outputs = self.forward(inputs)

        loss, _ = self.criterion(outputs, batch)
        self.log("loss", loss.detach().item())
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["img"]
        outputs = self.forward(inputs)
        loss, _ = self.criterion(outputs, batch)
        self.log("val_loss", loss.detach().item())
        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch["img"]
        outputs = self.forward(inputs)

        loss, _ = self.criterion(outputs, batch)
        self.log("test_loss", loss.detach().item())
        return loss

    def on_test_epoch_end(self):
        self.log("test_bleu_epoch", self.bleu_test, prog_bar=True)
        self.log("test_loss_epoch", self.loss_test, prog_bar=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self.forward(
            input_ids,
            attention_mask,
            decoder_input_ids=None,
            decoder_attention_mask=None,
        )
        _, pred_ids = torch.max(outputs["logits"], dim=1)
        outputs["batch_idx"] = batch_idx
        outputs["pred_ids"] = pred_ids
        return outputs
