import torch
import torch.nn as nn

from ..base import WrapperBase
from ultralytics.utils.loss import v8DetectionLoss


class UltralyticsDetectionWrapper(WrapperBase):
    def __init__(
        self,
        model,
        dataset_info,
        learning_rate=1e-4,
        weight_decay=0,
        scheduler_args=None,
        epochs=200,
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

    def training_step(self, batch, batch_idx):
        inputs = batch["img"]
        outputs = self.forward(
                inputs
        )
        
        loss = self.criterion(outputs, batch)
        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["img"]
        outputs = self.forward(
                inputs
        )

        loss = self.criterion(outputs, batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch["img"]
        outputs = self.forward(
                inputs
        )

        loss = self.criterion(outputs, batch)
        self.log("test_loss", loss)

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
