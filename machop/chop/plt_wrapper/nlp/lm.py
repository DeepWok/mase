import torch
import torch.nn as nn
from torchmetrics import MeanMetric

from ..base import WrapperBase


class NLPLanguageModelingModelWrapper(WrapperBase):
    def __init__(
        self,
        model,
        dataset_info,
        learning_rate=1e-4,
        weight_decay=0,
        epochs=100,
        optimizer=None,
    ):
        super().__init__(
            model=model,
            dataset_info=dataset_info,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            optimizer=optimizer,
        )

    def forward(self, input_ids, attention_mask, labels):
        """
        output.last_hidden_state (batch_size, token_num, hidden_size): hidden representation for each token in each sequence of the batch.
        """
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self.forward(input_ids, attention_mask, labels)
        loss = outputs["loss"]

        perplexity = torch.exp(loss)

        self.log("train_loss_step", loss, prog_bar=True)
        self.log(
            "train_perplexity_step",
            perplexity,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self.forward(input_ids, attention_mask, labels)
        loss = outputs["loss"]

        self.loss_val.update(loss)

        return loss

    def on_validation_epoch_end(self):
        loss_epoch = self.loss_val.compute()
        perplexity_epoch = torch.exp(loss_epoch)
        self.log("val_loss_epoch", loss_epoch, prog_bar=True)
        self.log("val_perplexity_epoch", perplexity_epoch, prog_bar=True)

        self.loss_val.reset()

        return loss_epoch

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self.forward(input_ids, attention_mask, labels)
        loss = outputs["loss"]
        self.loss_test.update(loss)

        return loss

    def on_test_epoch_end(self):
        loss_epoch = self.loss_test.compute()
        perplexity_epoch = torch.exp(loss_epoch)

        self.log("test_loss_epoch", loss_epoch, prog_bar=True)
        self.log("test_perplexity_epoch", perplexity_epoch, prog_bar=True)
        self.loss_test.reset()

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self.forward(input_ids, attention_mask, labels=None)
        outputs["batch_idx"] = batch_idx
        return outputs
