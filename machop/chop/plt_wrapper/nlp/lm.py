import torch
import torch.nn as nn
from torchmetrics import MeanMetric

from ..base import WrapperBase


class NLPLanguageModelingModelWrapper(WrapperBase):
    def __init__(self, model, info, learning_rate=1e-4, epochs=100, optimizer=None):
        super().__init__(
            model=model,
            info=info,
            learning_rate=learning_rate,
            epochs=epochs,
            optimizer=optimizer,
        )
        self.model = model["model"]
        self.tokenizer = model["tokenizer"]

        # self.loss_mean_train = MeanMetric()
        self.loss_mean_val = MeanMetric()
        self.loss_mean_test = MeanMetric()

    def forward(self, input_ids, attention_mask, labels):
        """
        output.last_hidden_state (batch_size, token_num, hidden_size): hidden representation for each token in each sequence of the batch.
        """
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        # loss = self.criterion(output, labels)
        loss = output["loss"]
        output = output["logits"]
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)

        perplexity = torch.exp(loss)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log(
            "train_perplexity",
            perplexity,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return {"loss": loss, "predictions": outputs, "perplexity": perplexity}

    # def on_train_epoch_end(self):
    #     mean_loss = self.loss_mean_train.compute()
    #     mean_perplexity = torch.exp(mean_loss)
    #     self.log(
    #         "train_perplexity_epoch",
    #         mean_perplexity,
    #         on_step=False,
    #         on_epoch=True,
    #         prog_bar=True,
    #         sync_dist=True,
    #     )
    #     self.loss_mean_train.reset()

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, _ = self.forward(input_ids, attention_mask, labels)

        # perplexity = torch.exp(loss)
        self.loss_mean_val.update(loss)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        return loss

    def on_validation_epoch_end(self):
        mean_loss = self.loss_mean_val.compute()
        mean_perplexity = torch.exp(mean_loss)

        self.log(
            "val_perplexity",
            mean_perplexity,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.loss_mean_val.reset()

        return {"val_mean_loss": mean_loss, "val_mean_perplexity": mean_perplexity}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        self.loss_mean_test.update(loss)

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def on_test_epoch_end(self):
        mean_loss = self.loss_mean_test.compute()
        mean_perplexity = torch.exp(mean_loss)

        self.log(
            "test_perplexity",
            mean_perplexity,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.loss_mean_test.reset()

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        _, outputs = self.forward(input_ids, attention_mask, labels=None)
        outputs["batch_idx"] = batch_idx
        return outputs
