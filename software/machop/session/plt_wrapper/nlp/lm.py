import torch
import torch.nn as nn

from ..base import WrapperBase


class NLPLanguageModelingModelWrapper(WrapperBase):
    def __init__(self, model, info, learning_rate, epochs=100, optimizer=None):
        super().__init__(
            model=model,
            info=info,
            learning_rate=learning_rate,
            epochs=epochs,
            optimizer=optimizer,
        )
        self.model = model["model"]
        self.tokenizer = model["tokenizer"]
        self.criterion = nn.CrossEntropyLoss()

        self.train_losses, self.train_perplexities = [], []
        self.val_losses, self.val_perplexities = [], []
        self.test_losses, self.test_perplexities = [], []

    def forward(self, input_ids, attention_mask, labels):
        """
        output.last_hidden_state (batch_size, token_num, hidden_size): hidden representation for each token in each sequence of the batch.
        """
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        # loss = self.criterion(output, labels)
        # breakpoint()
        loss = output["loss"]
        output = output["logits"]
        return loss, output

    def training_step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)

        self.train_losses.append(loss)
        perplexity = torch.exp(loss)
        self.train_perplexities.append(perplexity)
        self.log("train_loss", loss, prog_bar=True)
        # self.log("train_perp", perplexity, prog_bar=True, sync_dist=True)
        return {"loss": loss, "predictions": outputs, "perplexity": perplexity}

    def on_train_epoch_end(self):
        train_mean_loss = torch.mean(
            torch.tensor(self.train_losses, dtype=torch.float32)
        )
        train_mean_perp = torch.exp(train_mean_loss)
        # torch.mean(torch.tensor(self.train_perplexities, dtype=torch.float32))
        self.train_losses = []
        self.train_perplexities = []
        self.log(
            "train_mean_loss",
            train_mean_loss,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train_mean_perp",
            train_mean_perp,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return {"train_mean_loss": train_mean_loss, "train_mean_acc": train_mean_perp}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        self.val_losses.append(loss)
        perplexity = torch.exp(loss)
        self.val_perplexities.append(perplexity)
        self.log("val_loss", loss, prog_bar=True)
        # self.log("val_perp", perplexity, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        mean_loss = torch.mean(torch.tensor(self.val_losses, dtype=torch.float32))
        mean_perplexity = torch.exp(mean_loss)
        # torch.mean(torch.tensor(self.val_perplexities, dtype=torch.float32))
        self.val_losses = []
        self.val_perplexities = []
        self.log("val_mean_loss", mean_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log(
            "val_mean_perp", mean_perplexity, prog_bar=True, logger=True, sync_dist=True
        )
        return {"val_mean_loss": mean_loss, "val_mean_perplexity": mean_perplexity}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        perplexity = torch.exp(loss)
        self.test_losses.append(loss)
        self.test_perplexities.append(perplexity)
        return loss

    def on_test_epoch_end(self):
        mean_loss = torch.mean(torch.tensor(self.test_losses, dtype=torch.float32))
        mean_perplexity = torch.exp(mean_loss)
        # mean_perplexity = torch.mean(
        #     torch.tensor(self.test_perplexities, dtype=torch.float32))
        self.test_losses = []
        self.test_perplexities = []
        self.log(
            "test_mean_loss", mean_loss, prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            "test_mean_perp",
            mean_perplexity,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return {"test_mean_loss": mean_loss, "test_mean_acc": mean_perplexity}
