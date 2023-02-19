import torch
import torch.nn as nn
from transformers import AutoModel
from .base import WrapperBase


class NLPClassificationModelWrapper(WrapperBase):

    def __init__(self, model, info, learning_rate, epochs=200, optimizer=None):
        super().__init__(model=model,
                         info=info,
                         learning_rate=learning_rate,
                         epochs=epochs,
                         optimizer=optimizer)
        self.model = model['model']
        self.tokenizer = model['tokenizer']
        self.classifier = model['classifier']

    def forward(self, input_ids, attention_mask, labels=None):
        """
        output.last_hidden_state (batch_size, token_num, hidden_size): hidden representation for each token in each sequence of the batch. 
        output.pooler_output (batch_size, hidden_size): take hidden representation of [CLS] token in each sequence, run through BertPooler module (linear layer with Tanh activation)
        """
        output = self.model(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0

        if labels is not None:
            labels = labels.squeeze()
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        _, pred_ids = torch.max(outputs, dim=1)
        labels = labels[0] if len(labels) == 1 else labels.squeeze()
        acc = self.accuracy(pred_ids, labels)
        self.train_loss_arr.append(loss)
        self.train_acc_arr.append(acc)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_accuracy", acc, prog_bar=True, sync_dist=True)
        return {
            "loss": loss,
            "predictions": outputs,
            "labels": labels,
            "train_accuracy": acc
        }

    def on_train_epoch_end(self):
        train_mean_loss = torch.mean(
            torch.tensor(self.train_loss_arr, dtype=torch.float32))
        train_mean_acc = torch.mean(
            torch.tensor(self.train_acc_arr, dtype=torch.float32))
        self.train_loss_arr = []
        self.train_acc_arr = []
        self.log("train_mean_loss_per_epoch",
                 train_mean_loss,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True)
        self.log("train_mean_acc_per_epoch",
                 train_mean_acc,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True)
        return {
            "train_mean_loss": train_mean_loss,
            "train_mean_acc": train_mean_acc
        }

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        _, pred_ids = torch.max(outputs, dim=1)
        labels = labels[0] if len(labels) == 1 else labels.squeeze()
        acc = self.accuracy(pred_ids, labels)
        self.val_loss_arr.append(loss)
        self.val_acc_arr.append(acc)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_accuracy", acc, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        mean_loss = torch.mean(
            torch.tensor(self.val_loss_arr, dtype=torch.float32))
        mean_acc = torch.mean(
            torch.tensor(self.val_acc_arr, dtype=torch.float32))
        self.val_loss_arr = []
        self.val_acc_arr = []
        self.log("val_mean_loss_per_epoch",
                 mean_loss,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True)
        self.log("val_mean_acc_per_epoch",
                 mean_acc,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True)
        return {"val_mean_loss": mean_loss, "val_mean_acc": mean_acc}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        _, pred_ids = torch.max(outputs, dim=1)
        labels = labels[0] if len(labels) == 1 else labels.squeeze()
        acc = self.accuracy(pred_ids, labels)
        self.test_loss_arr.append(loss)
        self.test_acc_arr.append(acc)
        return loss

    def on_test_epoch_end(self):
        mean_loss = torch.mean(
            torch.tensor(self.test_loss_arr, dtype=torch.float32))
        mean_acc = torch.mean(
            torch.tensor(self.test_acc_arr, dtype=torch.float32))
        self.test_loss_arr = []
        self.test_acc_arr = []
        self.log("test_mean_loss",
                 mean_loss,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True)
        self.log("test_mean_acc",
                 mean_acc,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True)
        return {"test_mean_loss": mean_loss, "test_mean_acc": mean_acc}
