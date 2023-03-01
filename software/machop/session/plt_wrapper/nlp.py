import torch
import torch.nn as nn

from torchmetrics import Accuracy
from transformers import AutoModel
from .base import WrapperBase

name_to_final_module_map = {
    # TODO: double check on how to extract classifier from last_hidden_state
    'facebook/opt-350m': 'last_hidden_state',
    # BERT-ish model makes use a of a pooler
    'roberta-base': 'pooler_output',
    'roberta-large': 'pooler_output',
}


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
        self.fn = name_to_final_module_map[self.model.name_or_path]
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=self.num_classes)

        self.train_accs, self.train_losses = [], []
        self.val_accs, self.val_losses = [], []
        self.test_accs, self.test_losses = [], []


    def forward(self, input_ids, attention_mask, labels=None):
        """
        output.last_hidden_state (batch_size, token_num, hidden_size): hidden representation for each token in each sequence of the batch. 
        output.pooler_output (batch_size, hidden_size): take hidden representation of [CLS] token in each sequence, run through BertPooler module (linear layer with Tanh activation)
        """
        output = self.model(input_ids, attention_mask=attention_mask)
        hidden = getattr(output, self.fn)
        output = self.classifier(hidden)
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
        self.train_losses.append(loss)
        self.train_accs.append(acc)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, prog_bar=True, sync_dist=True)
        return {
            "loss": loss,
            "predictions": outputs,
            "labels": labels,
            "train_accuracy": acc
        }

    def on_train_epoch_end(self):
        train_mean_loss = torch.mean(
            torch.tensor(self.train_losses, dtype=torch.float32))
        train_mean_acc = torch.mean(
            torch.tensor(self.train_accs, dtype=torch.float32))
        self.train_losses = []
        self.train_accs = []
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
        self.val_losses.append(loss)
        self.val_accs.append(acc)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_accuracy", acc, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        mean_loss = torch.mean(
            torch.tensor(self.val_losses, dtype=torch.float32))
        mean_acc = torch.mean(
            torch.tensor(self.val_accs, dtype=torch.float32))
        self.val_losses = []
        self.val_accs = []
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
        self.test_losses.append(loss)
        self.test_accs.append(acc)
        return loss

    def on_test_epoch_end(self):
        mean_loss = torch.mean(
            torch.tensor(self.test_losses, dtype=torch.float32))
        mean_acc = torch.mean(
            torch.tensor(self.test_accs, dtype=torch.float32))
        self.test_losses = []
        self.test_accs = []
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
