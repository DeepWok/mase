import torch
import torch.nn as nn
from torchmetrics import Accuracy, MeanMetric

# from ....models.patched_nlp_models import patched_model_name_to_output_name
from ..base import WrapperBase


class NLPClassificationModelWrapper(WrapperBase):
    def __init__(
        self,
        model,
        tokenizer,
        dataset_info,
        learning_rate=1e-4,
        epochs=200,
        optimizer=None,
    ):
        super().__init__(
            model=model,
            dataset_info=dataset_info,
            learning_rate=learning_rate,
            epochs=epochs,
            optimizer=optimizer,
        )
        self.model = model
        self.acc_train = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.acc_val = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.acc_test = Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """
        output.last_hidden_state (batch_size, token_num, hidden_size): hidden representation for each token in each sequence of the batch.
        output.pooler_output (batch_size, hidden_size): take hidden representation of [CLS] token in each sequence, run through BertPooler module (linear layer with Tanh activation)
        """
        if token_type_ids is not None:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
        else:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        return outputs

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        token_type_ids = batch.get("token_type_ids", None)

        outputs = self.forward(input_ids, attention_mask, token_type_ids, labels)
        loss = outputs["loss"]
        logits = outputs["logits"]
        _, pred_ids = torch.max(logits, dim=1)
        labels = labels[0] if len(labels) == 1 else labels.squeeze()

        acc = self.acc_train(pred_ids, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log(
            "train_acc", self.acc_train, on_step=True, on_epoch=False, prog_bar=True
        )

        return {
            "loss": loss,
            "predictions": pred_ids,
            "labels": labels,
            "train_accuracy": acc,
        }

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        token_type_ids = batch.get("token_type_ids", None)
        outputs = self.forward(input_ids, attention_mask, token_type_ids, labels)
        logits = outputs["logits"]
        loss = outputs["loss"]
        _, pred_ids = torch.max(logits, dim=1)
        labels = labels[0] if len(labels) == 1 else labels.squeeze()

        self.acc_val(pred_ids, labels)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "val_acc",
            self.acc_val,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        token_type_ids = batch.get("token_type_ids", None)
        outputs = self.forward(input_ids, attention_mask, token_type_ids, labels)
        loss = outputs["loss"]
        logits = outputs["logits"]
        _, pred_ids = torch.max(logits, dim=1)
        labels = labels[0] if len(labels) == 1 else labels.squeeze()

        self.acc_test(pred_ids, labels)

        self.log(
            "test_acc",
            self.acc_test,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self.forward(input_ids, attention_mask, labels=None)
        logits = outputs["logits"]
        pred_ids = torch.max(logits, dim=1)
        return {"batch_idx": batch_idx, "outputs": outputs, "pred_ids": pred_ids}
