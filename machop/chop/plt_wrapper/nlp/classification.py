import torch
import torch.nn as nn
from torchmetrics import Accuracy, MeanMetric

# from ....models.patched_nlp_models import patched_model_name_to_output_name
from ..base import WrapperBase


class NLPClassificationModelWrapper(WrapperBase):
    def __init__(
        self,
        model,
        dataset_info,
        learning_rate=1e-4,
        weight_decay=0.0,
        epochs=200,
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

        self.acc_train(pred_ids, labels)

        self.log("train_loss_step", loss, prog_bar=True)
        self.log("train_acc_step", self.acc_train, prog_bar=True)

        return loss

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
        self.loss_val(loss)

        return loss

    def on_validation_epoch_end(self):
        self.log("val_acc_epoch", self.acc_val, prog_bar=True)
        self.log("val_loss_epoch", self.loss_val, prog_bar=True)

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

        return loss

    def on_test_epoch_end(self):
        self.log("test_acc_epoch", self.acc_test, prog_bar=True)
        self.log("test_loss_epoch", self.loss_test, prog_bar=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self.forward(input_ids, attention_mask, labels=None)
        logits = outputs["logits"]
        pred_ids = torch.max(logits, dim=1)
        return {"batch_idx": batch_idx, "outputs": outputs, "pred_ids": pred_ids}
