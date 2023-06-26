import torch
import torch.nn as nn
from torchmetrics import Accuracy, MeanMetric

# from ....models.patched_nlp_models import patched_model_name_to_output_name
from ..base import WrapperBase

# This map is used to fetch the output of the last layer.
# Some model has pooler so the pooler output can be used directly (like Bert)
# Other may not have pooler, thus the output hidden states will be fetched and fed to a pooler+linear classifier (like OPT)
name_to_final_module_map = {
    # TODO: double check on how to extract classifier from last_hidden_state
    "facebook/opt-125m": "last_hidden_state",
    "facebook/opt-350m": "last_hidden_state",
    "facebook/opt-1.3b": "last_hidden_state",
    "facebook/opt-2.7b": "last_hidden_state",
    "facebook/opt-6.7b": "last_hidden_state",
    "facebook/opt-13b": "last_hidden_state",
    "facebook/opt-30b": "last_hidden_state",
    "facebook/opt-66b": "last_hidden_state",
    # BERT-ish model makes use a of a pooler
    "bert-base-cased": "pooler_output",
    "bert-base-uncased": "pooler_output",
    "roberta-base": "pooler_output",
    "roberta-large": "pooler_output",
}
# } | patched_model_name_to_output_name


class NLPClassificationModelWrapper(WrapperBase):
    def __init__(self, model, info, learning_rate=1e-4, epochs=200, optimizer=None):
        super().__init__(
            model=model,
            info=info,
            learning_rate=learning_rate,
            epochs=epochs,
            optimizer=optimizer,
        )
        self.model = model["model"]
        self.tokenizer = model["tokenizer"]
        self.classifier = model["classifier"]
        self.hidden_name = name_to_final_module_map[self.model.name_or_path]
        self.criterion = nn.CrossEntropyLoss()
        self.acc_train = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.acc_val = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.acc_test = Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        """
        output.last_hidden_state (batch_size, token_num, hidden_size): hidden representation for each token in each sequence of the batch.
        output.pooler_output (batch_size, hidden_size): take hidden representation of [CLS] token in each sequence, run through BertPooler module (linear layer with Tanh activation)
        """
        if token_type_ids is not None:
            output = self.model(
                input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )
        else:
            output = self.model(
                input_ids,
                attention_mask=attention_mask,
            )

        hidden = output[self.hidden_name]
        output = self.classifier(hidden)
        output = torch.sigmoid(output)
        loss = 0

        if labels is not None:
            # !: the squeezed labels will have an empty shape if batch-size=1
            # labels = labels.squeeze()
            labels = labels.view(-1)
            output = output.view(-1, self.num_classes)
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        token_type_ids = batch.get("token_type_ids", None)

        loss, outputs = self.forward(input_ids, attention_mask, token_type_ids, labels)
        _, pred_ids = torch.max(outputs, dim=1)
        labels = labels[0] if len(labels) == 1 else labels.squeeze()

        acc = self.acc_train(pred_ids, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log(
            "train_acc", self.acc_train, on_step=True, on_epoch=False, prog_bar=True
        )

        return {
            "loss": loss,
            "predictions": outputs,
            "labels": labels,
            "train_accuracy": acc,
        }

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        token_type_ids = batch.get("token_type_ids", None)
        loss, outputs = self.forward(input_ids, attention_mask, token_type_ids, labels)
        _, pred_ids = torch.max(outputs, dim=1)
        labels = labels[0] if len(labels) == 1 else labels.squeeze()

        # self.loss_mean_val(loss)
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
        loss, outputs = self.forward(input_ids, attention_mask, token_type_ids, labels)
        _, pred_ids = torch.max(outputs, dim=1)
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
        _, outputs = self.forward(input_ids, attention_mask, labels=None)
        _, pred_ids = torch.max(outputs, dim=1)
        return {"batch_idx": batch_idx, "outputs": outputs, "pred_ids": pred_ids}
