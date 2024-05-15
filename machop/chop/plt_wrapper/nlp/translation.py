import torch
import torch.nn as nn

from torchmetrics.text import BLEUScore

from ..base import WrapperBase


class NLPTranslationModelWrapper(WrapperBase):
    def __init__(
        self,
        model,
        tokenizer,
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
        self.tokenizer = tokenizer

        self.bleu_train = BLEUScore(n_gram=4, smooth=False)
        self.bleu_val = BLEUScore(n_gram=4, smooth=False)
        self.bleu_test = BLEUScore(n_gram=4, smooth=False)

        self.criterion = nn.CrossEntropyLoss()

    def get_pred_ids_and_labels(self, output_ids, labels):
        label_str = self.tokenizer.batch_decode(labels)
        tgt_lns = [str.strip(s) for s in label_str]
        pred_str = self.tokenizer.batch_decode(output_ids)
        pred_lns = [str.strip(s) for s in pred_str]
        return (pred_lns, tgt_lns)

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        labels,
    ):
        """
        output.last_hidden_state (batch_size, token_num, hidden_size): hidden representation for each token in each sequence of the batch.
        output.pooler_output (batch_size, hidden_size): take hidden representation of [CLS] token in each sequence, run through BertPooler module (linear layer with Tanh activation)
        """
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        # if "t5" in self.model.name_or_path:
        #     logits = output.logits
        #     loss = self.criterion(
        #         logits.view(-1, logits.size(-1)), decoder_input_ids.view(-1)
        #     )
        # else:
        #     loss = output.loss
        return outputs

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_attention_mask = batch["decoder_attention_mask"]
        labels = decoder_input_ids
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        loss = outputs["loss"]
        logits = outputs["logits"]
        _, pred_ids = torch.max(logits, dim=1)
        labels = labels[0] if len(labels) == 1 else labels.squeeze()

        self.bleu_train(*self.get_pred_ids_and_labels(pred_ids, labels))
        self.log("train_bleu_step", self.bleu_train, rog_bar=True)
        self.log("train_loss_step", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_attention_mask = batch["decoder_attention_mask"]
        labels = decoder_input_ids
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        logits = outputs["logits"]
        loss = outputs["loss"]

        _, pred_ids = torch.max(logits, dim=1)

        labels = labels[0] if len(labels) == 1 else labels.squeeze()

        self.bleu_val(*self.get_pred_ids_and_labels(pred_ids, labels))
        self.loss_val(loss)

        return loss

    def on_validation_epoch_end(self):
        self.log("val_bleu_epoch", self.bleu_val, prog_bar=True)
        self.log("val_loss_epoch", self.loss_val, prog_bar=True)

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_attention_mask = batch["decoder_attention_mask"]
        labels = decoder_input_ids
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        logits = outputs["logits"]
        loss = outputs["loss"]
        _, pred_ids = torch.max(logits, dim=1)

        labels = labels[0] if len(labels) == 1 else labels.squeeze()

        self.bleu_test(*self.get_pred_ids_and_labels(pred_ids, labels))
        self.loss_test(loss)

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
