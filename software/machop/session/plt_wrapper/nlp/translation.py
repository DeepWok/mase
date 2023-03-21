import torch
import torch.nn as nn
from torchmetrics.text.bleu import BLEUScore

from ..base import WrapperBase

name_to_final_module_map = {
    # TODO: double check on how to extract classifier from last_hidden_state
    "facebook/opt-125m": "last_hidden_state",
    "facebook/opt-350m": "last_hidden_state",
    "facebook/opt-1.3b": "last_hidden_state",
    "facebook/opt-2.7b": "last_hidden_state",
    "facebook/opt-13b": "last_hidden_state",
    "facebook/opt-30b": "last_hidden_state",
    "facebook/opt-66b": "last_hidden_state",
    # BERT-ish model makes use a of a pooler
    "roberta-base": "pooler_output",
    "roberta-large": "pooler_output",
}


class NLPTranslationModelWrapper(WrapperBase):
    def __init__(self, model, info, learning_rate, epochs=200, optimizer=None):
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
        self.bleu = BLEUScore(n_gram=4, smooth=False)

        self.train_bleus, self.train_losses = [], []
        self.val_bleus, self.val_losses = [], []
        self.test_bleus, self.test_losses = [], []

        self.criterion = nn.CrossEntropyLoss()

    def get_bleu(self, output_ids, labels):
        label_str = self.tokenizer.batch_decode(labels)
        tgt_lns = [str.strip(s) for s in label_str]
        pred_str = self.tokenizer.batch_decode(output_ids)
        pred_lns = [str.strip(s) for s in pred_str]
        return self.bleu(preds=pred_lns, target=tgt_lns)

    def forward(
        self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask
    ):
        """
        output.last_hidden_state (batch_size, token_num, hidden_size): hidden representation for each token in each sequence of the batch.
        output.pooler_output (batch_size, hidden_size): take hidden representation of [CLS] token in each sequence, run through BertPooler module (linear layer with Tanh activation)
        """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        if "t5" in self.model.name_or_path:
            logits = output.logits
            loss = self.criterion(
                logits.view(-1, logits.size(-1)), decoder_input_ids.view(-1)
            )
        else:
            loss = output.loss
        return output, loss

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_attention_mask = batch["decoder_attention_mask"]
        outputs, loss = self.forward(
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask
        )
        _, pred_ids = torch.max(outputs["logits"], dim=1)
        labels = decoder_input_ids
        labels = labels[0] if len(labels) == 1 else labels.squeeze()
        bleu = self.get_bleu(pred_ids, labels)
        self.train_losses.append(loss)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_bleu", bleu, prog_bar=True)
        return {
            "loss": loss,
            "predictions": outputs,
            "labels": labels,
            "train_bleu": bleu,
        }

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_attention_mask = batch["decoder_attention_mask"]
        outputs, loss = self.forward(
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask
        )
        _, pred_ids = torch.max(outputs.logits, dim=1)
        labels = decoder_input_ids
        labels = labels[0] if len(labels) == 1 else labels.squeeze()
        bleu = self.get_bleu(pred_ids, labels)
        self.val_losses.append(loss)
        self.val_bleus.append(bleu)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_bleu", bleu, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_attention_mask = batch["decoder_attention_mask"]
        outputs, loss = self.forward(
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask
        )
        _, pred_ids = torch.max(outputs.logits, dim=1)
        labels = decoder_input_ids
        labels = labels[0] if len(labels) == 1 else labels.squeeze()
        bleu = self.get_bleu(pred_ids, labels)
        self.test_losses.append(loss)
        self.test_bleus.append(bleu)
        return loss

    def on_train_epoch_end(self):
        train_mean_loss = torch.mean(
            torch.tensor(self.train_losses, dtype=torch.float32)
        )
        train_mean_bleu = torch.mean(
            torch.tensor(self.train_bleus, dtype=torch.float32)
        )
        self.train_losses = []
        self.train_bleus = []
        self.log(
            "train_mean_loss",
            train_mean_loss,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train_mean_bleu",
            train_mean_bleu,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return {"train_mean_loss": train_mean_loss, "train_mean_bleu": train_mean_bleu}

    def on_validation_epoch_end(self):
        mean_loss = torch.mean(torch.tensor(self.val_losses, dtype=torch.float32))
        mean_bleu = torch.mean(torch.tensor(self.val_bleus, dtype=torch.float32))
        self.val_losses = []
        self.val_accs = []
        self.log("val_mean_loss", mean_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_mean_bleu", mean_bleu, prog_bar=True, logger=True, sync_dist=True)
        return {"val_mean_loss": mean_loss, "val_mean_bleu": mean_bleu}

    def on_test_epoch_end(self):
        mean_loss = torch.mean(torch.tensor(self.test_losses, dtype=torch.float32))
        mean_acc = torch.mean(torch.tensor(self.test_accs, dtype=torch.float32))
        self.test_losses = []
        self.test_accs = []
        self.log(
            "test_mean_loss", mean_loss, prog_bar=True, logger=True, sync_dist=True
        )
        self.log("test_mean_acc", mean_acc, prog_bar=True, logger=True, sync_dist=True)
        return {"test_mean_loss": mean_loss, "test_mean_acc": mean_acc}
