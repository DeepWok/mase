import torch
from torchmetrics import Accuracy


class ForwardMap:
    def __init__(self, task, info):
        self.loss = torch.nn.CrossEntropyLoss()
        self.multi_class_acc = Accuracy("multiclass", num_classes=info["num_classes"])
        self.forward_map = {
            "cls": self.basic_forward,
            "classificaiton": self.basic_forward,
            "lm": self.lm_forward,
        }
        self.forward = self.forward_map[task]

    def basic_forward(self, batch, model):
        x, y = batch[0], batch[1]
        y_hat = model(x)
        loss = self.loss(y_hat, y)
        acc = self.multi_class_acc(y_hat, y)
        return {"loss": loss, "acc": acc}

    def lm_forward(self, batch, model):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        output = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        # loss = self.criterion(output, labels)
        loss = output["loss"]
        output = output["logits"]
        perplexity = torch.exp(loss)
        return {"loss": loss, "predictions": output, "perplexity": perplexity}
