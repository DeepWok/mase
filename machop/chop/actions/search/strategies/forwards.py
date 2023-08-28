import torch
from torchmetrics import Accuracy


class ForwardMap:
    def __init__(self, task, info):
        self.loss = torch.nn.CrossEntropyLoss()

        num_classes = info["num_classes"]
        if num_classes is None:
            num_classes = 2
        self.multi_class_acc = Accuracy("multiclass", num_classes=num_classes)
        self.forward_map = {
            "cls": self.basic_forward,
            "classification": self.basic_forward,
            "lm": self.lm_forward,
        }
        self.forward = self.forward_map[task]

    def basic_forward(self, batch, model):
        x, y = batch[0].to(model.device), batch[1].to(model.device)
        y_hat = model(x)
        loss = self.loss(y_hat, y)
        acc = self.multi_class_acc(y_hat, y)
        return {"loss": loss, "acc": acc}

    def lm_forward(self, batch, model):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)
        output = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        # loss = self.criterion(output, labels)
        loss = output["loss"]
        output = output["logits"]
        perplexity = torch.exp(loss)
        return {"loss": loss, "predictions": output, "perplexity": perplexity}
