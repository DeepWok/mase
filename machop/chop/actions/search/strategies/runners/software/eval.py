import math
import torch
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.text import Perplexity
from torchmetrics import MeanMetric


from .base import SWRunnerBase


class RunnerBasicEval(SWRunnerBase):
    """
    A software runner that evaluates rebuilt model on the data_loader.

    ---
    The model's forward signature should follow Mase's convention. Check Mase's InputGenerator for more details.

    ---
    Available metrics:

    - Vision model:
        - classification: accuracy, loss
    - NLP model:
        - classification: accuracy, loss
        - language_modeling: perplexity, loss
    """

    available_metrics = ("loss", "accuracy", "perplexity")

    def _post_init_setup(self) -> None:
        self.loss = MeanMetric().to(self.accelerator)
        self._setup_metric()

        assert "num_samples" in self.config, "num_samples is not set in the config."

    def _setup_metric(self):
        if self.model_info.is_vision_model or self.model_info.is_physical_model:
            match self.task:
                case "classification" | "cls":
                    self.metric = MulticlassAccuracy(
                        num_classes=self.dataset_info.num_classes
                    ).to(self.accelerator)
                case _:
                    raise ValueError(f"task {self.task} is not supported.")
        elif self.model_info.is_nlp_model:
            match self.task:
                case "classification" | "cls":
                    self.metric = MulticlassAccuracy(
                        num_classes=self.dataset_info.num_classes
                    ).to(self.accelerator)
                case "language_modeling" | "lm":
                    self.metric = Perplexity().to(self.accelerator)
                case _:
                    raise ValueError(f"task {self.task} is not supported.")
        else:
            raise ValueError(f"model type {self.model_info} is not supported.")

    def forward(self, batch: dict[str, torch.Tensor], model):
        if self.model_info.is_vision_model or self.model_info.is_physical_model:
            match self.task:
                case "classification" | "cls":
                    return self.vision_cls_forward(batch, model)
                case _:
                    raise ValueError(f"task {self.task} is not supported.")
        elif self.model_info.is_nlp_model:
            match self.task:
                case "classification" | "cls":
                    return self.nlp_cls_forward(batch, model)
                case "language_modeling" | "lm":
                    return self.nlp_lm_forward(batch, model)
                case _:
                    raise ValueError(f"task {self.task} is not supported.")

    def vision_cls_forward(self, batch, model):
        x, y = batch[0].to(self.accelerator), batch[1].to(self.accelerator)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        acc = self.metric(logits, y)
        self.loss(loss)
        return {"loss": loss, "accuracy": acc}

    def nlp_cls_forward(self, batch, model):
        batch = {
            k: v.to(self.accelerator) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        outputs = model(**batch)
        loss = outputs["loss"]
        logits = outputs["logits"]
        acc = self.metric(logits, batch["labels"])
        self.loss(loss)
        return {"loss": loss, "accuracy": acc}

    def nlp_lm_forward(self, batch, model):
        batch = {
            k: v.to(self.accelerator) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        outputs = model(**batch)
        loss = outputs["loss"]
        logits = outputs["logits"]
        perplexity = self.metric(logits, batch["labels"])
        self.loss(loss)
        return {"loss": loss, "perplexity": perplexity}

    def compute(self) -> dict[str, float]:
        reduced = {"loss": self.loss.compute().item()}
        if isinstance(self.metric, Perplexity):
            reduced["perplexity"] = self.metric.compute().item()
        elif isinstance(self.metric, MulticlassAccuracy):
            reduced["accuracy"] = self.metric.compute().item()
        else:
            raise ValueError(f"metric {self.metric} is not supported.")
        return reduced

    def __call__(self, data_module, model, sampled_config: dict) -> dict[str, float]:
        if not isinstance(model, torch.nn.Module):
            forward_model = model.model
        else:
            forward_model = model

        num_batches = math.ceil(self.config["num_samples"] / data_module.batch_size)
        data_loader = getattr(
            data_module, self.config.get("dataloader", "val_dataloader")
        )()

        for i, batch in enumerate(data_loader):
            outputs = self.forward(batch, forward_model)
            self.loss(outputs["loss"])
            if i >= num_batches - 1:
                break
        return self.compute()
