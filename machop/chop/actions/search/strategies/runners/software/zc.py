import math
import torch
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.text import Perplexity
from torchmetrics import MeanMetric
from transformers import get_scheduler

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from .base import SWRunnerBase

import torch.nn.functional as F
from ....search_space.zero_cost_nas.pruners.predictive import find_measures

'''
This Python file is dedicated to the implementation of a Runner for Zero Cost Neural Architecture Search (NAS). Zero Cost NAS aims to reduce the computational cost of NAS by using proxy measures to estimate the performance of neural network architectures without the need for full training.

Key Components:

1. **RunnerZeroCost Class**: This is a subclass of `SWRunnerBase`, specifically designed to run the zero-cost NAS process.

2. **_post_init_setup Method**: This initializes the available metrics for zero-cost NAS. The metrics include 'fisher', 'grad_norm', 'grasp', 'l2_norm', 'plain', 'snip', 'synflow', 'naswot', 'naswot_relu', 'tenas', and 'zico'. These are proxy measures that estimate the quality of a neural network architecture without requiring full training.

3. **_setup_metric Method**: This sets up the metric to be used based on the task (classification or language modeling) and the type of model (vision or NLP). Different tasks and model types may require different evaluation metrics.

4. **nlp_cls_forward, nlp_lm_forward, and vision_cls_forward Methods**: These define the forward pass for NLP classification, NLP language modeling, and vision classification tasks respectively. They are responsible for processing the input batch, passing it through the model, computing the loss, updating the metric, and returning the loss.

5. **forward Method**: This performs the forward pass based on the task and the type of model. It delegates the forward pass to the appropriate method based on the task and model type.

6. **compute Method**: This computes the loss and the metric. These values are used to evaluate the performance of the current architecture.

7. **__call__ Method**: This is the main method that runs the zero-cost NAS process. It computes the zero-cost metrics for the given model and data, supporting both training and validation data loaders.

The zero-cost metrics are computed using the `find_measures` function from the `predictive` module in the `zero_cost_nas.pruners` package. This function takes the model, the data loader, information about the data load, the device, the loss function, the names of the measures to compute, and an optional array to store the measures. It returns a dictionary with the computed measures.

For each metric in the configuration, the `__call__` method checks if it is in the list of available metrics. If it is, it computes the metric using the `find_measures` function and stores it in the `zero_cost_metrics` dictionary. If the metric is not in the list of available metrics, it raises a ValueError.

In summary, the `RunnerZeroCost` class enables efficient zero-cost NAS by computing proxy measures that estimate the performance of different neural network architectures without requiring full training.
'''


def get_optimizer(model, optimizer: str, learning_rate, weight_decay=0.0):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    match optimizer:
        case "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters, lr=learning_rate
            )
        case "adam":
            optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=learning_rate)
        case "sgd":
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=learning_rate)
        case _:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

    return optimizer


class RunnerZeroCost(SWRunnerBase):

    def _post_init_setup(self) -> None:
        self.available_metrics = ["fisher", "grad_norm", "grasp", "l2_norm",
                        "plain", "snip", "synflow", "naswot", "naswot_relu", "tenas", "zico"]
        self.loss = MeanMetric().to(self.accelerator)
        self._setup_metric()

    def _setup_metric(self):
        if self.model_info.is_vision_model:
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

    def nlp_cls_forward(self, batch, model):
        batch.pop("sentence")
        batch = {
            k: v.to(self.accelerator) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        outputs = model(**batch)
        loss = outputs["loss"]
        logits = outputs["logits"]
        labels = batch["labels"]
        labels = labels[0] if len(labels) == 1 else labels.squeeze()
        self.metric(logits, labels)
        self.loss(loss)
        return loss

    def nlp_lm_forward(self, batch, model):
        raise NotImplementedError()

    def vision_cls_forward(self, batch, model):
        raise NotImplementedError()

    def forward(self, task: str, batch: dict, model):
        if self.model_info.is_vision_model:
            match self.task:
                case "classification" | "cls":
                    loss = self.vision_cls_forward(batch, model)
                case _:
                    raise ValueError(f"task {self.task} is not supported.")
        elif self.model_info.is_nlp_model:
            match self.task:
                case "classification" | "cls":
                    loss = self.nlp_cls_forward(batch, model)
                case "language_modeling" | "lm":
                    loss = self.nlp_lm_forward(batch, model)
                case _:
                    raise ValueError(f"task {self.task} is not supported.")
        else:
            raise ValueError(f"model type {self.model_info} is not supported.")

        return loss

    def compute(self) -> dict[str, float]:
        reduced = {"loss": self.loss.compute().item()}
        if isinstance(self.metric, Perplexity):
            reduced["perplexity"] = self.metric.compute().item()
        elif isinstance(self.metric, MulticlassAccuracy):
            reduced["accuracy"] = self.metric.compute().item()
        else:
            raise ValueError(f"metric {self.metric} is not supported.")
        return reduced

    def __call__(self, data_module, model, sampled_config) -> dict[str, float]:

        zero_cost_metrics = {}

        data_loader = self.config["data_loader"]
        metric_names = self.config["metrics"]
        
        if data_loader == "train_dataloader":
            dataloader = data_module.train_dataloader()
        elif data_loader == "val_dataloader":
            dataloader = data_module.val_dataloader()

        dataload_info = ('random', 1, 10)
        device = self.accelerator
        
        # import pdb; pdb.set_trace()
        for metric_name in metric_names:
            if metric_name in self.available_metrics:
                # print(f"Computing {metric_name}")
                zero_cost_metrics[metric_name] = find_measures(model, 
                                                dataloader,
                                                dataload_info, # a tuple with (dataload_type = {random, grasp}, number_of_batches_for_random_or_images_per_class_for_grasp, number of classes)
                                                device, 
                                                loss_fn=F.cross_entropy, 
                                                measure_names=[metric_name],
                                                measures_arr=None)[metric_name]
                # print("zero_cost_metrics")
                # print(zero_cost_metrics)
                # print(zero_cost_metrics[metric_name])
            else:
                raise ValueError("Zero cost metrics should be chosen from ['fisher', 'grad_norm', 'grasp', 'l2_norm', 'plain', 'snip', 'synflow', 'naswot', 'naswot_relu', 'tenas', 'zico']!!!")

        return zero_cost_metrics
