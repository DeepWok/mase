import os
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


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def lm_cls_forward(rank, model, batch, metric_obj, loss_obj):
    batch.pop("sentence")
    batch = {
        k: v.to(rank) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
    outputs = model(**batch)
    loss = outputs["loss"]
    logits = outputs["logits"]
    labels = batch["labels"]
    labels = labels[0] if len(labels) == 1 else labels.squeeze()

    metric_obj.update(logits, labels)
    loss_obj.update(loss)
    return loss


def forward(rank, model, batch, metric_obj, loss_obj, model_info, task):
    if model_info.is_vision_model:
        match task:
            case "classification" | "cls":
                raise NotImplementedError()
            case _:
                raise ValueError(f"task {task} is not supported.")
    elif model_info.is_nlp_model:
        match task:
            case "classification" | "cls":
                loss = lm_cls_forward(rank, model, batch, metric_obj, loss_obj)
            case "language_modeling" | "lm":
                raise NotImplementedError()
            case _:
                raise ValueError(f"task {task} is not supported.")
    else:
        raise ValueError(f"model type {model_info} is not supported.")

    return loss


def prepare_dataloader(
    rank, world_size, dataset, batch_size, num_workers=0, pin_memory=False
):
    # When using a sampler in distributed mode, num_workers > 0 will cause
    # multiple workers to download the same data. This is because each worker
    # will have a different sampler and will download the same data.
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
    #

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        sampler=sampler,
    )

    return dataloader


def create_metric_obj(rank, task, model_info, dataset_info):
    if model_info.is_vision_model:
        match task:
            case "classification" | "cls":
                metric = MulticlassAccuracy(num_classes=dataset_info.num_classes).to(
                    rank
                )
            case _:
                raise ValueError(f"task {task} is not supported.")
    elif model_info.is_nlp_model:
        match task:
            case "classification" | "cls":
                metric = MulticlassAccuracy(num_classes=dataset_info.num_classes).to(
                    rank
                )
            case "language_modeling" | "lm":
                metric = Perplexity().to(rank)
            case _:
                raise ValueError(f"task {task} is not supported.")
    else:
        raise ValueError(f"model type {model_info} is not supported.")

    return metric


def compute_metric_obj(loss_obj, metric_obj) -> dict[str, float]:
    reduced = {"loss": loss_obj.compute().item()}
    if isinstance(metric_obj, Perplexity):
        reduced["perplexity"] = metric_obj.compute().item()
    elif isinstance(metric_obj, MulticlassAccuracy):
        reduced["accuracy"] = metric_obj.compute().item()
    else:
        raise ValueError(f"metric {metric_obj} is not supported.")
    return reduced


def train_loop(
    rank,
    world_size,
    model,
    model_info,
    train_dataset,
    dataset_info,
    batch_size,
    task,
    runner_config,
    metric_reference,
):
    setup(rank=rank, world_size=world_size)
    train_dataloader = prepare_dataloader(
        rank=rank,
        world_size=world_size,
        dataset=train_dataset,
        batch_size=batch_size,
        # num_workers=os.cpu_count() // world_size,
        num_workers=0,
        pin_memory=False,
    )

    metric_obj = create_metric_obj(rank, task, model_info, dataset_info)
    loss_obj = MeanMetric().to(rank)

    model = model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    optimizer = get_optimizer(
        model=model,
        optimizer=runner_config["optimizer"],
        learning_rate=runner_config["learning_rate"],
        weight_decay=runner_config.get("weight_decay", 0.0),
    )

    lr_scheduler = get_scheduler(
        name=runner_config["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=runner_config["num_warmup_steps"],
        num_training_steps=runner_config["max_epochs"]
        * (len(train_dataloader) // world_size),
    )

    for epoch in range(runner_config["max_epochs"]):
        train_dataloader.sampler.set_epoch(epoch)

        loss_obj.reset()
        metric_obj.reset()
        for batch in train_dataloader:
            optimizer.zero_grad()
            model.train()
            loss = forward(
                rank=rank,
                model=model,
                batch=batch,
                metric_obj=metric_obj,
                loss_obj=loss_obj,
                model_info=model_info,
                task=task,
            )
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

    if rank == 0:
        reduced = compute_metric_obj(loss_obj, metric_obj)
        for k, v in reduced.items():
            metric_reference[k] = v
    cleanup()


class RunnerBasicTrain(SWRunnerBase):
    available_metrics = ("loss", "accuracy", "perplexity")

    def _post_init_setup(self) -> None:
        # self.loss = MeanMetric().to(self.accelerator)
        # self._setup_metric()

        self.metric_reference = dict()

    def __call__(self, data_module, model, sampled_config) -> dict[str, float]:
        """
        !: This ddp is not working. To be fixed.
        """
        world_size = torch.cuda.device_count()
        mp.spawn(
            train_loop,
            args=(
                world_size,
                model,
                self.model_info,
                data_module.train_dataset,
                self.dataset_info,
                data_module.batch_size,
                self.task,
                self.config,
                self.metric_reference,
            ),
            nprocs=world_size,
            join=True,
        )

        return self.metric_reference
