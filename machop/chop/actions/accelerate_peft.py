import math
import os
import copy
from functools import partial
import types
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from chop.tools.logger import getLogger
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from tqdm.auto import tqdm
from transformers import get_scheduler
import sklearn
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
import argparse
import toml
from deepspeed.profiling.flops_profiler import FlopsProfiler
import sys
import torch.nn.functional as F

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "machop"
    )
)
from chop.models.manual.sparse_modules import LinearSparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = getLogger(__name__)


def evaluate_lm_step(accelerator: Accelerator, model: torch.nn.Module, batch):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_attention_mask = attention_mask[..., 1:].contiguous()

    # HF's evaluate perplexity: https://huggingface.co/spaces/evaluate-metric/perplexity/blob/main/perplexity.py
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    ppl_step = torch.exp(
        (
            loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask
        ).sum(1)
        / shift_attention_mask.sum(1)
    )

    # HF's evaluate accuracy: https://huggingface.co/spaces/evaluate-metric/accuracy/blob/main/accuracy.py
    predictions = shift_logits.argmax(dim=-1)
    acc_step = float(
        accuracy_score(shift_labels.cpu().view(-1), predictions.cpu().view(-1))
    )

    return ppl_step, acc_step


def evaluate(accelerator, model, task, eval_dataloader, num_eval_samples):
    model.eval()
    ppl_step_results = []
    acc_step_results = []
    for step, batch in enumerate(eval_dataloader):
        match task:
            case "lm" | "language_modeling":
                ppl_step, acc_step = evaluate_lm_step(accelerator, model, batch)
                ppl_step_results += ppl_step.tolist()
                acc_step_results.append(acc_step)
            case _:
                raise ValueError(f"Unsupported task: {task}")

    match task:
        case "lm" | "language_modeling":
            ppl = np.mean(ppl_step_results)
            acc = np.mean(acc_step_results)
            eval_results = {"eval_ppl": ppl, "eval_acc": acc}
        case _:
            raise ValueError(f"Unsupported task: {task}")

    return eval_results


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument(
        "--peft-config-path", type=str, help="Path to a config TOML file"
    )
    config_path = parser.parse_args().peft_config_path
    config_dict = toml.load(config_path)
    return config_dict


def checkpoint_model(accelerator: Accelerator, model: torch.nn.Module, save_dir):
    # gather the checkpoint to rank 0's CPU memory to avoid GPU OOM
    # this requires enough CPU memory
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.num_processes > 1:
        full_state_dict_config = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )
        with FSDP.state_dict_type(
            unwrapped_model, StateDictType.FULL_STATE_DICT, full_state_dict_config
        ):
            state = accelerator.get_state_dict(unwrapped_model)
            # unwrapped_model is HuggingFace AutoPretrainedModel, so we can use save_pretrained() to save the checkpoint
            unwrapped_model.save_pretrained(
                save_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=state,
            )
    else:
        unwrapped_model.save_pretrained(
            save_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )


def train_step(accelerator: Accelerator, model: torch.nn.Module, batch, task):
    match task:
        case "lm" | "language_modeling":
            input_ids = batch["input_ids"]
            labels = batch["input_ids"].detach().clone()
            attention_mask = batch["attention_mask"]
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs["loss"]
        case _:
            raise ValueError(f"Unsupported task: {task}")
    return loss


def get_optimizer(accelerator, model, optimizer: str, learning_rate, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
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

    optimizer = accelerator.prepare(optimizer)
    return optimizer


def compute_gradient_mask(
    accelerator: Accelerator,
    model: torch.nn.Module,
    train_dataloader,
    task,
    gradient_accumulation_steps,
    batch_number,
):
    # Create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    model = copy.deepcopy(model)
    linear_sparse_layers = [
        layer for layer in model.modules() if isinstance(layer, LinearSparse)
    ]

    train_dataloader = accelerator.prepare(train_dataloader)
    train_iterator = iter(train_dataloader)

    # Monkey-patch the Linear forward
    def temp_forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    batches = 0
    for step in range(batch_number):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)

        for layer in model.modules():
            if isinstance(layer, LinearSparse):
                layer.weight.requires_grad = True
                layer.forward = types.MethodType(temp_forward, layer)

        model.zero_grad()
        loss = train_step(accelerator, model, batch, task)
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        batches += 1

        grads_abs = [torch.zeros_like(layer.weight) for layer in linear_sparse_layers]
        for layer_idx, layer in enumerate(linear_sparse_layers):
            if isinstance(layer, LinearSparse):
                grads_abs[layer_idx] += torch.abs(layer.weight.grad)

    avg_grads_abs = [grad / batches for grad in grads_abs]

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in avg_grads_abs])
    norm_factor = torch.sum(all_scores)

    grads_abs_normalised = [
        avg_grads_abs[i] / norm_factor for i in range(len(avg_grads_abs))
    ]

    return grads_abs_normalised


def train(
    model,
    task,
    data_module,
    optimizer,
    max_epochs: int,
    max_steps: int = -1,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.0,
    gradient_accumulation_steps: int = 1,
    lr_scheduler_type: str = "linear",
    num_warmup_steps: int = 0,
    save_path: str = ".",
    load_name: str = None,
    load_type: str = "",
    evaluate_before_training: bool = False,
    profile: bool = False,
):
    fsdp_plugin = FullyShardedDataParallelPlugin(
        cpu_offload=CPUOffload(offload_params=False),
        auto_wrap_policy=partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={},
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=save_path,
        fsdp_plugin=fsdp_plugin,
    )

    if accelerator.is_main_process:
        logger.info(f"Using accelerator: {accelerator.state}")

    # Prepare model and FlopsProfiler
    if load_name is not None:
        model = type(model).from_pretrained(load_name)
    model = accelerator.prepare(model)
    prof = FlopsProfiler(model)

    # dataset
    if accelerator.is_local_main_process:
        data_module.prepare_data()

    accelerator.wait_for_everyone()
    data_module.setup()
    train_dataloader = data_module.train_dataloader()
    eval_dataloader = data_module.val_dataloader()
    train_dataloader, eval_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader
    )

    # optimizer
    optimizer = get_optimizer(
        accelerator, model, optimizer, learning_rate, weight_decay
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps / accelerator.num_processes
    )

    if max_steps == -1:
        max_steps = max_epochs * num_update_steps_per_epoch
    else:
        max_steps = min(max_steps, max_epochs * num_update_steps_per_epoch)
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_steps,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # logging
    experiment_config = {
        "model": type(model).__name__,
        "dataset": type(data_module.train_dataset).__name__,
        "task": task,
        "max_epochs": max_epochs,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lr_scheduler_type": lr_scheduler_type,
        "num_warmup_steps": num_warmup_steps,
        "save_path": save_path,
        "load_name": load_name,
        "load_type": load_type,
    }
    accelerator.init_trackers(save_path.replace("/", "_"), config=experiment_config)

    # training
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Instantaneous batch size per device = {data_module.batch_size}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {data_module.batch_size * accelerator.num_processes * gradient_accumulation_steps}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {gradient_accumulation_steps}, Total optimization steps = {max_steps}, Num update steps per epoch = {num_update_steps_per_epoch}"
        )

    progress_bar = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Training steps")

    train_iterator = iter(train_dataloader)

    # append gradients to LinearSparse Layers
    if any(isinstance(layer, LinearSparse) for layer in model.modules()):
        gradients = compute_gradient_mask(
            accelerator,
            model,
            train_dataloader,
            task,
            gradient_accumulation_steps,
            batch_number=64,
        )
        linear_sparse_layers = [
            layer for layer in model.modules() if isinstance(layer, LinearSparse)
        ]

        for layer, gradient in zip(linear_sparse_layers, gradients):
            layer.gradient = gradient.flatten()

    eval_results = {
        "eval_ppl": "NA",
        "eval_acc": "NA",
    }
    # calc perplexity and accuracy before training dependant on flag
    if evaluate_before_training:
        eval_results = evaluate(
            accelerator, model, task, eval_dataloader, len(eval_dataloader.dataset)
        )
    accelerator.log(
        eval_results,
        step=0,
    )

    performance_metrics = {"macs": "NA", "flops": "NA", "params": "NA"}

    # begin training
    for step in range(max_steps):
        epoch = step // num_update_steps_per_epoch
        model = model.train()

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)

        # Starts deepseed FLOPs profiler
        if step == max_steps - 1:
            prof.start_profile()

        loss = train_step(accelerator, model, batch, task)
        loss = loss / gradient_accumulation_steps

        def get_profile(print=profile):
            prof.stop_profile()
            macs, flops, params = (
                prof.get_total_macs(),
                prof.get_total_flops(),
                prof.get_total_params(),
            )
            if print == True:
                prof.print_model_profile(
                    profile_step=max_steps - 1
                )  # prints in detail information regarding each of the models submodules
            else:
                pass

            prof.end_profile()
            return macs, flops, params

        if step == max_steps - 1:
            macs, flops, params = get_profile()
            performance_metrics = {"macs": macs, "flops": flops, "params": params}

        accelerator.backward(loss)

        if (step + 1) % gradient_accumulation_steps == 0 or step == max_steps - 1:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        accelerator.log({"train_loss": loss.detach()}, step=step)
        accelerator.log(performance_metrics, step=step)
        # complete an epoch

        if (step + 1) % num_update_steps_per_epoch == 0 or step == max_steps - 1:
            # evaluate
            eval_results = evaluate(
                accelerator, model, task, eval_dataloader, len(eval_dataloader.dataset)
            )
            accelerator.log(
                eval_results,
                step=epoch + 1,
            )

            # checkpoint
            save_dir = os.path.join(save_path, f"{epoch + 1}")
            checkpoint_model(accelerator, model, save_dir)
            if accelerator.is_local_main_process:
                logger.info(f"Saving checkpoint to {save_dir}")

        progress_bar.set_postfix(
            {
                "epoch": epoch + 1,
                "train_loss": loss.item(),
                "eval_ppl": eval_results["eval_ppl"],
                "eval_acc": eval_results["eval_acc"],
            }
        )

        progress_bar.update(1)
    accelerator.wait_for_everyone()
    accelerator.end_training()

    if accelerator.is_local_main_process:
        logger.info("Training completed")
