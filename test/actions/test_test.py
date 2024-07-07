import os
import sys
from pathlib import PosixPath
import torch
import torch.nn as nn

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "machop"
    )
)

from chop.actions.test import test
from chop.models import get_model_info, get_model, get_tokenizer
from chop.dataset import get_dataset_info, MaseDataModule


def test_test():
    model = "toy"
    dataset = "toy_tiny"
    task = "cls"
    optimizer = "adam"
    learning_rate = 1e-5
    weight_decay = 0.99999
    scheduler_args = None
    plt_trainer_args = {
        "max_epochs": 3,
        "devices": 1,
        "accelerator": "cpu",
    }

    is_pretrained = False
    batch_size = 2
    max_token_len = 512
    num_workers = 0
    disable_dataset_cache = False
    accelerator = "cpu"
    load_name = None
    load_type = "pl"

    dataset_info = get_dataset_info(dataset)

    checkpoint, tokenizer = None, None

    model_info = get_model_info(model)
    model = get_model(
        name=model,
        task=task,
        dataset_info=dataset_info,
        checkpoint=checkpoint,
        pretrained=is_pretrained,
        quant_config=None,
    )

    data_module = MaseDataModule(
        name=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        tokenizer=tokenizer,
        max_token_len=max_token_len,
        load_from_cache_file=not disable_dataset_cache,
        model_name=model,
    )

    params = {
        "model": model,
        "model_info": model_info,
        "data_module": data_module,
        "dataset_info": dataset_info,
        "task": task,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "plt_trainer_args": plt_trainer_args,
        "save_path": None,
        "visualizer": "tensorboard",
        "load_name": load_name,
        "load_type": load_type,
        "auto_requeue": False,
    }

    test(**params)


test_test()
