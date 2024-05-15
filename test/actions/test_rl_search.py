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

from chop.actions.search import search
from chop.models import get_model_info, get_model, get_tokenizer
from chop.dataset import get_dataset_info, MaseDataModule


def test_rl_search():
    model = "toy"
    dataset = "toy_tiny"
    task = "cls"
    is_pretrained = False
    batch_size = 2
    max_token_len = 512
    num_workers = 0
    disable_dataset_cache = False
    accelerator = "cpu"
    load_name = None
    load_type = "pl"

    config = {
        "search": {
            "search_space": {
                "name": "graph/quantize/mixed_precision_ptq",
                "setup": {"by": "name"},
                "seed": {
                    "default": {
                        "config": {
                            "name": ["integer"],
                            "data_in_width": [8, 16, 32],
                            "data_in_frac_width": [4, 8, 16],
                            "weight_width": [8, 16, 32],
                            "weight_frac_width": [4, 8, 16],
                            "bias_width": [8, 16, 32],
                            "bias_frac_width": [4, 8, 16],
                        }
                    },
                    "linear": {
                        "config": {
                            "name": ["integer"],
                            "data_in_width": [8, 16, 32],
                            "data_in_frac_width": ["NA"],
                            "weight_width": [8, 16, 32],
                            "weight_frac_width": ["NA"],
                            "bias_width": [8, 16, 32],
                            "bias_frac_width": ["NA"],
                        }
                    },
                    "conv2d": {
                        "config": {
                            "name": ["integer"],
                            "data_in_width": [8, 16, 32],
                            "data_in_frac_width": ["NA"],
                            "weight_width": [8, 16, 32],
                            "weight_frac_width": ["NA"],
                            "bias_width": [8, 16, 32],
                            "bias_frac_width": ["NA"],
                        }
                    },
                },
            },
            "strategy": {
                "name": "rl",
                "algorithm": "ppo",
                "env": "mixed_precision_paper",
                "device": "cpu",
                "total_timesteps": 1000,
                "n_steps": 32,
                "n_envs": 4,
                "eval_freq": 256,
                "save_freq": 256,
                "episode_max_len": 1000,
                "learning_rate": 2.5e-4,
                "save_name": "tmp_rl",
                "setup": {"sum_scaled_metrics": False},
                "sw_runner": {
                    "basic_evaluation": {
                        "data_loader": "val_dataloader",
                        "num_samples": 512,
                    }
                },
                "hw_runner": {"average_bitwidth": {"compare_to": 32}},
                "metrics": {
                    "accuracy": {
                        "scale": 0.7,
                        "direction": "maximize",
                        "lower_bound": 0,
                        "upper_bound": 1,
                    },
                    "average_bitwidth": {
                        "scale": 0.3,
                        "direction": "minimize",
                        "lower_bound": 6,
                        "upper_bound": 12,
                    },
                },
            },
        },
    }

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

    search_params = {
        "model": model,
        "model_info": model_info,
        "task": task,
        "dataset_info": dataset_info,
        "data_module": data_module,
        "accelerator": accelerator,
        "search_config": config,
        "save_path": PosixPath("./logs/rl_search"),
        "load_name": load_name,
        "load_type": load_type,
        "visualizer": "tensorboard",
    }

    search(**search_params)


test_rl_search()
