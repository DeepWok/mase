import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from .opt_patched import (
    opt_patched_model_cls_to_original_model_cls,
    opt_patched_model_cls_to_required_input_args,
    opt_patched_model_name_to_hidden_size,
    opt_patched_model_name_to_output_hidden_states_name,
    opt_patched_model_name_to_pooler_size,
    opt_patched_name_to_patched_model_mapping,
)

model_name_to_patched_model_mapping = {} | opt_patched_name_to_patched_model_mapping

# patched_model_cls_to_get_dummy_input = {} | opt_patched_model_cls_to_dummy_input_fn
patched_model_cls_to_required_input_args = (
    {} | opt_patched_model_cls_to_required_input_args
)

patched_model_cls_to_original_model_cls = (
    {} | opt_patched_model_cls_to_original_model_cls
)

# pooler: ? -> hidden_size
model_to_pooler_size = {} | opt_patched_model_name_to_pooler_size

# classifier: config.hidden_size -> num_classes
model_to_hidden_size = {} | opt_patched_model_name_to_hidden_size

# see 'name_to_final_module_map' in machop.session.plt_wrapper.nlp_classification
patched_model_name_to_output_hidden_states_name = (
    {} | opt_patched_model_name_to_output_hidden_states_name
)


def get_modeling_mapping(model_name: str):
    assert (
        model_name in model_name_to_patched_model_mapping
    ), f"model name {model_name} is not one of nlp patched models"
    model_mapping = model_name_to_patched_model_mapping[model_name]
    return model_mapping


def tokenizer_from_pretrained(
    model_name: str, cache_dir: str, return_dict: bool = True
):
    model_mapping = get_modeling_mapping(model_name)

    tokenizer_cls = model_mapping.get("tokenizer", None)
    tokenizer_name = "@".join(model_name.split("@")[:-1])
    if tokenizer_cls is None:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, cache_dir=cache_dir, return_dict=return_dict
        )
    else:
        tokenizer = tokenizer_cls.from_pretrained(
            tokenizer_name, cache_dir=cache_dir, return_dict=return_dict
        )
    return tokenizer


def config_from_pretrained(model_name: str):
    model_mapping = get_modeling_mapping(model_name)
    config_cls = model_mapping["config"]
    original_config_cls = patched_model_cls_to_original_model_cls[config_cls]
    original_model_name = "@".join(model_name.split("@")[:-1])
    pretrained_original_config = original_config_cls.from_pretrained(
        original_model_name
    )
    config = config_cls.from_dict(pretrained_original_config.to_dict())
    return config


def model_for_task_from_config(model_name: str, task: str):
    model_mapping = get_modeling_mapping(model_name)
    model_cls = model_mapping.get(task, None)
    if model_cls is None:
        raise ValueError(f"Task {task} is not supported for model {model_name}")
    config = config_from_pretrained(model_name=model_name)
    model = model_cls(config=config)
    return model


def model_for_task_from_pretrained_original(
    model_name: str, task: str, return_dict: bool, cache_dir: str
):
    model: torch.nn.Module = model_for_task_from_config(
        model_name=model_name, task=task
    )

    original_model_cls = patched_model_cls_to_original_model_cls[type(model)]
    original_model_name = "@".join(model_name.split("@")[:-1])

    original_model: torch.nn.Module = original_model_cls.from_pretrained(
        original_model_name,
        return_dict=return_dict,
        cache_dir=cache_dir,
    )
    pretrained_state_dict = original_model.state_dict()

    model.load_state_dict(pretrained_state_dict)
    return model


# TODO: check the pooler? should we pool at the first token for opt?
class Pooler(nn.Module):
    def __init__(self, in_hidden_size, out_hidden_size):
        super().__init__()
        self.dense = nn.Linear(in_hidden_size, out_hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def get_patched_nlp_model(name, task, info, checkpoint=None, pretrained=True):
    if task not in [
        "classification",
        "cls",
        "translation",
        "tran",
        "language_modeling",
        "lm",
    ]:
        raise ValueError("task must be a valid value for NLP models")

    num_classes = info["num_classes"]

    tokenizer = tokenizer_from_pretrained(
        name, cache_dir=os.path.abspath("./cache/tokenizer_cache_dir"), return_dict=True
    )

    print(f"Loaded tokenizer from {name}")
    if pretrained:
        if checkpoint is not None:
            model_cls = get_modeling_mapping(name)[task]
            model = model_cls.from_pretrained(checkpoint)
            print(f"Loaded model from {checkpoint}")
        else:
            if task in ["language_modeling", "lm"]:
                model = model_for_task_from_pretrained_original(
                    model_name=name,
                    task="lm",
                    return_dict=True,
                    cache_dir=os.path.abspath("./cache/model_cache_dir"),
                )
            elif task in ["translation", "tran"]:
                model = model_for_task_from_pretrained_original(
                    model_name=name,
                    task="tran",
                    return_dict=True,
                    cache_dir=os.path.abspath("./cache/model_cache_dir"),
                )
            else:
                model = model_for_task_from_pretrained_original(
                    model_name=name,
                    task="model",
                    return_dict=True,
                    cache_dir=os.path.abspath("./cache/model_cache_dir"),
                )
            print(f"Loaded pretrained original model in HuggingFace into {name}")
    else:
        if task in ["language_modeling", "lm"]:
            # raise ValueError(
            #     "Language modeling task is not supported to train from scratch, please use --pretrained flag"
            # )
            model = model_for_task_from_config(model_name=name, task="lm")

        elif task in ["classification", "cls"]:
            model = model_for_task_from_config(model_name=name, task="model")
        else:
            model = model_for_task_from_config(model_name=name, task="tran")
        print(f"Model randomly initialized")

    if task in ["classification", "cls"]:
        hidden_size = model_to_hidden_size.get(name, model.config.hidden_size)
        classifier = nn.Linear(hidden_size, num_classes)
        if name in model_to_pooler_size:
            in_hidden, out_hidden = model_to_pooler_size[name]
            pooler = Pooler(in_hidden, out_hidden)
            classifier = nn.Sequential(pooler, classifier)
    else:
        classifier = None

    # set name_or_path
    model.name_or_path = name
    return {
        "model": model,
        "tokenizer": tokenizer,
        "classifier": classifier,
    }
