# Model Zoo for machop
#
from os import PathLike

# TODO: fix patched models
from .patched import (
    is_patched_model,
    get_patched_model,
    get_patched_model_info,
    get_patched_model_tokenizer,
    get_patched_model_cls,
    get_patched_model_config_cls,
    get_patched_model_tokenizer_cls,
)

from .manual import (
    is_manual_model,
    get_manual_model,
    get_manual_model_cls,
    get_manual_model_config_cls,
    get_manual_model_tokenizer_cls,
    get_manual_model_info,
    get_manual_model_tokenizer,
)

from .huggingface_nlp_models import (
    is_hf_nlp_model,
    get_hf_nlp_model,
    get_hf_nlp_model_info,
    get_hf_nlp_model_tokenizer,
    get_hf_nlp_model_tokenizer_cls,
    get_hf_nlp_model_config_cls,
)
from .vision import get_vision_model, get_vision_model_info, is_vision_model
from .physical import (
    get_physical_model,
    get_physical_model_info,
    is_physical_model,
)
from .nerf import (
    get_nerf_model,
    get_nerf_model_info,
    is_nerf_model,
)
from .toys import get_toy_model, get_toy_model_info, is_toy_model
from .utils import MaseModelInfo, ModelSource, ModelTaskType

import torch.nn as nn


def get_model_info(name: str) -> MaseModelInfo:
    if is_manual_model(name):
        info = get_manual_model_info(name)
    elif is_hf_nlp_model(name):
        info = get_hf_nlp_model_info(name)
    elif is_vision_model(name):
        info = get_vision_model_info(name)
    elif is_physical_model(name):
        info = get_physical_model_info(name)
    elif is_nerf_model(name):
        info = get_nerf_model_info(name)
    elif is_toy_model(name):
        info = get_toy_model_info(name)
    elif is_patched_model(name):
        info = get_patched_model_info(name)
    else:
        raise ValueError(f"Model {name} not found")

    return info


def get_model(
    name: str,
    task: str,
    dataset_info: dict,
    pretrained: bool,
    checkpoint: str | PathLike = None,
    quant_config: dict = None,
):
    model_info = get_model_info(name)

    model_kwargs = {
        "name": name,
        "task": task,
        "dataset_info": dataset_info,
        "pretrained": pretrained,
        "checkpoint": checkpoint,
    }

    if model_info.is_quantized:
        model_kwargs["quant_config"] = quant_config

    match model_info.model_source:
        case ModelSource.HF_TRANSFORMERS:
            model = get_hf_nlp_model(**model_kwargs)
        case ModelSource.MANUAL:
            model = get_manual_model(**model_kwargs)
        case ModelSource.PATCHED:
            model = get_patched_model(**model_kwargs)
        case ModelSource.TORCHVISION | ModelSource.VISION_OTHERS:
            model = get_vision_model(**model_kwargs)
        case ModelSource.PHYSICAL:
            model = get_physical_model(**model_kwargs)
        case ModelSource.TOY:
            model = get_toy_model(**model_kwargs)
        case ModelSource.NERF:
            model = get_nerf_model(**model_kwargs)
        case _:
            raise ValueError(f"Model source {model_info.model_source} not supported")

    if dataset_info.name == "mnist" and model_info.task_type._value_ == "vision":
        if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
            layers = model.features
        elif isinstance(model, nn.Sequential):
            layers = model
        else:
            # If model is neither, treat it as a generic nn.Module.
            layers = model.children()

        # Now extract the very first input layer
        for i, layer in enumerate(layers):
            first_layer = next(layer.children())
            break

        # Check if first layer is a Conv2d expecting 3 input channels (whilst MNIST comes in gray-scale)
        if isinstance(first_layer, nn.Conv2d):
            if first_layer.in_channels != 1:
                # Define a new Conv2d layer to take 1 input channel and output 3 channels
                new_first_layer = nn.Conv2d(
                    1, 3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
                )

                # Now, add this layer at the beginning of the model accepting 3 channels
                model = nn.Sequential(new_first_layer, model)

    return model


def get_tokenizer(name: str, checkpoint: str | PathLike = None):
    """
    Get the tokenizer for the model.

    Args:
        name: The name of the model.
        checkpoint: The path to the checkpoint. This is optional for HuggingFace models, but required for manual models.
    """
    model_info = get_model_info(name)
    assert model_info.is_nlp_model, f"Model {name} is not an NLP model"

    match model_info.model_source:
        case ModelSource.HF_TRANSFORMERS:
            tokenizer = get_hf_nlp_model_tokenizer(name, checkpoint)
        case ModelSource.MANUAL:
            tokenizer = get_manual_model_tokenizer(name, checkpoint)
        case ModelSource.PATCHED:
            tokenizer = get_patched_model_tokenizer(name, checkpoint)
        case _:
            raise ValueError(f"Model source {model_info.model_source} not supported")
    return tokenizer
