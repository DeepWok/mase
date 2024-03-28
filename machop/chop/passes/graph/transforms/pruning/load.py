from functools import partial
import logging

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


# Constants ----------------------------------------------------------------------------
# Pruneable MASE operators
# NOTE: We don't do activation pruning for conv1d and linear layers. 
PRUNEABLE_OPS = {"conv1d": nn.Conv1d, "conv2d": nn.Conv2d, "linear": nn.Linear}

WEIGHT_PRUNE_METHODS = ["random", "l1-norm", "l2-norm", "l1-norm-single", "l2-norm-single", "l1-norm-multi", "l2-norm-multi"]
ACTIVATION_PRUNE_METHODS = ["random", "l1-norm", "l2-norm", "feature-map-similarity", "feature-map-l1-norm", "feature-map-l2-norm"]

# A registry of available pruning strategies (i.e. algorithms)
# PRUNE_METHODS = {
#     # A basic one-shot pruner that prunes to a given sparsity level
#     "level-pruner": LevelPruner,
#     "channel-pruner": ChannelPruner,
#     # Add more here...
# }


def load(config: dict):
    sparsity = config.get("sparsity", 0.0)
    scope = config.get("scope", "local")
    granularity = config.get("granularity", "elementwise")

    # if granularity not in ["channel", "elementwise", "kernelwise", "channelwise"]:
    if granularity not in ["elementwise", "kernelwise", "channelwise"]:
        raise ValueError(
            "Unsupported pruning granularity {}. Please choose from {}".format(
                #granularity, ["channel", "elementwise", "kernelwise", "channelwise"]
                granularity, ["elementwise", "kernelwise", "channelwise"]
            )
        )
    if scope not in ["local", "global"]:
        raise ValueError(
            "Unsupported pruning scope {}. Please choose from {}".format(
                scope, ["local", "global"]
            )
        )
    if isinstance(sparsity, dict):
        # Make sure that the scope is local
        if scope != "local":  # not local
            raise ValueError("Layer-wise budgets only possible with a local scope!")

        # Verify that the keys are valid node names and that the values are valid
        names = [node.target for node in graph.fx_graph.nodes]
        for name in sparsity.keys():
            if name not in names:
                raise ValueError(f"Node {name} not found in the graph!")
            _verify_sparsity(sparsity[name])
    else:
        _verify_sparsity(sparsity)
    return sparsity, scope, granularity


def load_weight_prune_config(config: dict, graph):
    sparsity, scope, granularity = load(config)
    #print("sparsity");print(sparsity)
    #print("scope");print(scope)
    #print("granularity");print(granularity)

    method = config.get("method", "random")
    #print("method") ; print(method)
    # Validate the parameters
    if method not in WEIGHT_PRUNE_METHODS:
        raise ValueError(
            "Unsupported pruning method {}. Please choose from {}".format(
                method, WEIGHT_PRUNE_METHODS
            )
        )

    return {
        "method": method,
        "granularity": granularity,
        "scope": scope,
        "sparsity": sparsity,
    }


def load_activation_prune_config(config: dict, graph):
    sparsity, scope, granularity = load(config)

    method = config.get("method", "random")
    # Validate the parameters
    if method not in ACTIVATION_PRUNE_METHODS:
        raise ValueError(
            "Unsupported pruning method {}. Please choose from {}".format(
                method, ACTIVATION_PRUNE_METHODS
            )
        )

    return {
        "method": method,
        "granularity": granularity,
        "scope": scope,
        "sparsity": sparsity,
    }


# Get a pre-trained vision model from Torchvision and return the layer-wise sparsity
# distribution that a globally scoped pruner would compute for it.
# NOTE: Here, we assume the dataset is ImageNet, as indicated by the number of classes
# in the dataset info.  This will need fixing later to support any dataset the user
# wants to use. Also, this code is really similar to the one in the level pruner, except
# that here we do it for the full model. We should probably refactor this later.
def get_unfused_distribution(sparsity: float, criterion: callable, name: str):
    # We import here to avoid a failing test via a circular import in the CI.
    from chop.models.vision import VISION_MODELS, get_vision_model

    if name is None or name not in VISION_MODELS:
        raise ValueError(f"Expected valid model name. Got {name}")

    model = get_vision_model(name, "cls", {"num_classes": 1000}, pretrained=True)
    tensors = {}
    sparsities = {}

    for name, module in model.named_modules():
        if not isinstance(module, tuple(PRUNEABLE_OPS.values())):
            continue

        # Collect all the weight tensors in the model
        tensors[name] = {
            "tensor": module.weight.data.clone().flatten(),
            "shape": module.weight.shape,
        }

    # Compute the layer-wise sparsities!
    concatenated = torch.cat([t["tensor"] for t in tensors.values()])
    mask = (
        # Currently, there's no criterion with custom kwargs, so we don't pass it in
        criterion(concatenated, sparsity)
        if sparsity > 0.0
        else torch.ones_like(concatenated, dtype=torch.bool)
    )
    # Split the masks into chunks and compute the sparsity of each chunk
    sizes = [t["tensor"].numel() for t in tensors.values()]
    masks = torch.split(mask, sizes)
    sparsities = [1 - m.count_nonzero() / m.numel() for m in masks]
    sparsities = dict(zip(tensors.keys(), sparsities))

    return sparsities


def _verify_sparsity(sparsity):
    if not isinstance(sparsity, float):
        raise ValueError("Sparsity must be a float. Got {}".format(type(sparsity)))
    if sparsity < 0 or sparsity > 1:
        raise ValueError("Sparsity must be between 0 and 1. Got {}".format(sparsity))


# def _log_metadata(graph):
#     logger.info(
#         "\n"
#         f"Sparsity    : {measure_sparsity(graph.model):>14.3f}\n"
#         f"Params (TOT): {count_parameters(graph.model):>14,}\n"
#         f"Params (NNZ): {count_parameters(graph.model, nonzero_only=True):>14,}\n"
#         f"Buffers     : {count_buffers(graph.model):>14,}\n"
#         f"Size        : {estimate_model_size(graph.model):>14.3f} MB"
#     )
