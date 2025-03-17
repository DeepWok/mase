from functools import partial
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Pruneable MASE operators
# NOTE: We don't do activation pruning for conv1d and linear layers.
# ------------------------------------------------------------------------------
PRUNEABLE_OPS = {
    "conv1d": nn.Conv1d,
    "conv2d": nn.Conv2d,
    "linear": nn.Linear
}

# ------------------------------------------------------------------------------
# Add "movement" to these lists so the pruning pass recognizes it.
# ------------------------------------------------------------------------------
# Add "hwpq" to the valid weight pruning methods
WEIGHT_PRUNE_METHODS = ["random", "l1-norm", "movement", "hwpq"]
ACTIVATION_PRUNE_METHODS = ["random", "l1-norm", "movement"]

# ------------------------------------------------------------------------------
# The main load functions that parse and validate pruning configs
# ------------------------------------------------------------------------------
def load(config: dict, graph):
    """
    Common loader for the config, which also verifies scope, granularity, and sparsity.
    """
    sparsity = config.get("sparsity", 0.0)
    scope = config.get("scope", "local")
    granularity = config.get("granularity", "elementwise")

    if granularity not in ["channel", "elementwise"]:
        raise ValueError(
            "Unsupported pruning granularity {}. Please choose from {}".format(
                granularity, ["channel", "elementwise"]
            )
        )
    if scope not in ["local", "global"]:
        raise ValueError(
            "Unsupported pruning scope {}. Please choose from {}".format(
                scope, ["local", "global"]
            )
        )
    # If the user provided a dictionary of layer-wise sparsities, validate them:
    if isinstance(sparsity, dict):
        # Make sure that the scope is local
        if scope != "local":
            raise ValueError("Layer-wise budgets only possible with a local scope!")
        # Verify that the keys are valid node names in the FX graph
        names = [node.target for node in graph.fx_graph.nodes]
        for name in sparsity.keys():
            if name not in names:
                raise ValueError(f"Node {name} not found in the graph!")
            _verify_sparsity(sparsity[name])
    else:
        _verify_sparsity(sparsity)

    return sparsity, scope, granularity

def load_weight_prune_config(config: dict, graph):
    """
    Loads and validates the weight pruning config.
    Expects config to have 'method', 'sparsity', 'scope', and 'granularity'.
    """
    sparsity, scope, granularity = load(config, graph)
    method = config.get("method", "random")

    # Validate the method
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
    """
    Loads and validates the activation pruning config.
    Expects config to have 'method', 'sparsity', 'scope', and 'granularity'.
    """
    sparsity, scope, granularity = load(config, graph)
    method = config.get("method", "random")

    # Validate the method
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

# ------------------------------------------------------------------------------
# Example function for distributing global sparsity (not used by MASE by default)
# ------------------------------------------------------------------------------
def get_unfused_distribution(sparsity: float, criterion: callable, name: str):
    """
    Example function that could compute a layer-wise sparsity distribution
    for a given global sparsity target. Not used by default in MASE.
    """
    from chop.models.vision import VISION_MODELS, get_vision_model

    if name is None or name not in VISION_MODELS:
        raise ValueError(f"Expected valid model name. Got {name}")

    model = get_vision_model(name, "cls", {"num_classes": 1000}, pretrained=True)
    tensors = {}

    for mod_name, module in model.named_modules():
        if not isinstance(module, tuple(PRUNEABLE_OPS.values())):
            continue
        # Collect weight tensors
        tensors[mod_name] = {
            "tensor": module.weight.data.clone().flatten(),
            "shape": module.weight.shape,
        }

    concatenated = torch.cat([t["tensor"] for t in tensors.values()])
    # If sparsity=0, we keep everything
    mask = criterion(concatenated, sparsity) if sparsity > 0.0 else torch.ones_like(concatenated, dtype=torch.bool)

    sizes = [t["tensor"].numel() for t in tensors.values()]
    masks = torch.split(mask, sizes)
    layer_sparsities = [1 - m.count_nonzero() / m.numel() for m in masks]
    layer_sparsities = dict(zip(tensors.keys(), layer_sparsities))

    return layer_sparsities

# ------------------------------------------------------------------------------
# Helper function to verify a numeric sparsity value
# ------------------------------------------------------------------------------
def _verify_sparsity(sparsity):
    if not isinstance(sparsity, float):
        raise ValueError("Sparsity must be a float. Got {}".format(type(sparsity)))
    if sparsity < 0 or sparsity > 1:
        raise ValueError("Sparsity must be between 0 and 1. Got {}".format(sparsity))

