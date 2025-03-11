
# Ranking criteria for pruning weights
# NOTE: For now, we assume that all ranking functions take in two positional arguments
# (a tensor along with a sparsity target) and return an appropriately computed mask. All
# other arguments are passed in via keywords.
# --------------------------------------------------------------------------------------
# We assume that the sparsity isn't 0.0 as (1) the mask (all ones) is known beforehand
# and (2) the criterion function may not generate a valid mask. The L1 ranking function
# uses the quantile function, which, when the sparisty is 0, uses the lowest value as
# the threshold. So, at least one value in the mask is always set to False.

import torch

"""
These implementations are for the pruning functional we assume info always have the following form:
    an info entry = {
        'module_type': 'conv2d',
        'value': w_value,
        'stats': w_stats,   # here we assume w_stats may contain a "movement" entry, among others
        'shape': w_shape,
        ...
    }
"""

def random(tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    """set sparsity percentage of values in the mask to False (i.e. 0) randomly.
    Pre: sparsity is not 0.0
    """
    mask = torch.ones(tensor.size(), dtype=torch.bool, device=tensor.device)
    mask[torch.rand(tensor.size()) < sparsity] = False
    return mask

def l1(tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    """Use the L1 norm of values in the tensor to rank them and return a mask where values
    lower than the threshold are set to False (i.e. 0).
    Pre: sparsity is not 0.0
    """
    threshold = torch.quantile(tensor.abs().flatten(), sparsity)
    mask = (tensor.abs() > threshold).to(torch.bool).to(tensor.device)
    return mask

def global_weight_l1(tensor: torch.Tensor, info: dict, sparsity: float):
    tensors = [v["weight_value"] for _, v in info.items() if v is not None]
    flattened_tensors = [t.abs().flatten() for t in tensors]
    threshold = torch.quantile(torch.cat(flattened_tensors, dim=0), sparsity)
    mask = (tensor.abs() > threshold).to(torch.bool).to(tensor.device)
    return mask

def global_activation_l1(tensor: torch.Tensor, info: dict, sparsity: float):
    tensors = [v["activation_value"] for _, v in info.items() if v is not None]
    flattened_tensors = [t.abs().flatten() for t in tensors]
    threshold = torch.quantile(torch.cat(flattened_tensors, dim=0), sparsity)
    mask = (tensor.abs() > threshold).to(torch.bool).to(tensor.device)
    return mask

"""this is from the old pruning, leaving it here in case we need them later"""
def neurons_random_rank(tensor: torch.Tensor, sparsity: float, layer_type: str) -> torch.Tensor:
    mask = torch.ones(tensor.size(), dtype=torch.bool)
    mask = mask.reshape(tensor.shape[0], -1)
    if layer_type == "Linear":
        for i in range(tensor.shape[0]):
            mask[i, torch.rand(tensor.shape[1]) < sparsity] = False
    elif layer_type == "Conv2d":
        for i in range(tensor.shape[0]):
            mask[i, torch.rand(tensor.shape[1]) < sparsity] = False
    else:
        raise ValueError(f"{layer_type} is not supported")
    mask.reshape(*tensor.shape)
    return mask

"""this is from the old pruning, leaving it here in case we need them later"""
def neurons_random_fan_in(tensor: torch.Tensor, sparsity: float, layer_type: str, fan_in: int) -> torch.Tensor:
    if fan_in is None:
        raise ValueError("fan_in has not been specified")
    mask = torch.zeros(tensor.size(), dtype=torch.bool)
    mask = mask.reshape(tensor.shape[0], -1)
    if layer_type == "Linear":
        for i in range(tensor.shape[0]):
            mask[i, torch.randperm(tensor.shape[1])[:fan_in]] = True
    elif layer_type == "Conv2d":
        for i in range(tensor.shape[0]):
            mask[i, torch.randperm(tensor.shape[1])[:fan_in]] = True
    mask.reshape(*tensor.shape)
    return mask

# -------------------------------
# New: Movement-based pruning functions
# -------------------------------

def movement(tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    """
    Movement pruning ranking function for a given tensor.
    Assumes that info["stats"] contains a "movement" entry representing accumulated movement scores.
    We prune (set to False) the weights with the lowest movement (in absolute terms).
    """
    if "movement" not in info.get("stats", {}):
        raise ValueError("Movement information not found in info stats.")
    movement_scores = info["stats"]["movement"]
    threshold = torch.quantile(movement_scores.abs().flatten(), sparsity)
    mask = (movement_scores.abs() > threshold).to(torch.bool).to(tensor.device)
    return mask

def global_weight_movement(tensor: torch.Tensor, info: dict, sparsity: float):
    """
    Global movement pruning ranking function.
    Aggregates movement scores from the info dictionary over all relevant weight tensors to compute a global threshold.
    """
    movements = [v["stats"]["movement"] for _, v in info.items() if v is not None and "movement" in v.get("stats", {})]
    if len(movements) == 0:
        raise ValueError("No movement information found in info stats.")
    flattened_movements = [m.abs().flatten() for m in movements]
    threshold = torch.quantile(torch.cat(flattened_movements, dim=0), sparsity)
    # For the current tensor, use its own movement scores:
    current_movement = info["stats"]["movement"]
    mask = (current_movement.abs() > threshold).to(torch.bool).to(tensor.device)
    return mask

def activation_movement(tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    """
    Movement-based pruning ranking function for activations.
    Assumes that info["stats"] contains a "movement" entry for activations.
    """
    if "movement" not in info.get("stats", {}):
        raise ValueError("Movement information not found in info stats.")
    movement_scores = info["stats"]["movement"]
    threshold = torch.quantile(movement_scores.abs().flatten(), sparsity)
    mask = (movement_scores.abs() > threshold).to(torch.bool).to(tensor.device)
    return mask

def global_activation_movement(tensor: torch.Tensor, info: dict, sparsity: float):
    """
    Global movement-based pruning for activations.
    """
    movements = [v["stats"]["movement"] for _, v in info.items() if v is not None and "movement" in v.get("stats", {})]
    if len(movements) == 0:
        raise ValueError("No movement information found in info stats.")
    flattened_movements = [m.abs().flatten() for m in movements]
    threshold = torch.quantile(torch.cat(flattened_movements, dim=0), sparsity)
    current_movement = info["stats"]["movement"]
    mask = (current_movement.abs() > threshold).to(torch.bool).to(tensor.device)
    return mask

# -------------------------------
# Update mapping dictionaries to include movement-based methods
# -------------------------------

weight_criteria_map = {
    "local": {
        "elementwise": {
            "random": random,
            "l1-norm": l1,
            "movement": movement,  # Added movement pruning for weights (local)
        }
    },
    "global": {
        "elementwise": {
            "random": random,
            "l1-norm": global_weight_l1,
            "movement": global_weight_movement,  # Added movement pruning for weights (global)
        }
    },
}

activation_criteria_map = {
    "local": {
        "elementwise": {
            "random": random,
            "l1-norm": l1,
            "movement": activation_movement,  # Added movement pruning for activations (local)
        }
    },
    "global": {
        "elementwise": {
            "random": random,
            "l1-norm": global_activation_l1,
            "movement": global_activation_movement,  # Added movement pruning for activations (global)
        }
    },
}

