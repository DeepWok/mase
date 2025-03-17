from chop.passes.graph.transforms.pruning.hwpq_pruning import hwpq_pruning_only, PruningParameterization

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

def global_weight_l1(tensor: torch.Tensor, info: dict, sparsity: float, node_name: str, bins: int = 2048,
                     _state={"did_fail": False}) -> torch.Tensor:
    r"""
    Attempt a "full flatten + quantile" global L1 pruning. If we hit a RuntimeError
    (OOM or "tensor too large" on GPU/MPS), fall back to a histogram-based approach.

    :param tensor:   The weight tensor of the current module to prune.
    :param info:     The full dictionary (across all modules) of metadata.
    :param sparsity: The desired global sparsity (fraction to prune).
                     E.g. 0.2 means keep the top 80% of weights by magnitude.
    :param bins:     Number of bins to use in fallback histogram approach (2048 by default).
    :return:         A boolean mask (same shape as `tensor`) that is True for weights to keep
                     and False for weights to prune.
    """
    if not _state["did_fail"]:
        try:
            arrays = []
            for _, v in info.items():
                if v is not None:
                    arrays.append(v["weight_value"].abs().flatten())
            big_flat = torch.cat(arrays, dim=0)
            threshold = torch.quantile(big_flat, sparsity)
            mask = (tensor.abs() > threshold).bool().to(tensor.device)
            return mask

        except RuntimeError as e:
            print("[DEBUG] Global L1 full-quantile approach failed. Falling back to histogram.")
            print("[DEBUG] Caught RuntimeError:", str(e))
            _state["did_fail"] = True

    global_max = 0.0
    total_elements = 0
    for _, v in info.items():
        if v is not None:
            w = v["weight_value"]
            current_max = w.abs().max().item()
            if current_max > global_max:
                global_max = current_max
            total_elements += w.numel()

    if global_max == 0.0:
        threshold = 0.0
    else:
        bin_edges = torch.linspace(0, global_max, steps=bins+1)  # on CPU now
        global_hist = torch.zeros(bins)
        for _, v in info.items():
            if v is not None:
                w_vals = v["weight_value"].abs().flatten().cpu()
                h = torch.histc(w_vals, bins=bins, min=0, max=global_max)
                global_hist += h

        target_count = sparsity * total_elements
        cumulative = torch.cumsum(global_hist, dim=0)
        bin_idx_tensor = (cumulative >= target_count).nonzero(as_tuple=False)
        if len(bin_idx_tensor) == 0:
            threshold = global_max
        else:
            idx = bin_idx_tensor[0].item()
            threshold = (bin_edges[idx] + bin_edges[idx + 1]) / 2

    mask = (tensor.abs() > threshold).bool().to(tensor.device)
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

def global_weight_movement(tensor: torch.Tensor, info: dict, sparsity: float, node_name: str, bins: int = 2048,
                           _state={"did_fail": False}) -> torch.Tensor:
    """
    Global movement pruning ranking function with fallback method for memory constrained applications.
    Aggregates movement scores from the info dictionary over all relevant weight tensors to compute a global threshold.
    """

    def _apply_movement_threshold(tensor: torch.Tensor, info: dict, threshold: float, node_name: str) -> torch.Tensor:
        if node_name not in info:
            raise ValueError(f"Node {node_name} not found in info dictionary.")
        
        current_node_meta = info[node_name]
        if "weight_stats" not in current_node_meta or "movement" not in current_node_meta["weight_stats"]:
            raise ValueError(
                f"Current node '{node_name}' has no movement data, "
                "but is being pruned with global movement."
            )
        
        current_movement = current_node_meta["weight_stats"]["movement"]
        mask = (current_movement.abs() > threshold).bool().to(tensor.device)
        return mask

    all_movements = []
    for _, v in info.items():
        if v is not None and "weight_stats" in v:
            ws = v["weight_stats"]
            if "movement" in ws:
                all_movements.append(ws["movement"])

    if not all_movements:
        raise ValueError("No global movement data found: no 'weight_stats'/'movement' in info.")
    
    if not _state["did_fail"]:
        try:
            big_flat = torch.cat([m.abs().flatten().to(tensor.device) for m in all_movements], dim=0)
            threshold = torch.quantile(big_flat, sparsity)

            return _apply_movement_threshold(tensor, info, threshold, node_name)

        except RuntimeError as e:
            print("[DEBUG] Global movement flatten+quantile approach failed. Falling back to histogram.")
            print("[DEBUG] Caught RuntimeError:", str(e))
            _state["did_fail"] = True

    global_max = 0.0
    total_elements = 0
    for m in all_movements:
        mx = m.abs().max().item()
        if mx > global_max:
            global_max = mx
        total_elements += m.numel()

    if global_max == 0.0:
        threshold = 0.0
    else:
        bin_edges = torch.linspace(0, global_max, steps=bins + 1)  # CPU by default
        global_hist = torch.zeros(bins)

        for m in all_movements:
            m_cpu = m.abs().flatten().cpu()
            h = torch.histc(m_cpu, bins=bins, min=0, max=global_max)
            global_hist += h

        target_count = sparsity * total_elements
        cumulative = torch.cumsum(global_hist, dim=0)
        idx_tensor = (cumulative >= target_count).nonzero(as_tuple=False)
        if idx_tensor.numel() == 0:
            threshold = global_max
        else:
            idx = idx_tensor[0].item()
            threshold = (bin_edges[idx] + bin_edges[idx + 1]) / 2

    return _apply_movement_threshold(tensor, info, threshold, node_name)

# Not Implemented Yet, Needs Changing
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

# Not Implemented Yet, Needs Changing
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


weight_criteria_map = {
    "local": {
        "elementwise": {
            "random": random,
            "l1-norm": l1,
            "movement": movement,
            "hwpq": hwpq_pruning,  # Add the HWPQ method here
        }
    },
    "global": {
        "elementwise": {
            "random": random,
            "l1-norm": global_weight_l1,
            "movement": global_weight_movement, 
            # We're not implementing global HWPQ for now
        }
    },
}

activation_criteria_map = {
    "local": {
        "elementwise": {
            "random": random,
            "l1-norm": l1,
            "movement": activation_movement,  # Yet to be implemented, activation movement pruning (local)
        }
    },
    "global": {
        "elementwise": {
            "random": random,
            "l1-norm": global_activation_l1,
            "movement": global_activation_movement,  # Yet to be implemented, activation movement pruning (global)
        }
    },
}

