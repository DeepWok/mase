# Ranking criteria for pruning weights
# NOTE: For now, we assume that all ranking functions take in two positional arguments
# (a tensor along with a sparsity target) and return an appropriately computed mask. All
# other arguments are passed in via keywords.
# --------------------------------------------------------------------------------------
# We assume that the sparsity isn't 0.0 as (1) the mask (all ones) is known beforehand
# and (2) the criterion function may not generate a valid mask. The L1 ranking function
# uses the quantile function, which, when the sparisty is 0, uses the lowest value as
# the threshold. So, at least one value in the mask is always set to False.

import numpy as np
import torch

"""
These implemntations are for the pruning functional we assume info always have the following form:
    an info entry = {
        'module_type': 'conv2d',
        'value': w_value,
        'stats': w_stats,
        'shape': w_shape,
        ...
    }
"""


def random(tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    """set sparsity percentage of values
    in the mask to False (i.e. 0) randomly
    Pre: sparsity is not 0.0

    :param tensor: input tensor
    :type tensor: torch.Tensor
    :param sparsity: sparsity level, this suppose to be a ratio in between 0.0 and 1.0
    :type sparsity: float
    :return: a random sparsity mask generated based on the sparsity value
    :rtype: torch.Tensor
    """
    mask = torch.ones(tensor.size(), dtype=torch.bool, device=tensor.device)
    mask[torch.rand(tensor.size()) < sparsity] = False
    return mask


def l1(tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    """Use the L1 norm of values in the tensor
    to rank them and return a mask where values
    lower than the threshold are set to False (i.e. 0).
    Pre: sparsity is not 0.0

    :param tensor: input tensor
    :type tensor: torch.Tensor
    :param sparsity: sparsity level, this suppose to be a ratio between 0.0 and 1.0
    :type sparsity: float
    :return: a sparsity mask
    :rtype: torch.Tensor
    """
    with torch.no_grad():
        flattened = tensor.abs().cpu().numpy().flatten()
        
        threshold = np.quantile(flattened, sparsity)
    mask = (tensor.abs() > threshold).to(torch.bool).to(tensor.device)
    return mask


def channel_l1(tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    """Use the L1 norm of values in the tensor
    to rank them and return a mask where values
    lower than the threshold are set to False (i.e. 0).
    Pre: sparsity is not 0.0

    :param tensor: input tensor
    :type tensor: torch.Tensor
    :param sparsity: sparsity level, this suppose to be a ratio between 0.0 and 1.0
    :type sparsity: float
    :return: a sparsity mask
    :rtype: torch.Tensor
    """
    
    if (tensor.dim() == 2):
        # It's a linear layer tensor; create a mask of True values
        mask = torch.ones_like(tensor, dtype=torch.bool).to(tensor.device)
        return mask
    

    elif (tensor.dim() == 3):
        
        filter_weights_sum = tensor.abs().sum(dim=[1, 2])

        threshold = torch.quantile(filter_weights_sum, sparsity)

        # Create a mask: 1 for filters with sum above the threshold, 0 for those below
        mask = ((filter_weights_sum > threshold).float().unsqueeze(1).unsqueeze(2).expand_as(tensor)).to(torch.bool).to(tensor.device)
        # Adjust mask shape to match the weight tensor shape for 1D convolution

        return mask

    elif (tensor.dim() == 4):
        filter_weights_sum = tensor.abs().sum(dim=[1, 2, 3])  # Sum across in_channels, H, and W dimensions

        # Find the pruning threshold, below which filters will be pruned
        
        threshold = torch.quantile(filter_weights_sum, sparsity)
        # Keep all filters if sparsity is 0

        # Create a mask: 1 for filters with sum above the threshold, 0 for those below
        mask = ((filter_weights_sum > threshold).float().unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(tensor)).to(torch.bool).to(tensor.device)
        # Adjust mask shape to match the weight tensor shape  
        return mask
    else:
        "Error"
        return

def global_weight_l1(tensor: torch.Tensor, info: dict, sparsity: float):
    masks = [v["weight_masks"] for _, v in info.items() if v is not None]
    masks = [(torch.all(torch.stack(list(m.values())), dim=0) if m is not None else 1) for m in masks]
    tensors = [v["weight_value"] for _, v in info.items() if v is not None]
    tensors_to_prune = [tensor * mask for (mask, tensor) in zip(masks, tensors)]
    
    with torch.no_grad():
        flattened_tensors = [t.abs().cpu().numpy().flatten() for t in tensors_to_prune]
        flattened_tensor = np.concatenate(flattened_tensors, axis=0)
        
        threshold = np.quantile(flattened_tensor, sparsity)

    mask = (tensor.abs() > threshold).to(torch.bool).to(tensor.device)
    return mask


def global_activation_l1(tensor: torch.Tensor, info: dict, sparsity: float):
    tensors = [v["activation_value"] for _, v in info.items() if v is not None]
    threshold = 0
    with torch.no_grad():
        flattened_tensors = [t.abs().flatten() for t in tensors]
        threshold = np.quantile(torch.cat(flattened_tensors, dim=0).detach().numpy(), sparsity)
    mask = (tensor.abs() > threshold).to(torch.bool).to(tensor.device)
    return mask


"""this is from the old pruning old, leaving it here in case we need them later"""


def neurons_random_rank(
    tensor: torch.Tensor, sparsity: float, layer_type: str
) -> torch.Tensor:
    """set sparsity percentage of values
    in the mask to False (i.e. 0) randomly
    pre: sparsity is not 0.0

    :param tensor: input tensor
    :type tensor: torch.Tensor
    :param sparsity: sparsity level, this suppose to be a ratio between 0.0 and 1.0
    :type sparsity: float
    :param layer_type: layer type can be in ["Linear", "Conv2d"]
    :type layer_type: str
    :return: a sparsity mask
    :rtype: torch.Tensor
    """
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


"""this is from the old pruning old, leaving it here in case we need them later"""


# Pruning each neurons connections by specified fan_in randomly
def neurons_random_fan_in(
    tensor: torch.Tensor, sparsity: float, layer_type: str, fan_in: int
) -> torch.Tensor:
    if fan_in == None:
        raise ValueError("fan_in is not been specified")
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


weight_criteria_map = {
    "local": {"elementwise": {"random": random, "l1-norm": l1},"channel":{"l1-norm": channel_l1}},
    "global": {"elementwise": {"random": random, "l1-norm": global_weight_l1}},
}

activation_criteria_map = {
    "local": {"elementwise": {"random": random, "l1-norm": l1}},
    "global": {"elementwise": {"random": random, "l1-norm": global_activation_l1}},
}
