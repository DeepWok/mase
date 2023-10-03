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


# Set sparsity percentage of values in the mask to False (i.e. 0) randomly
# Pre: sparsity is not 0.0
def random_rank(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    mask = torch.ones(tensor.size(), dtype=torch.bool)
    mask[torch.rand(tensor.size()) < sparsity] = False
    return mask


# Set sparsity percentage of values in the mask to False (i.e. 0) randomly
# Pre: sparsity is not 0.0
def neurons_random_rank(
    tensor: torch.Tensor, sparsity: float, layer_type: str
) -> torch.Tensor:
    mask = torch.ones(tensor.size(), dtype=torch.bool)
    mask = mask.reshape(tensor.shape[0], -1)
    if layer_type == "Linear":
        for i in range(tensor.shape[0]):
            mask[i, torch.rand(tensor.shape[1]) < sparsity] = False
            # x = torch.randperm(tensor.shape[1])[:self.fan_in]
    elif layer_type == "Conv2d":
        for i in range(tensor.shape[0]):
            mask[i, torch.rand(tensor.shape[1]) < sparsity] = False
            # mask[i, torch.rand(tensor.shape[1]*tensor.shape[2]*tensor.shape[3]) < sparsity] = False
            # x = torch.randperm(tensor.shape[1]*tensor.shape[2]*tensor.shape[3])[:self.fan_in]
    mask.reshape(*tensor.shape)
    return mask


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


# Use the L1 norm of values in the tensor to rank them and return a mask where values
# lower than the threshold are set to False (i.e. 0).
# Pre: sparsity is not 0.0
def l1_rank(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    threshold = torch.quantile(tensor.abs().flatten(), sparsity)
    mask = (tensor.abs() > threshold).to(torch.bool)
    return mask


RANK_CRITERIA = {
    "random": random_rank,
    "l1": l1_rank,
    "neuron_wise_random": neurons_random_rank,
    "neuron_wise_fan_in_random": neurons_random_fan_in,
}
