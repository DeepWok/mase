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


def random_rank(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
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
    mask = torch.ones(tensor.size(), dtype=torch.bool)
    mask[torch.rand(tensor.size()) < sparsity] = False
    return mask


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
        raise ValueError(f"{layer_type}" is not supported)
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


def l1_rank(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
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
    threshold = torch.quantile(tensor.abs().flatten(), sparsity)
    mask = (tensor.abs() > threshold).to(torch.bool)
    return mask


RANK_CRITERIA = {
    "random": random_rank,
    "l1": l1_rank,
    "neuron_wise_random": neurons_random_rank,
    "neuron_wise_fan_in_random": neurons_random_fan_in,
}
