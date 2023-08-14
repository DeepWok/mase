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


# Use the L1 norm of values in the tensor to rank them and return a mask where values
# lower than the threshold are set to False (i.e. 0).
# Pre: sparsity is not 0.0
def l1_rank(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    threshold = torch.quantile(tensor.abs().flatten(), sparsity)
    mask = (tensor.abs() > threshold).to(torch.bool)
    return mask


RANK_CRITERIA = {"random": random_rank, "l1": l1_rank}
