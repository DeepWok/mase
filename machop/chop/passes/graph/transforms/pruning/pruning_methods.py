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
These implemntations are for the pruning functional we assume info always have the following form:
    an info entry = {
        'module_type': 'conv2d',
        'value': w_value,
        'stats': w_stats,
        'shape': w_shape,
        ...
    }
"""

# random pruning

def random(tensor: torch.Tensor, info: dict, sparsity: float, name:str) -> torch.Tensor:
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

# -----------------------------------------------------------------------------------------------
# tensor-wise pruning

def tensor_element_l1(tensor: torch.Tensor, info: dict, sparsity: float, name: str) -> torch.Tensor:
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
    
    threshold = torch.quantile(tensor.abs().float().flatten(), sparsity)
    mask = (tensor.abs() > threshold).to(torch.bool).to(tensor.device)
    return mask

def tensor_element_l2(tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    threshold = torch.quantile(tensor.float().square().flatten(), sparsity)
    mask = (tensor.square() > threshold).to(torch.bool).to(tensor.device)
    return mask

'''
Channel-wise pruning
Take each channel of the input tensor, calculate the specific threshold for the channel,
and set the weights to False if the channel weight is under the threshold.
'''
def tensor_channel_l1(tensor: torch.Tensor, info: dict, sparsity: float, name: str):
    total_mask = []
    channel_weights = []
    device = 'cuda'

    for channel in range(tensor.size(0)):
        weights = tensor[channel]
        channel_weight = torch.norm(weights, p=1)
        channel_weights.append(channel_weight)
    threshold = torch.quantile(torch.tensor(channel_weight), sparsity)
    for channel in range(tensor.size(0)):
        if channel_weights[channel] > threshold:
            mask = tensor[channel].to(torch.bool).to(device)
        else:
            mask = torch.full_like(tensor[channel],False,dtype=torch.bool).to(device)
        total_mask.append(mask)
    return torch.stack(total_mask)

def tensor_channel_l2(tensor: torch.Tensor, info: dict, sparsity: float, name: str):
    total_mask = []
    channel_weights = []
    device = 'cuda'

    for channel in range(tensor.size(0)):
        weights = tensor[channel]
        channel_weight = torch.sqrt(torch.sum(torch.norm(weights, p=1)**2))
        channel_weights.append(channel_weight)
    threshold = torch.quantile(torch.tensor(channel_weight), sparsity)
    for channel in range(tensor.size(0)):
        if channel_weights[channel] > threshold:
            mask = tensor[channel].to(torch.bool).to(device)
        else:
            mask = torch.full_like(tensor[channel],False,dtype=torch.bool).to(device)
        total_mask.append(mask)
    return torch.stack(total_mask)

# -----------------------------------------------------------------------------------------------
# Layer-wise pruning
def layer_element_l1(tensor: torch.Tensor, info: dict, sparsity: dict, name:str):
    layer_sparsity = sparsity[name]

    threshold = torch.quantile(tensor.abs().flatten(), layer_sparsity)
    mask = (tensor.abs() > threshold).to(torch.bool).to(tensor.device)
    return mask


def layer_element_l2(tensor: torch.Tensor, info: dict, sparsity: dict, name:str):
    layer_sparsity = sparsity[name]

    threshold = torch.quantile(tensor.square().flatten(), layer_sparsity)
    mask = (tensor.square() > threshold).to(torch.bool).to(tensor.device)
    return mask


def layer_channel_l1(tensor: torch.Tensor, info: dict, sparsity: float, name: str):
    total_mask = []
    channel_weights = []
    device = 'cuda'
    layer_sparsity = sparsity[name]

    for channel in range(tensor.size(0)):
        weights = tensor[channel]
        channel_weight = torch.norm(weights, p=1)
        channel_weights.append(channel_weight)
    threshold = torch.quantile(torch.tensor(channel_weight), layer_sparsity)
    for channel in range(tensor.size(0)):
        if channel_weights[channel] > threshold:
            mask = tensor[channel].to(torch.bool).to(device)
        else:
            mask = torch.full_like(tensor[channel],False,dtype=torch.bool).to(device)
        total_mask.append(mask)
    return torch.stack(total_mask)

def layer_channel_l2(tensor: torch.Tensor, info: dict, sparsity: float, name: str):
    total_mask = []
    channel_weights = []
    device = 'cuda'
    layer_sparsity = sparsity[name]

    for channel in range(tensor.size(0)):
        weights = tensor[channel]
        channel_weight = torch.sqrt(torch.sum(torch.norm(weights, p=1)**2))
        channel_weights.append(channel_weight)
    threshold = torch.quantile(torch.tensor(channel_weight), layer_sparsity)
    for channel in range(tensor.size(0)):
        if channel_weights[channel] > threshold:
            mask = tensor[channel].to(torch.bool).to(device)
        else:
            mask = torch.full_like(tensor[channel],False,dtype=torch.bool).to(device)
        total_mask.append(mask)
    return torch.stack(total_mask)


# -----------------------------------------------------------------------------------------------
# Global pruning


def global_weight_l1(tensor: torch.Tensor, info: dict, sparsity: float, name: str):
    tensors = [v["weight_value"] for _, v in info.items() if v is not None]
    flattened_tensors = [t.abs().flatten() for t in tensors]
    threshold = torch.quantile(torch.cat(flattened_tensors, dim=0), sparsity)
    mask = (tensor.abs() > threshold).to(torch.bool).to(tensor.device)
    return mask


def global_activation_l1(tensor: torch.Tensor, info: dict, sparsity: float, name: str):
    tensors = [v["activation_value"] for _, v in info.items() if v is not None]
    flattened_tensors = [t.abs().flatten() for t in tensors]

    # change the device of tensors been stored
    device = 'cuda'
    flattened_tensors = [tensor.to(device) for tensor in flattened_tensors]
    # Since the torch.quantile() cannot take entire flatten tensor (too large)
    # we select a sub-sample from the tensor and calculate the threshold
    concatenated_tensor =torch.cat(flattened_tensors, dim=0)
    num_elements = concatenated_tensor.numel()
    sampled_indices = torch.randint(num_elements, (1600000,), device=device)
    sampled_tensor = torch.index_select(concatenated_tensor.view(-1), 0, sampled_indices)

    threshold = torch.quantile(sampled_tensor, sparsity)
    mask = (tensor.abs() > threshold).to(torch.bool).to(device)
    return mask


def global_weight_l2(tensor: torch.Tensor, info: dict, sparsity: float, name: str):
    tensors = [v["weight_value"] for _, v in info.items() if v is not None]
    flattened_tensors = [t.square().flatten() for t in tensors]
    threshold = torch.quantile(torch.cat(flattened_tensors, dim=0), sparsity)
    mask = (tensor.square() > threshold).to(torch.bool).to(tensor.device)
    return mask


def global_activation_l2(tensor: torch.Tensor, info: dict, sparsity: float, name: str):
    tensors = [v["activation_value"] for _, v in info.items() if v is not None]
    flattened_tensors = [t.square().flatten() for t in tensors]

    # change the device of tensors been stored
    device = 'cuda'
    flattened_tensors = [tensor.to(device) for tensor in flattened_tensors]

    # Since the torch.quantile() cannot take entire flatten tensor (too large)
    # we select a sub-sample from the tensor and calculate the threshold
    concatenated_tensor =torch.cat(flattened_tensors, dim=0)
    num_elements = concatenated_tensor.numel()
    sampled_indices = torch.randint(num_elements, (1600000,), device=device)
    sampled_tensor = torch.index_select(concatenated_tensor.view(-1), 0, sampled_indices)

    threshold = torch.quantile(sampled_tensor, sparsity)
    mask = (tensor.square() > threshold).to(torch.bool).to(device)
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
    "tensor": {"elementwise": {"random": random, "l1-norm": tensor_element_l1, "l2-norm": tensor_element_l2},
              "channelwise":{"random": random, "l1-norm": tensor_channel_l1, "l2-norm": tensor_channel_l2}},
    "layer": {"elementwise": {"random": random, "l1-norm": layer_element_l1, "l2-norm": layer_element_l2},
              "channelwise":{"random": random, "l1-norm": layer_channel_l1, "l2-norm": layer_channel_l2}},
    "global": {"elementwise": {"random": random, "l1-norm": global_weight_l1, "l2-norm": global_weight_l2}},
}

activation_criteria_map = {
    "tensor": {"elementwise": {"random": random, "l1-norm": tensor_element_l1, "l2-norm": tensor_element_l2},
              "channelwise":{"random": random, "l1-norm": tensor_channel_l1, "l2-norm": tensor_channel_l2}},
    "layer": {"elementwise": {"random": random, "l1-norm": layer_element_l1, "l2-norm": layer_element_l2},
              "channelwise":{"random": random, "l1-norm": layer_channel_l1, "l2-norm": layer_channel_l2}},
    "global": {"elementwise": {"random": random, "l1-norm": global_activation_l1, "l2-norm": global_activation_l2}},
}
