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
import torch.nn.utils.prune as prune
import pdb

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
    mask = torch.ones(tensor.size(), dtype=torch.bool, device=tensor.device)
    mask[torch.rand(tensor.size()) < sparsity] = False
    return mask

def handle_large_input_data(flat_tensor: torch.Tensor, sparsity: float):
    #print(f"the input tensor is {flat_tensor.shape} and is divided into small batches")
    batch_unit = int(1e6)
    num_batches = (flat_tensor.size(0) + batch_unit - 1) // batch_unit
    quantiles = []
    for i in range(num_batches-1):
        batch = flat_tensor[i*batch_unit:(i+1)*batch_unit]
        quantiles.append(torch.quantile(batch, sparsity))
    batch = flat_tensor[(num_batches-1)*batch_unit:]
    quantiles.append(torch.quantile(batch, sparsity))  # 其他的排序方法（分开来）
    threshold = torch.mean(torch.tensor(quantiles))
    return threshold

# Default: local

# 1. weights
## 1.1: magnitude-based (element-wise / kernel-wise / channel-wise)
def l1_weight(tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    #pdb.set_trace() # 针对activation
    flat_tensor = tensor.abs().flatten()
    try:
        threshold = torch.quantile(flat_tensor, sparsity)
    except RuntimeError as e:
        threshold = handle_large_input_data(flat_tensor, sparsity)
    mask = (tensor.abs() > threshold).to(torch.bool).to(tensor.device)
    return mask

def l2_weight(tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    l2_norms = tensor.pow(2).sqrt()
    flattened_l2_norms = l2_norms.flatten()
    try:
        threshold = torch.quantile(flattened_l2_norms, sparsity)
    except RuntimeError as e:
        threshold = handle_large_input_data(flattened_l2_norms, sparsity)
    mask = (l2_norms > threshold).to(torch.bool).to(tensor.device)
    return mask

def kernel_l1_weight(tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    l1_norms = tensor.abs().sum(dim=(2,3))
    flattened_l1_norms = l1_norms.flatten()
    try:
        threshold = torch.quantile(flattened_l1_norms, sparsity)  
    except:
        threshold = handle_large_input_data(flattened_l1_norms, sparsity)
    mask = (l1_norms > threshold).to(torch.bool).to(tensor.device)
    return mask

def kernel_l2_weight(tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    l2_norms = torch.norm(tensor, p=2, dim=(2,3))
    flattened_l2_norms = l2_norms.flatten()
    try:
        threshold = torch.quantile(flattened_l2_norms, sparsity)  
    except:
        threshold = handle_large_input_data(flattened_l2_norms, sparsity)
    mask = (l2_norms > threshold).to(torch.bool).to(tensor.device)
    return mask


def channel_l1_weight_multi_layer(tensor: torch.Tensor, next_tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    # multi-layer channel pruning
    l1_norms = tensor.abs().sum(dim=(1,2,3))
    flattened_l1_norms = l1_norms.flatten()
    if next_tensor != None:
        print("next tensor");print(next_tensor.shape)
    if next_tensor != None:
        next_l1_norms = next_tensor.abs().sum(dim=(0,2,3))
        flattened_next_l1_norms = next_l1_norms.flatten()
        final_flattened_l1_norms = 0.5 * flattened_l1_norms + 0.5 * flattened_next_l1_norms   # change it to 0.9 and 0.1
        try:
            threshold = torch.quantile(final_flattened_l1_norms, sparsity)  
        except:
            threshold = handle_large_input_data(final_flattened_l1_norms, sparsity)
    else:
        try:
            threshold = torch.quantile(flattened_l1_norms, sparsity)  
        except:
            threshold = handle_large_input_data(flattened_l1_norms, sparsity)
    mask = (l1_norms > threshold).to(torch.bool).to(tensor.device)
    return mask


def channel_l2_weight_multi_layer(tensor: torch.Tensor, next_tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    # multi-layer channel pruning
    l2_norms = torch.norm(tensor, p=2, dim=(1,2,3))
    flattened_l2_norms = l2_norms.flatten()
    if next_tensor != None:
        print("next tensor");print(next_tensor.shape)
    if next_tensor != None:
        next_l2_norms = next_tensor.abs().sum(dim=(0,2,3))
        flattened_next_l2_norms = next_l2_norms.flatten()
        final_flattened_l2_norms = 0.5 * flattened_l2_norms + 0.5 * flattened_next_l2_norms   # change it to 0.9 and 0.1
        try:
            threshold = torch.quantile(final_flattened_l2_norms, sparsity)  
        except:
            threshold = handle_large_input_data(final_flattened_l2_norms, sparsity)
    else:
        try:
            threshold = torch.quantile(flattened_l2_norms, sparsity) 
        except:
            threshold = handle_large_input_data(flattened_l2_norms, sparsity)
    mask =  (l2_norms > threshold).to(torch.bool).to(tensor.device)
    return mask


def channel_l1_weight(tensor: torch.Tensor, next_tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    # single-layer channel pruning
    l1_norms = tensor.abs().sum(dim=(1,2,3))
    flattened_l1_norms = l1_norms.flatten()
    if next_tensor != None:
        print("next tensor");print(next_tensor.shape)
    if next_tensor != None:
        next_l1_norms = next_tensor.abs().sum(dim=(0,2,3))
        flattened_next_l1_norms = next_l1_norms.flatten()
        final_flattened_l1_norms = flattened_l1_norms
        try:
            threshold = torch.quantile(final_flattened_l1_norms, sparsity)  
        except:
            threshold = handle_large_input_data(final_flattened_l1_norms, sparsity)
    else:
        try:
            threshold = torch.quantile(flattened_l1_norms, sparsity)  
        except:
            threshold = handle_large_input_data(flattened_l1_norms, sparsity)
    mask = (l1_norms > threshold).to(torch.bool).to(tensor.device)
    return mask


def channel_l2_weight(tensor: torch.Tensor, next_tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    # single-layer channel pruning
    l2_norms = torch.norm(tensor, p=2, dim=(1,2,3))
    flattened_l2_norms = l2_norms.flatten()
    if next_tensor != None:
        print("next tensor");print(next_tensor.shape)
    if next_tensor != None:
        next_l2_norms = next_tensor.abs().sum(dim=(0,2,3))
        flattened_next_l2_norms = next_l2_norms.flatten()
        final_flattened_l2_norms = flattened_l2_norms
        try:
            threshold = torch.quantile(final_flattened_l2_norms, sparsity)  
        except:
            threshold = handle_large_input_data(final_flattened_l2_norms, sparsity)
    else:
        try:
            threshold = torch.quantile(flattened_l2_norms, sparsity) 
        except:
            threshold = handle_large_input_data(flattened_l2_norms, sparsity)
    mask =  (l2_norms > threshold).to(torch.bool).to(tensor.device)
    return mask


# 2. activation outputs 
## 2.1: focus on neurons
### 2.1.1: magnitude-based
def activation_l1(tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    flat_tensor = tensor.abs().flatten()
    try:
        threshold = torch.quantile(flat_tensor, sparsity)
    except RuntimeError as e:
        threshold = handle_large_input_data(flat_tensor, sparsity)
    mask = (tensor.abs() > threshold).to(torch.bool).to(tensor.device)
    return mask

def activation_l2(tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    l2_norms = tensor.pow(2).sqrt()
    flattened_l2_norms = l2_norms.flatten()
    try:
        threshold = torch.quantile(flattened_l2_norms, sparsity)
    except RuntimeError as e:
        threshold = handle_large_input_data(flattened_l2_norms, sparsity)
    mask = (tensor.abs() > threshold).to(torch.bool).to(tensor.device)
    return mask


## 2.2: focus on feature maps (magnitude-based)
### 2.2.1 magnitude-based
def activation_l1_feature_map(tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    l1_norms = tensor.abs().sum(dim=(0,2,3)) # e.g: 3*(512*32*32)
    flattened_l1_norms = l1_norms.flatten()
    try:
        threshold = torch.quantile(flattened_l1_norms, sparsity)  
    except:
        threshold = handle_large_input_data(flattened_l1_norms, sparsity)
    mask = (l1_norms > threshold)
    mask = mask.view(1, tensor.shape[1], 1, 1)
    mask = mask.expand(tensor.shape[0], -1, tensor.shape[2], tensor.shape[3])
    mask = mask.to(torch.bool).to(tensor.device)
    return mask

def activation_l2_feature_map(tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    l2_norms = tensor.abs().sum(dim=(0,2,3)) # e.g: 3*(512*32*32)
    flattened_l2_norms = l2_norms.flatten()
    try:
        threshold = torch.quantile(flattened_l2_norms, sparsity)  
    except:
        threshold = handle_large_input_data(flattened_l2_norms, sparsity)
    mask = (l2_norms > threshold)
    mask = mask.view(1, tensor.shape[1], 1, 1)
    mask = mask.expand(tensor.shape[0], -1, tensor.shape[2], tensor.shape[3])
    mask = mask.to(torch.bool).to(tensor.device)
    return mask


### 2.2.2 similarity-based
def channel_similarity_feature_map(tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    # tensor: n(i)*n(i-1)*k*k
    flat_tensor = tensor.reshape(tensor.shape[1], -1)  # e.g: 3*(512*32*32)
    # normalization
    norms = torch.norm(flat_tensor, dim=1, keepdim=True)
    normalized_tensor = flat_tensor / norms
    cosine_similarity_matrix = torch.mm(normalized_tensor, normalized_tensor.t())  # cosine similarity
    # Get the upper triangular part of the cosine similarity matrix, excluding the diagonal
    triu_indices = torch.triu_indices(cosine_similarity_matrix.size(0), cosine_similarity_matrix.size(1), offset=1)
    similarities = cosine_similarity_matrix[triu_indices[0], triu_indices[1]]
    # Pair each similarity score with its index
    pair_indices = list(zip(triu_indices[0].tolist(), triu_indices[1].tolist()))
    indexed_similarities = list(zip(similarities.tolist(), pair_indices))
    # Sort by similarity
    indexed_similarities.sort(reverse=True)
    # Determine the number of pairs to remove based on sparsity
    num_pairs_to_remove = int(sparsity * len(indexed_similarities))
    # Select indices to remove
    indices_to_remove = set()
    for _, (i, j) in indexed_similarities[:num_pairs_to_remove]:
        # Arbitrarily choose one of the pair to remove, here we choose the second
        indices_to_remove.add(j)
    indices_to_remove = sorted(indices_to_remove)
    # Create a mask
    mask = torch.ones(cosine_similarity_matrix.size(0), dtype=torch.bool)
    mask[list(indices_to_remove)] = False
    mask = mask.view(1, tensor.shape[1], 1, 1)
    mask = mask.expand(tensor.shape[0], -1, tensor.shape[2], tensor.shape[3])
    mask = mask.to(tensor.device)
    return mask


## EXTRA: global
## We haven't implemented too many as they are basically the same as local methods.
## It's relatively easy to change from each method in local to global.

def global_weight_l1(tensor: torch.Tensor, info: dict, sparsity: float):
    tensors = [v["weight_value"] for _, v in info.items() if v is not None]
    flattened_tensors = [tensor.abs().flatten() for tensor in tensors]
    flattened_tensors = torch.cat(flattened_tensors)
    try:
        threshold = torch.quantile(flattened_tensors, sparsity)
    except RuntimeError as e:
        threshold = handle_large_input_data(flattened_tensors, sparsity)
    mask = (tensor.abs() > threshold).to(torch.bool).to(tensor.device)
    return mask

def global_weight_l1(tensor: torch.Tensor, info: dict, sparsity: float):
    tensors = [v["weight_value"] for _, v in info.items() if v is not None]
    flattened_tensors = [tensor.abs().flatten() for tensor in tensors]
    flattened_tensors = torch.cat(flattened_tensors)
    try:
        threshold = torch.quantile(flattened_tensors, sparsity)
    except RuntimeError as e:
        threshold = handle_large_input_data(flattened_tensors, sparsity)
    mask = (tensor.abs() > threshold).to(torch.bool).to(tensor.device)
    return mask

def global_activation_l1(tensor: torch.Tensor, info: dict, sparsity: float):
    tensors = [v["activation_value"] for _, v in info.items() if v is not None]
    flattened_tensors = [tensor.abs().sum(dim=(1,2,3)).flatten() for tensor in tensors]
    flattened_tensors = torch.cat(flattened_tensors)
    #flattened_tensors = l1_norms.flatten()
    try:
        threshold = torch.quantile(flattened_tensors, sparsity)
    except RuntimeError as e:
        threshold = handle_large_input_data(flattened_tensors, sparsity)
    mask = (tensor.abs() > threshold).to(torch.bool).to(tensor.device)
    return mask


weight_criteria_map = {
    "local": {
        "elementwise": {
            "random": random, "l1-norm": l1_weight, "l2-norm": l2_weight
        }, 
        "kernelwise": {
            "l1-norm": kernel_l1_weight, "l2-norm": kernel_l2_weight
        }, 
        "channelwise": {
            "l1-norm-multi": channel_l1_weight_multi_layer, "l2-norm-multi": channel_l2_weight_multi_layer,
            "l1-norm-single": channel_l1_weight,  "l2-norm-single": channel_l2_weight
        },
    },
    "global": {"elementwise": {"l1-norm": global_weight_l1}},
}

activation_criteria_map = {
    "local": {
        "elementwise": {
            "l1-norm": activation_l1, "l2-norm": activation_l2
        },
        "channelwise": {
            "feature-map-l1-norm": activation_l1_feature_map, "feature-map-l2-norm": activation_l2_feature_map, "feature-map-similarity": channel_similarity_feature_map
        }
    },
    "global": {
        "elementwise": {"l1-norm": global_activation_l1}
    },
}
