import torch
import torch.nn.functional as F

def _get_similarity(tensor_raw, tensor_sim, metric=None):
    if metric == "cosine":
        similarity = F.cosine_similarity(tensor_raw, tensor_sim, dim=-1)
    elif metric == "pearson":
        similarity = F.cosine_similarity(
            tensor_raw - torch.mean(tensor_raw, dim=-1, keepdim=True),
            tensor_sim - torch.mean(tensor_sim, dim=-1, keepdim=True),
            dim=-1,
        )
    else:
        if metric == "L1_norm":
            similarity = -torch.abs(tensor_raw - tensor_sim)
        elif metric == "L2_norm":
            similarity = -((tensor_raw - tensor_sim) ** 2)
        elif metric == "linear_weighted_L2_norm":
            similarity = -tensor_raw.abs() * (tensor_raw - tensor_sim) ** 2
        elif metric == "square_weighted_L2_norm":
            similarity = -((tensor_raw * (tensor_raw - tensor_sim)) ** 2)
        else:
            raise NotImplementedError(f"metric {metric} not implemented!")
        similarity = torch.mean(similarity, dim=-1)
    return similarity

def reshape_to_block(tensor, t1, t0, p1, p0):
    return tensor.reshape(-1, t1 // p1, p1, t0 // p0, p0)\
                 .permute(0, 1, 3, 2, 4)

def reshape_back(tensor, t1, t0, p1, p0):
    return tensor.reshape(-1, t1 // p1, t0 // p0, p1, p0)\
                 .permute(0, 1, 3, 2, 4)