import torch.nn.functional as F
import torch


def get_quant_cfg(graph):
    quant_cfg = {"by": "name"}
    for node in graph.fx_graph.nodes:
        if node.meta["mase"].parameters["software"].get("quant") is not None:
            quant_cfg[node.name] = {}
            quant_cfg[node.name]["config"] = node.meta["mase"].parameters["software"][
                "quant"
            ]["config"]
            quant_cfg[node.name]["similarity"] = node.meta["mase"].parameters[
                "software"
            ]["quant"]["similarity"]
    return quant_cfg


def set_meta_quant_cfg(node, similarity, cfg):
    node.meta["mase"].parameters["software"]["quant"] = {}
    node.meta["mase"].parameters["software"]["quant"]["similarity"] = similarity
    node.meta["mase"].parameters["software"]["quant"]["config"] = cfg


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
