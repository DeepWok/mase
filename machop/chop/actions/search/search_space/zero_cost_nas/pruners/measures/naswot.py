"""
Author: ViolinSolo
Date: 2023-04-11 15:15:33
LastEditTime: 2023-04-11 17:57:40
LastEditors: ViolinSolo
Description: naswot function to calculate the zc proxy
FilePath: /zero-cost-proxies/alethiometer/zero_cost_metrics/naswot.py
"""

import torch
import torch.nn as nn
import numpy as np
from . import measure


def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    return ld


def safe_hooklogdet(K):
    s, ld = np.linalg.slogdet(K)
    return 0 if (np.isneginf(ld) and s == 0) else ld


@measure("naswot", bn=True, layerwise=False, return_Kmats=False)
@measure("nwot_Kmats", bn=True, layerwise=False, return_Kmats=True)
@measure("lnwot", bn=True, layerwise=True, return_Kmats=False)
@measure("lnwot_Kmats", bn=True, layerwise=True, return_Kmats=True)
def compute_naswot(
    net, inputs, targets, loss_fn, split_data=1, layerwise=False, return_Kmats=False
):
    net.eval()

    K_layer_names = []  # list of registered layer (module) names.
    if layerwise:
        K_mats = []  # list of **naswot matrix**, layer-wise | [e]([mat, ...])
        K_mats_logdet = (
            []
        )  # list of **naswot matrix** logdet, layer-wise | [e]([mat, ...])

        def counting_forward_hook(module, inp, out):
            out = out.view(out.size(0), -1)
            x = (out > 0).float()
            K = x @ x.t()
            K2 = (1.0 - x) @ (1.0 - x.t())
            matrix = K + K2

            K_layer_names.append(module.alias)
            K_mats.append(matrix)
            K_mats_logdet.append(safe_hooklogdet(K_mats[-1].cpu().numpy()))

    else:
        K_mat = 0.0  # **naswot matrix**, NONE-layer-wise | e (mat,) ===> using torch broadcasting tech to init zero-like matrix
        K_mat_logdet = 0.0  # **naswot matrix** logdet, NONE-layer-wise | e (mat,)

        def counting_forward_hook(module, inp, out):
            out = out.view(out.size(0), -1)
            x = (out > 0).float()
            K = x @ x.t()
            K2 = (1.0 - x) @ (1.0 - x.t())
            matrix = K + K2

            K_layer_names.append(module.alias)
            nonlocal K_mat
            K_mat = K_mat + matrix

    # register forward hook fn
    registered_layers = []
    for name, module in net.named_modules():
        if (
            isinstance(module, nn.Conv2d)
            or isinstance(module, nn.Linear)
            or isinstance(module, nn.Conv1d)
        ):
            module.alias = name
            module.register_forward_hook(counting_forward_hook)
            registered_layers.append(name)

    with torch.no_grad():
        net(inputs)

    # using set instead, since some under some conditions, the list order changed.
    assert set(registered_layers) == set(
        K_layer_names
    ), "Not all module forward hook fn were triggered successfully"

    if layerwise:
        return (K_mats, K_mats_logdet) if return_Kmats else K_mats_logdet
    else:
        K_mat_logdet = safe_hooklogdet(K_mat.cpu().numpy())
        return (K_mat, K_mat_logdet) if return_Kmats else K_mat_logdet
