"""
Author: ViolinSolo
Date: 2023-04-23 12:57:57
LastEditTime: 2023-04-29 10:45:14
LastEditors: ViolinSolo
Description: 
FilePath: /zero-cost-proxies/alethiometer/zero_cost_metrics/zico.py
"""

# =============================================================================
#   Copyright (C) 2010-2021 Alibaba Group Holding Limited.
# https://github.com/SLDGroup/ZiCo/blob/main/ZeroShotProxy/compute_zico.py
# =============================================================================


import torch
from torch import nn
import numpy as np
from . import measure


def getgrad(model: torch.nn.Module, grad_dict: dict, step_iter=0):
    if step_iter == 0:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                # print(mod.weight.grad.data.size())
                # print(mod.weight.data.size())
                # print(name, mod.weight.grad.data.size())
                if mod.weight.grad is None:
                    # print(f"{name} grad is None")
                    continue

                # print(name, mod.weight.grad.data.size())
                grad_dict[name] = [mod.weight.grad.data.cpu().reshape(-1).numpy()]
    else:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                if mod.weight.grad is None:
                    # print(f"{name} grad is None")
                    continue
                # print(name, mod.weight.grad.data.size())
                grad_dict[name].append(mod.weight.grad.data.cpu().reshape(-1).numpy())
    return grad_dict


def caculate_zico(grad_dict):
    allgrad_array = None
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname] = np.array(grad_dict[modname])
    nsr_mean_sum = 0
    nsr_mean_sum_abs = 0
    nsr_mean_avg = 0
    nsr_mean_avg_abs = 0
    for j, modname in enumerate(grad_dict.keys()):
        # if grad_dict[modname].shape is (1, N), this would be a bug, bacause std would be list of 0
        nsr_std = np.std(grad_dict[modname], axis=0)
        nonzero_idx = np.nonzero(nsr_std)[0]
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
        tmpsum = np.sum(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
        if tmpsum == 0:
            pass
        else:
            nsr_mean_sum_abs += np.log(tmpsum)
            nsr_mean_avg_abs += np.log(
                np.mean(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
            )
    return nsr_mean_sum_abs


@measure("zico", bn=True)
def getzico(
    net,
    inputs,
    targets,
    loss_fn=None,
    split_data=1,  # these are necessary arguments limited by *zero_cost_metrics.__init__.calc_metric*, if you want to add more arguments, modify @metric decorator's parameters to provide dynamic default values.
):
    # def getzico(network, trainloader, lossfunc):
    grad_dict = {}
    net.train()

    # split_data = 2
    # =====================
    # mandatory double the split_data parameter, to correctly calculate the std of cross-batch gradient.
    N = inputs.shape[0]
    split_data = min(split_data * 2, N)

    assert (
        split_data > 1
    ), "zico need split_data > 1, at least 2, so cross-batch gradient std can be non-zero list."

    # net.cuda()
    # for i, batch in enumerate(trainloader):
    # network.zero_grad()
    # data,label = batch[0],batch[1]
    # data,label=data.cuda(),label.cuda()

    # logits = network(data)
    # loss = loss_fn(logits, label)
    # loss.backward()
    # grad_dict= getgrad(network, grad_dict,i)

    net.zero_grad()
    # N = inputs.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        outputs = net.forward(inputs[st:en])
        if type(outputs) is tuple:
            # outputs, _ = outputs #original code
            # original code don't came into bug, is original input feat into nn.cross_entroy
            # which means, it mistakenly identify n_class from original 10[cifar10], to 64.
            # feat, logits = outputs
            # feat.shape = [64, 64]
            # logits.shape = [64, 10]
            # nn.cross_entropy(inputs, targets) takes: input [N, C] and targets [N, ] of class_index or class_prob
            _, outputs = (
                outputs  # TODO: need logits just like synflow and snip, so fix here temporarily
            )
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()
        grad_dict = getgrad(net, grad_dict, step_iter=sp)

    res = caculate_zico(grad_dict)
    return res
