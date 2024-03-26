#Copyright (C) 2010-2021 Alibaba Group Holding Limited.
# =============================================================================

import torch
import numpy as np

from torch import nn
from . import measure


def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is None:
                    continue
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net

@measure("zen", bn=True)
def compute_zen_score(net, inputs, targets, loss_fn=None, split_data=1,
                      repeat=1, mixup_gamma=1e-2, fp16=False):
    nas_score_list = []

    device = inputs.device
    dtype = torch.half if fp16 else torch.float32


    with torch.no_grad():
        for repeat_count in range(repeat):
            network_weight_gaussian_init(net)
            input = torch.randn(size=list(inputs.shape), device=device, dtype=dtype)
            input2 = torch.randn(size=list(inputs.shape), device=device, dtype=dtype)
            mixup_input = input + mixup_gamma * input2

            try:
                # 尝试调用 forward_before_global_avg_pool
                output = net.forward_before_global_avg_pool(input)
                mixup_output = net.forward_before_global_avg_pool(mixup_input)
            except AttributeError:
                # 如果没有 forward_before_global_avg_pool 方法，则直接调用 forward 方法
                output = net.forward(input.to('cpu'))
                mixup_output = net.forward(mixup_input.to('cpu'))

            try:
                nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
                nas_score = torch.mean(nas_score)
            except IndexError:
                nas_score = torch.sum(torch.abs(output - mixup_output))
                nas_score = torch.mean(nas_score)
            # compute BN scaling
            log_bn_scaling_factor = 0.0
            for m in net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                    log_bn_scaling_factor += torch.log(bn_scaling_factor)
                pass
            pass
            nas_score = torch.log(nas_score) + log_bn_scaling_factor
            nas_score_list.append(float(nas_score))

    avg_nas_score = float(np.mean(nas_score_list))

    return avg_nas_score
