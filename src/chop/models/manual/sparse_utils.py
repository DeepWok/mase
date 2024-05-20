import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


def print_trainable_parameters(model: nn.Module):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"Trainable params: {trainable_params} || All params: {all_param} || Trainable %: {100 * trainable_params / all_param:.2f}"
    )
