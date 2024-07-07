import torch


@torch.fx.wrap
def splitter(x):
    return (x, x)
