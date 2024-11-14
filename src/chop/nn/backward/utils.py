import torch


def clone_autograd_fn(autograd_fn: torch.autograd.Function):
    class ClonedAutogradFn(autograd_fn):
        pass

    return ClonedAutogradFn
