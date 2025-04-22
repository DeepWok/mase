import torch
import distributed_utils as linklink


class Distributed(object):
    """Decorator for Distributed tensor range"""
    def __init__(self, func):
        self._func = func

        def sync(data_min, data_max):
            linklink.allreduce(data_min, reduce_op=linklink.allreduceOp_t.Min)
            linklink.allreduce(data_max, reduce_op=linklink.allreduceOp_t.Max)
            return data_min, data_max

        self._sync = sync

    def __call__(self, args, **kwargs):
        return self._sync(*(self._func(args, **kwargs)))


class AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.zeros_like(input)
        output.copy_(input)
        linklink.allreduce(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        in_grad = torch.zeros_like(grad_output)
        in_grad.copy_(grad_output)
        linklink.allreduce(in_grad)
        return in_grad


def allaverage(tensor):
    tensor.data /= linklink.get_world_size()
    linklink.allreduce(tensor.data)
    return tensor


def allaverage_autograd(tensor):
    tensor /= linklink.get_world_size()
    tensor = AllReduce().apply(tensor)
    return tensor


def allreduce(tensor):
    linklink.allreduce(tensor.data)
