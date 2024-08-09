# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import cast, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import torch
from torch.distributed._tensor._op_schema import (
    OutputSpecType,
)
from torch.distributed._tensor._tp_conv import (
    convolution_backward_handler,
    convolution_handler,
)
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Replicate,
    TensorMeta,
)

from torch.distributed.device_mesh import DeviceMesh

import chop.distributed.tensor.api as dtensor
from chop.distributed.tensor._sharding_prop import ShardingPropagator

aten = torch.ops.aten


def get_replicate_spec(tensor_arg: torch.Tensor, mesh: "DeviceMesh") -> DTensorSpec:
    # scalar tensor can be safely treated as replicated
    replication_spec = DTensorSpec(
        mesh,
        (Replicate(),) * mesh.ndim,
        tensor_meta=TensorMeta(
            shape=tensor_arg.shape,
            stride=tensor_arg.stride(),
            dtype=tensor_arg.dtype,
        ),
    )

    return replication_spec


def decompose_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    """
    Decomposes a op to core ATen op, this handler is mostly here
    for inference mode usage where the ops are not core aten ops.
    """
    r = op_call.decompose(*args, **kwargs)
    if r is not NotImplemented:
        return r
    else:
        raise RuntimeError("Decomposition failed")


def is_same_size_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> bool:
    lhs = cast(torch.Tensor, args[0])
    rhs = cast(torch.Tensor, args[1])
    return lhs.shape == rhs.shape


class OpDispatcher:
    """
    Op dispatching class instance to handle args/kwargs pre-processing (un-wrapping), sharding
    propagation, redistribute local args, local compute, and post-processing (re-wrapping). It
    also handles any op specific logic if necessary.
    """

    def __init__(self) -> None:
        self.sharding_propagator = ShardingPropagator()
        self._random_ops = {
            aten.native_dropout.default,
            aten.normal_.default,
            aten.rand_like.default,
            aten.randn_like.default,
            aten.randint_like.default,
            aten.randint_like.low_dtype,
            aten.randint_like.low_dtype_out,
            aten.uniform_.default,
            aten.bernoulli.default,
            aten.bernoulli_.float,
        }
        self._custom_op_handlers = {
            aten.linear.default: decompose_handler,
            aten.is_same_size.default: is_same_size_handler,
            aten.convolution.default: convolution_handler,
            aten.convolution_backward.default: convolution_backward_handler,
        }

        # This flag is used internally to control whether we treat the torch.Tensor(non-DTensor)
        # as implicitly replicated or we throw error to user.
        self._allow_implicit_replication = True

    def dispatch(
        self,
        op_call: torch._ops.OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ) -> object:
        """
        Main dispatching logic
        """
        # operators that does not need to go through sharding propagation

        # run local op computation with potentially modified args/kwargs
        local_tensor_args = [
            arg._local_tensor if isinstance(arg, dtensor.DTensor) else arg
            for arg in args
        ]

        local_tensor_kwargs = {
            k: v.local_tensor if isinstance(v, dtensor.DTensor) else v
            for k, v in kwargs.items()
        }

        local_results = op_call(*local_tensor_args, **local_tensor_kwargs)

        # We still need to wrap the local result in a DTensor here in two cases
        # 1. When creating a nn.Parameter from a DTensor, it must call tensor.detach
        #   and the return type must match the input type (DTensor).
        # 2. When a single FX op decomposes into multiple aten ops (e.g. torch.embedding)
        if op_call._name == "aten::detach":
            return self.wrap(
                local_results,
                DTensorSpec(
                    mesh=DeviceMesh(
                        "cuda",
                        mesh=torch.Tensor(
                            # todo: generalize
                            [
                                [0, 1, 2, 3],
                                [4, 5, 6, 7],
                            ]
                        ),
                    ),
                    placements=args[0]._spec.placements,
                    tensor_meta=TensorMeta(
                        shape=args[0]._spec.tensor_meta.shape,
                        stride=args[0]._spec.tensor_meta.stride,
                        dtype=args[0]._spec.tensor_meta.dtype,
                    ),
                ),
            )

        return local_results

    @staticmethod
    def wrap(
        res: object,
        spec: OutputSpecType,
    ) -> object:
        if isinstance(res, torch.Tensor):
            return dtensor.DTensor(
                res,
                spec,
                requires_grad=res.requires_grad,
            )
        elif isinstance(res, (list, tuple)):
            res_list = []
            for e, s in zip(res, spec):
                res_list.append(OpDispatcher.wrap(e, s))

            return tuple(res_list) if isinstance(res, tuple) else res_list
        else:
            # if the res contains only non tensor values (i.e. int/float/none), we simply return it
            # without rewrapping to DTensor.
            return res
