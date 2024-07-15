import itertools
import operator

import torch
import torch.nn.functional as F
from torch.distributed._tensor._op_schema import OpStrategy, PlacementStrategy
from torch.distributed._tensor.placement_types import (
    Replicate,
    Shard,
    DTensorSpec,
    TensorMeta,
)

from chop.tools import get_logger

from .ops.matrix_ops import (
    transpose_strategy,
    mm_strategy,
    addmm_strategy,
    bmm_strategy,
    baddmm_strategy,
)
from .ops.view_ops import get_reshape_strategy
from .ops.pointwise_ops import pointwise_strategy, linear_pointwise_strategy
from .ops.math_ops import softmax_strategy, layer_norm_strategy
from .ops.embedding_ops import embedding_strategy
from .ops.tensor_ops import tensor_op_strategy, tensor_equal_strategy

logger = get_logger(__name__)


def find_shape_and_dtype(arg):

    if isinstance(arg, dict):
        in_shape = arg["shape"]
        in_dtype = arg["torch_dtype"]
    elif isinstance(arg, (tuple, list)):
        arg = torch.Tensor(arg)
        in_shape = arg.shape
        in_dtype = arg.dtype
    elif isinstance(arg, torch.Size):
        arg = torch.Tensor(list(arg))
        in_shape = arg.shape
        in_dtype = arg.dtype
    elif isinstance(arg, (float, int)):
        arg = torch.Tensor([arg])
        in_shape = arg.shape
        in_dtype = arg.dtype
    else:
        logger.warning(f"Unknown type for arg: {arg}")
        in_shape = tuple()
        in_dtype = type(arg)

    return in_shape, in_dtype


def placeholder_or_getattr_strategy(meta, mesh, skip_fully_replicated=False):
    ndims = len(meta["common"]["results"]["data_out_0"]["shape"])
    opts = [Replicate()] + [Shard(dim) for dim in range(ndims)]

    tensor_meta = TensorMeta(
        shape=meta["common"]["results"]["data_out_0"]["shape"],
        stride=None,
        dtype=meta["common"]["results"]["data_out_0"]["torch_dtype"],
    )

    shardings = []
    for sharding in itertools.product(opts, repeat=2):
        if skip_fully_replicated and sharding == (Replicate(), Replicate()):
            continue
        spec = DTensorSpec(mesh=mesh, placements=sharding, tensor_meta=tensor_meta)
        shardings.append(PlacementStrategy(input_specs=spec, output_specs=spec))
    return OpStrategy(shardings)


def fully_replicated_strategy(meta, mesh):
    """
    Output of ops like size, getitem etc are always fully replicated
    """
    sharding = [Replicate(), Replicate()]

    # call_method nodes don't list input tensor in the args list, but
    # tensor is copied into meta["common"]["self"] when add_value = True
    # is passed to add_common_metadata_pass
    if meta.node.op == "call_method":
        in_shape = meta["common"]["self"].shape
        in_dtype = meta["common"]["self"].dtype
    else:
        first_arg_key = (
            "data_in_0"
            if "data_in_0" in meta["common"]["args"]
            else [i for i in meta["common"]["args"].keys()][0]
        )
        arg = meta["common"]["args"][first_arg_key]
        in_shape, in_dtype = find_shape_and_dtype(arg)

    in_spec = DTensorSpec(
        mesh,
        sharding,
        tensor_meta=TensorMeta(
            shape=in_shape,
            stride=None,
            dtype=in_dtype,
        ),
    )

    dtype_key = (
        "torch_dtype"
        if "torch_dtype" in meta["common"]["results"]["data_out_0"].keys()
        else "type"
    )
    out_dtype = meta["common"]["results"]["data_out_0"][dtype_key]
    out_spec = DTensorSpec(
        mesh,
        sharding,
        tensor_meta=TensorMeta(
            shape=meta["common"]["results"]["data_out_0"]["shape"],
            stride=None,
            dtype=out_dtype,
        ),
    )

    shardings = [PlacementStrategy(input_specs=in_spec, output_specs=out_spec)]

    return OpStrategy(shardings)


AUTOSHARDING_FUNCTIONS = {
    # embedding_ops.py
    F.embedding: embedding_strategy,
    # math_ops.py
    torch.softmax: softmax_strategy,
    F.softmax: softmax_strategy,
    torch.log_softmax: softmax_strategy,
    F.log_softmax: softmax_strategy,
    F.layer_norm: layer_norm_strategy,
    # matrix_ops.py
    torch.transpose: transpose_strategy,
    torch.mm: mm_strategy,
    torch.matmul: bmm_strategy,
    torch.addmm: addmm_strategy,
    torch.bmm: bmm_strategy,
    torch.baddbmm: baddmm_strategy,
    # pointwise_ops.py
    torch.add: linear_pointwise_strategy,
    operator.add: linear_pointwise_strategy,
    torch.Tensor.add_: linear_pointwise_strategy,
    torch.Tensor.to: linear_pointwise_strategy,
    torch.abs: pointwise_strategy,
    torch.acos: pointwise_strategy,
    torch.acosh: pointwise_strategy,
    torch.addcdiv: pointwise_strategy,
    torch.addcmul: pointwise_strategy,
    torch.angle: pointwise_strategy,
    torch.asin: pointwise_strategy,
    torch.asinh: pointwise_strategy,
    torch.atan: pointwise_strategy,
    torch.atan2: pointwise_strategy,
    torch.atanh: pointwise_strategy,
    torch.bitwise_and: pointwise_strategy,
    torch.bitwise_left_shift: pointwise_strategy,
    torch.bitwise_not: pointwise_strategy,
    torch.bitwise_or: pointwise_strategy,
    torch.bitwise_right_shift: pointwise_strategy,
    torch.bitwise_xor: pointwise_strategy,
    torch.ceil: pointwise_strategy,
    torch.clamp: pointwise_strategy,
    torch.clip: pointwise_strategy,
    torch.conj_physical: pointwise_strategy,
    torch.copysign: pointwise_strategy,
    torch.cos: pointwise_strategy,
    torch.cosh: pointwise_strategy,
    torch.deg2rad: pointwise_strategy,
    torch.digamma: pointwise_strategy,
    torch.div: pointwise_strategy,
    torch.eq: pointwise_strategy,
    # operator.eq: pointwise_strategy,
    torch.erf: pointwise_strategy,
    torch.erfc: pointwise_strategy,
    torch.erfinv: pointwise_strategy,
    torch.exp: pointwise_strategy,
    torch.exp2: pointwise_strategy,
    torch.expm1: pointwise_strategy,
    torch.float_power: pointwise_strategy,
    torch.floor: pointwise_strategy,
    torch.fmod: pointwise_strategy,
    torch.frac: pointwise_strategy,
    torch.ge: pointwise_strategy,
    torch.gt: pointwise_strategy,
    operator.gt: pointwise_strategy,
    torch.hypot: pointwise_strategy,
    torch.i0: pointwise_strategy,
    torch.igamma: pointwise_strategy,
    torch.igammac: pointwise_strategy,
    torch.isnan: pointwise_strategy,
    torch.ldexp: pointwise_strategy,
    torch.lt: pointwise_strategy,
    operator.lt: pointwise_strategy,
    torch.le: pointwise_strategy,
    torch.lerp: pointwise_strategy,
    torch.lgamma: pointwise_strategy,
    torch.log: pointwise_strategy,
    torch.log10: pointwise_strategy,
    torch.log1p: pointwise_strategy,
    torch.log2: pointwise_strategy,
    torch.logaddexp: pointwise_strategy,
    torch.logaddexp2: pointwise_strategy,
    torch.logical_and: pointwise_strategy,
    torch.logical_not: pointwise_strategy,
    torch.logical_or: pointwise_strategy,
    torch.logical_xor: pointwise_strategy,
    torch.logit: pointwise_strategy,
    torch.masked_fill: pointwise_strategy,
    torch.maximum: pointwise_strategy,
    torch.mul: pointwise_strategy,
    operator.mul: pointwise_strategy,
    torch.mvlgamma: pointwise_strategy,
    torch.nan_to_num: pointwise_strategy,
    torch.ne: pointwise_strategy,
    operator.ne: pointwise_strategy,
    torch.neg: pointwise_strategy,
    torch.nextafter: pointwise_strategy,
    torch.polygamma: pointwise_strategy,
    torch.positive: pointwise_strategy,
    torch.pow: pointwise_strategy,
    torch.reciprocal: pointwise_strategy,
    torch.rad2deg: pointwise_strategy,
    torch.relu: pointwise_strategy,
    torch.remainder: pointwise_strategy,
    torch.round: pointwise_strategy,
    torch.rsqrt: pointwise_strategy,
    torch.rsub: pointwise_strategy,
    torch.sgn: pointwise_strategy,
    torch.sigmoid: pointwise_strategy,
    torch.sign: pointwise_strategy,
    torch.signbit: pointwise_strategy,
    torch.sin: pointwise_strategy,
    torch.sinc: pointwise_strategy,
    torch.sinh: pointwise_strategy,
    torch.sqrt: pointwise_strategy,
    torch.square: pointwise_strategy,
    torch.sub: pointwise_strategy,
    operator.sub: pointwise_strategy,
    torch.tan: pointwise_strategy,
    torch.tanh: pointwise_strategy,
    torch.true_divide: pointwise_strategy,
    torch.trunc: pointwise_strategy,
    torch.where: pointwise_strategy,
    torch.xlogy: pointwise_strategy,
    F.gelu: pointwise_strategy,
    F.relu: pointwise_strategy,
    F.sigmoid: pointwise_strategy,
    F.silu: pointwise_strategy,
    F.tanh: pointwise_strategy,
    torch.Tensor.abs_: pointwise_strategy,
    torch.Tensor.acos_: pointwise_strategy,
    torch.Tensor.acosh_: pointwise_strategy,
    torch.Tensor.add_: pointwise_strategy,
    torch.Tensor.addcdiv_: pointwise_strategy,
    torch.Tensor.addcmul_: pointwise_strategy,
    torch.Tensor.asin_: pointwise_strategy,
    torch.Tensor.asinh_: pointwise_strategy,
    torch.Tensor.atan2_: pointwise_strategy,
    torch.Tensor.atan_: pointwise_strategy,
    torch.Tensor.atanh_: pointwise_strategy,
    torch.Tensor.bitwise_and_: pointwise_strategy,
    torch.Tensor.bitwise_left_shift_: pointwise_strategy,
    torch.Tensor.bitwise_not_: pointwise_strategy,
    torch.Tensor.bitwise_or_: pointwise_strategy,
    torch.Tensor.bitwise_right_shift_: pointwise_strategy,
    torch.Tensor.bitwise_xor_: pointwise_strategy,
    torch.Tensor.ceil_: pointwise_strategy,
    torch.Tensor.clamp_: pointwise_strategy,
    torch.Tensor.clip_: pointwise_strategy,
    torch.Tensor.conj_physical_: pointwise_strategy,
    torch.Tensor.copysign_: pointwise_strategy,
    torch.Tensor.cos_: pointwise_strategy,
    torch.Tensor.cosh_: pointwise_strategy,
    torch.Tensor.deg2rad_: pointwise_strategy,
    torch.Tensor.digamma_: pointwise_strategy,
    torch.Tensor.div_: pointwise_strategy,
    torch.Tensor.erf_: pointwise_strategy,
    torch.Tensor.erfc_: pointwise_strategy,
    torch.Tensor.erfinv_: pointwise_strategy,
    torch.Tensor.exp2_: pointwise_strategy,
    torch.Tensor.exp_: pointwise_strategy,
    torch.Tensor.expm1_: pointwise_strategy,
    torch.Tensor.float_power_: pointwise_strategy,
    torch.Tensor.floor_: pointwise_strategy,
    torch.Tensor.fmod_: pointwise_strategy,
    torch.Tensor.frac_: pointwise_strategy,
    torch.Tensor.hypot_: pointwise_strategy,
    torch.Tensor.i0_: pointwise_strategy,
    torch.Tensor.igamma_: pointwise_strategy,
    torch.Tensor.igammac_: pointwise_strategy,
    torch.Tensor.ldexp_: pointwise_strategy,
    torch.Tensor.lerp_: pointwise_strategy,
    torch.Tensor.lgamma_: pointwise_strategy,
    torch.Tensor.log10_: pointwise_strategy,
    torch.Tensor.log1p_: pointwise_strategy,
    torch.Tensor.log2_: pointwise_strategy,
    torch.Tensor.log_: pointwise_strategy,
    torch.Tensor.logical_and_: pointwise_strategy,
    torch.Tensor.logical_not_: pointwise_strategy,
    torch.Tensor.logical_or_: pointwise_strategy,
    torch.Tensor.logical_xor_: pointwise_strategy,
    torch.Tensor.logit_: pointwise_strategy,
    torch.Tensor.mul_: pointwise_strategy,
    torch.Tensor.mvlgamma_: pointwise_strategy,
    torch.Tensor.nan_to_num_: pointwise_strategy,
    torch.Tensor.neg_: pointwise_strategy,
    torch.Tensor.nextafter_: pointwise_strategy,
    torch.Tensor.polygamma_: pointwise_strategy,
    torch.Tensor.pow_: pointwise_strategy,
    torch.Tensor.reciprocal_: pointwise_strategy,
    torch.Tensor.rad2deg_: pointwise_strategy,
    torch.Tensor.relu_: pointwise_strategy,
    torch.Tensor.remainder_: pointwise_strategy,
    torch.Tensor.round_: pointwise_strategy,
    torch.Tensor.rsqrt_: pointwise_strategy,
    torch.Tensor.sgn_: pointwise_strategy,
    torch.Tensor.sigmoid_: pointwise_strategy,
    torch.Tensor.sign_: pointwise_strategy,
    torch.Tensor.sin_: pointwise_strategy,
    torch.Tensor.sinc_: pointwise_strategy,
    torch.Tensor.sinh_: pointwise_strategy,
    torch.Tensor.sqrt_: pointwise_strategy,
    torch.Tensor.square_: pointwise_strategy,
    torch.Tensor.sub_: pointwise_strategy,
    torch.Tensor.tan_: pointwise_strategy,
    torch.Tensor.tanh_: pointwise_strategy,
    torch.Tensor.trunc_: pointwise_strategy,
    torch.Tensor.xlogy_: pointwise_strategy,
    # tensor_ops.py
    torch.ones: fully_replicated_strategy,
    torch.full: fully_replicated_strategy,
    torch.Tensor.clone: tensor_op_strategy,
    torch.Tensor.contiguous: tensor_op_strategy,
    torch.Tensor.copy_: tensor_op_strategy,
    torch.Tensor.detach: tensor_op_strategy,
    torch.Tensor.fill_: tensor_op_strategy,
    torch.Tensor.zero_: tensor_op_strategy,
    torch.Tensor.equal: tensor_equal_strategy,
    torch.Tensor.is_same_size: tensor_equal_strategy,
}

AUTOSHARDING_METHODS = {
    # view_ops.py
    "view": get_reshape_strategy(torch.Tensor.view),
    "reshape": get_reshape_strategy(torch.Tensor.reshape),
    "expand": get_reshape_strategy(torch.Tensor.expand),
    "permute": get_reshape_strategy(torch.Tensor.permute),
    "transpose": get_reshape_strategy(torch.Tensor.transpose),
    "masked_fill": pointwise_strategy,
    "masked_fill_": pointwise_strategy,
    "contiguous": tensor_op_strategy,
}

IMPLICIT_FUNCS = [
    operator.getitem,
    getattr,
    torch.finfo,
    torch.arange,
]

IMPLICIT_METHODS = [
    "size",
]
