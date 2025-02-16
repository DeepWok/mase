import torch

from chop.ir.onnx.onnx_operators import (
    onnx_gemm,
    onnx_slice,
    onnx_squeeze,
    onnx_unsqueeze,
    onnx_gather,
    onnx_shape,
    onnx_reshape,
    onnx_identity,
    onnx_expand,
    onnx_where,
    onnx_full,
    onnx_min,
    onnx_permute,
)

ONNX_TO_TORCH_DTYPE = [
    torch.float16,
    torch.float,
    torch.double,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.float,  # uint16
    torch.float,  # uint32
    torch.float,  # uint64
    torch.bool,
    torch.float,  # string
    torch.bfloat16,
    torch.float,  # float8e4m3fn
    torch.float,  # float8e4m3fnuz
    torch.float,  # float8e5m2
    torch.float,  # float8e5m2fnuz
    torch.float,  # uint4
    torch.float,  # int4
]


def onnx_to_torch_dtype(dtype):
    """
    Return Pytorch equivalent of ONNX data type.
    When an equivalent is not available, return float.
    """
    return ONNX_TO_TORCH_DTYPE[dtype]


ONNX_OP_MAPPING = {
    "Greater": {
        "fx_op": "call_function",
        "target": torch.greater,
        "input_mapping": ["input", "other"],
        "attribute_mapping": {},
    },
    "Abs": {
        "fx_op": "call_function",
        "target": torch.abs,
        "input_mapping": ["input"],
        "attribute_mapping": {},
    },
    "Sin": {
        "fx_op": "call_function",
        "target": torch.sin,
        "input_mapping": ["input"],
        "attribute_mapping": {},
    },
    "Cos": {
        "fx_op": "call_function",
        "target": torch.cos,
        "input_mapping": ["input"],
        "attribute_mapping": {},
    },
    "Sigmoid": {
        "fx_op": "call_function",
        "target": torch.sigmoid,
        "input_mapping": ["input"],
        "attribute_mapping": {},
    },
    "ArgMax": {
        "fx_op": "call_function",
        "target": torch.argmax,
        "input_mapping": ["input"],
        "attribute_mapping": {},
    },
    "Flatten": {
        "fx_op": "call_function",
        "target": torch.flatten,
        "input_mapping": ["input"],
        "attribute_mapping": {},
    },
    "Split": {
        "fx_op": "call_function",
        "target": torch.split,
        "input_mapping": ["tensor"],
        "attribute_mapping": {},
    },
    "Not": {
        "fx_op": "call_function",
        "target": torch.logical_not,
        "input_mapping": ["input"],
        "attribute_mapping": {},
    },
    "Identity": {
        "fx_op": "call_function",
        "target": onnx_identity,
        "input_mapping": ["input"],
        "attribute_mapping": {},
    },
    "Tile": {
        "fx_op": "call_function",
        "target": torch.tile,
        "input_mapping": ["input"],
        "attribute_mapping": {},
    },
    "Erf": {
        "fx_op": "call_function",
        "target": torch.special.erf,
        "input_mapping": ["input"],
        "attribute_mapping": {},
    },
    "Tanh": {
        "fx_op": "call_function",
        "target": torch.tanh,
        "input_mapping": ["input"],
        "attribute_mapping": {},
    },
    "LessOrEqual": {
        "fx_op": "call_function",
        "target": torch.le,
        "input_mapping": ["input"],
        "attribute_mapping": {},
    },
    "Min": {
        "fx_op": "call_function",
        "target": onnx_min,
        "input_mapping": ["input"],
        "attribute_mapping": {},
    },
    "Neg": {
        "fx_op": "call_function",
        "target": torch.neg,
        "input_mapping": ["input"],
        "attribute_mapping": {},
    },
    "Log": {
        "fx_op": "call_function",
        "target": torch.log,
        "input_mapping": ["input"],
        "attribute_mapping": {},
    },
    "Slice": {
        "fx_op": "call_function",
        "target": onnx_slice,
        "input_mapping": ["data", "starts", "ends", "axes", "steps"],
        "attribute_mapping": {},
    },
    "Gemm": {
        "fx_op": "call_function",
        "target": onnx_gemm,
        "input_mapping": ["A", "B", "C"],
        "attribute_mapping": {
            "alpha": "alpha",
            "beta": "beta",
            "transA": "transA",
            "transB": "transB",
        },
        "attribute_default": {
            "alpha": 1.0,
            "beta": 1.0,
            "transA": False,
            "transB": False,
        },
    },
    "ConstantOfShape": {
        "name_override": "full",
        "fx_op": "call_function",
        "target": onnx_full,
        "input_mapping": ["size"],
        "attribute_mapping": {"value": "fill_value"},
        "attribute_transform": {"value": None},
    },
    "ReduceMean": {
        "name_override": "mean",
        "fx_op": "call_function",
        "target": torch.mean,
        "input_mapping": ["input"],
        "attribute_mapping": {"keepdims": "", "axes": ""},
        "attribute_transform": {
            "keepdims": None,
            "axes": None,
        },
        "attribute_default": {"keepdims": 1, "axes": None},
    },
    "Expand": {
        "fx_op": "call_function",
        "target": onnx_expand,
        "input_mapping": ["input", "size"],
        "attribute_mapping": {},
    },
    "Shape": {
        "fx_op": "call_function",
        "target": onnx_shape,
        "input_mapping": ["input"],
        "attribute_mapping": {},
        "attribute_transform": {},
    },
    "Cast": {
        "name_override": "to",
        "fx_op": "call_method",
        "target": "to",
        "input_mapping": ["self"],
        "attribute_mapping": {"to": "dtype", "saturate": ""},
        "attribute_transform": {"to": onnx_to_torch_dtype, "saturate": None},
    },
    "Gather": {
        "fx_op": "call_function",
        "target": onnx_gather,
        "input_mapping": ["input", "index"],
        "attribute_mapping": {"axis": "dim"},
        "attribute_transform": {"axis": None},
        "attribute_default": {"axis": 0},
    },
    "Unsqueeze": {
        "fx_op": "call_function",
        "target": onnx_unsqueeze,
        "input_mapping": ["input", "dim"],
        "attribute_mapping": {
            "axes": "dim"
        },  # axes given as an attribute up to version 13, and input afterwards
        "attribute_transform": {"axes": None},
        # "attribute_default": {"axes": None},
    },
    "Concat": {
        "name_override": "cat",
        "fx_op": "call_function",
        "target": torch.cat,
        "input_mapping": ["tensors"],
        "attribute_mapping": {"axis": "dim"},
        "attribute_transform": {"axis": None},
    },
    "Reshape": {
        "fx_op": "call_function",
        "target": onnx_reshape,
        "input_mapping": ["input", "shape"],
        "attribute_mapping": {"allowzero": ""},
        "attribute_transform": {"allowzero": None},
    },
    "Add": {
        "fx_op": "call_function",
        "target": torch.add,
        "input_mapping": ["input", "other"],
        "attribute_mapping": {},
    },
    "Sub": {
        "fx_op": "call_function",
        "target": torch.sub,
        "input_mapping": ["input", "other"],
        "attribute_mapping": {},
    },
    "Squeeze": {
        "fx_op": "call_function",
        "target": onnx_squeeze,
        "input_mapping": ["input", "dim"],
        "attribute_mapping": {"axes": "dim"},
        "attribute_transform": {"axes": lambda x: tuple([i for i in x])},
    },
    "Range": {
        "fx_op": "call_function",
        "target": torch.arange,
        "input_mapping": ["start", "end", "step"],
        "attribute_mapping": {},
    },
    "Less": {
        "fx_op": "call_function",
        "target": torch.less,
        "input_mapping": ["input", "other"],
        "attribute_mapping": {},
    },
    "Where": {
        "fx_op": "call_function",
        "target": onnx_where,
        "input_mapping": ["condition", "input", "other"],
        "attribute_mapping": {},
    },
    "Mul": {
        "fx_op": "call_function",
        "target": torch.mul,
        "input_mapping": ["input", "other"],
        "attribute_mapping": {},
    },
    "Equal": {
        "name_override": "eq",
        "fx_op": "call_function",
        "target": torch.eq,
        "input_mapping": ["input", "other"],
        "attribute_mapping": {},
    },
    "CumSum": {
        "fx_op": "call_function",
        "target": torch.cumsum,
        "input_mapping": ["input", "dim"],
        "attribute_mapping": {},
    },
    "Pow": {
        "fx_op": "call_function",
        "target": torch.pow,
        "input_mapping": ["input", "exponent"],
        "attribute_mapping": {},
    },
    "Sqrt": {
        "fx_op": "call_function",
        "target": torch.sqrt,
        "input_mapping": ["input"],
        "attribute_mapping": {},
    },
    "Div": {
        "fx_op": "call_function",
        "target": torch.div,
        "input_mapping": ["input", "other"],
        "attribute_mapping": {},
    },
    "MatMul": {
        "fx_op": "call_function",
        "target": torch.matmul,
        "input_mapping": ["input", "other"],
        "attribute_mapping": {},
    },
    "Transpose": {
        "name_override": "Permute",
        "fx_op": "call_function",
        "target": onnx_permute,
        "input_mapping": ["input"],
        "attribute_mapping": {"perm": "dims"},
        "attribute_transform": {"perm": lambda x: [i for i in x]},
        "attribute_default": {
            "perm": None,
        },
    },
    "Max": {
        "fx_op": "call_function",
        "target": torch.max,
        "input_mapping": ["input"],
        "attribute_mapping": {},
    },
    "Softmax": {
        "fx_op": "call_function",
        "target": torch.nn.functional.softmax,
        "input_mapping": ["input"],
        "attribute_mapping": {"axis": "dim"},
        "attribute_transform": {"axis": None},
    },
    "Relu": {
        "fx_op": "call_function",
        "target": torch.relu,
        "input_mapping": ["input"],
        "attribute_mapping": {},
    },
}
