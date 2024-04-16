import torch

from chop.tools.onnx_operators import onnx_gemm, onnx_slice, onnx_unsqueeze, onnx_gather

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
    "Greater": {"fx_op": "call_function", "target": torch.gt, "attribute_mapping": []},
    "Abs": {"fx_op": "call_function", "target": torch.abs, "attribute_mapping": []},
    "Sin": {"fx_op": "call_function", "target": torch.sin, "attribute_mapping": []},
    "Cos": {"fx_op": "call_function", "target": torch.cos, "attribute_mapping": []},
    "Sigmoid": {
        "fx_op": "call_function",
        "target": torch.sigmoid,
        "attribute_mapping": [],
    },
    "ArgMax": {
        "fx_op": "call_function",
        "target": torch.argmax,
        "attribute_mapping": [],
    },
    "Flatten": {
        "fx_op": "call_function",
        "target": torch.flatten,
        "attribute_mapping": [],
    },
    "Split": {"fx_op": "call_function", "target": torch.split, "attribute_mapping": []},
    "Not": {
        "fx_op": "call_function",
        "target": torch.logical_not,
        "attribute_mapping": [],
    },
    "Identity": {
        "fx_op": "call_function",
        "target": torch.nn.Identity(),
        "attribute_mapping": [],
    },
    "Tile": {"fx_op": "call_function", "target": torch.tile, "attribute_mapping": []},
    "Erf": {
        "fx_op": "call_function",
        "target": torch.special.erf,
        "input_mapping": ["input"],
        "attribute_mapping": [],
    },
    "Tanh": {"fx_op": "call_function", "target": torch.tanh, "attribute_mapping": []},
    "LessOrEqual": {
        "fx_op": "call_function",
        "target": torch.le,
        "attribute_mapping": [],
    },
    "Min": {"fx_op": "call_function", "target": torch.min, "attribute_mapping": []},
    "Neg": {"fx_op": "call_function", "target": torch.neg, "attribute_mapping": []},
    "Log": {"fx_op": "call_function", "target": torch.log, "attribute_mapping": []},
    "Slice": {
        "fx_op": "call_function",
        "target": onnx_slice,
        "input_mapping": ["data", "starts", "ends", "axes", "steps"],
        "attribute_mapping": [],
    },
    "Gemm": {"fx_op": "call_function", "target": onnx_gemm, "attribute_mapping": []},
    "ConstantOfShape": {
        "fx_op": "call_function",
        "target": torch.full,
        "attribute_mapping": [],
    },
    "ReduceMean": {
        "fx_op": "call_function",
        "target": torch.mean,
        "input_mapping": ["input", "dim"],
        "attribute_mapping": ["", ""],
        "attribute_transform": [None, None],
        "attribute_default": [None, None],
    },
    "Expand": {"fx_op": "call_method", "target": "expand", "attribute_mapping": []},
    "Shape": {
        "fx_op": "call_method",
        "target": "size",
        "input_mapping": ["self"],
        "attribute_mapping": [],
        "attribute_transform": [],
    },
    "Cast": {
        "fx_op": "call_method",
        "target": "to",
        "input_mapping": ["self"],
        "attribute_mapping": ["dtype"],
        "attribute_transform": [onnx_to_torch_dtype],
    },
    "Gather": {
        "fx_op": "call_function",
        "target": onnx_gather,
        "input_mapping": ["input", "index"],
        "attribute_mapping": ["dim"],
        "attribute_transform": [None],
        "attribute_default": [0],
    },
    # Unsqueeze ONNX doc says "axes" is an input, but it's actually an attribute
    "Unsqueeze": {
        "fx_op": "call_function",
        "target": onnx_unsqueeze,
        "input_mapping": ["input"],
        "attribute_mapping": ["dim"],
        "attribute_transform": [lambda x: [i for i in x]],
    },
    "Concat": {
        "fx_op": "call_function",
        "target": torch.cat,
        "input_mapping": ["tensors"],
        "attribute_mapping": ["dim"],
        "attribute_transform": [None],
    },
    "Reshape": {
        "fx_op": "call_function",
        "target": torch.reshape,
        "input_mapping": ["input", "shape"],
        "attribute_mapping": [""],
        "attribute_transform": [None],
    },
    "Add": {
        "fx_op": "call_function",
        "target": torch.add,
        "input_mapping": ["input", "other"],
        "attribute_mapping": [],
    },
    "Sub": {
        "fx_op": "call_function",
        "target": torch.sub,
        "input_mapping": ["input", "other"],
        "attribute_mapping": [],
    },
    "Squeeze": {
        "fx_op": "call_function",
        "target": torch.squeeze,
        "input_mapping": ["input", "dim"],
        "attribute_mapping": [],
    },
    "Range": {
        "fx_op": "call_function",
        "target": torch.arange,
        "input_mapping": ["start", "end", "step"],
        "attribute_mapping": [],
    },
    "Less": {
        "fx_op": "call_function",
        "target": torch.lt,
        "input_mapping": ["input", "other"],
        "attribute_mapping": [],
    },
    "Where": {
        "fx_op": "call_function",
        "target": torch.where,
        "input_mapping": ["condition", "input", "other"],
        "attribute_mapping": [],
    },
    "Mul": {
        "fx_op": "call_function",
        "target": torch.mul,
        "input_mapping": ["input", "other"],
        "attribute_mapping": [],
    },
    "Equal": {
        "fx_op": "call_function",
        "target": torch.eq,
        "input_mapping": ["input", "other"],
        "attribute_mapping": [],
    },
    "CumSum": {
        "fx_op": "call_function",
        "target": torch.cumsum,
        "input_mapping": ["input", "other"],
        "attribute_mapping": [],
    },
    "Pow": {
        "fx_op": "call_function",
        "target": torch.pow,
        "input_mapping": ["input", "exponent"],
        "attribute_mapping": [],
    },
    "Sqrt": {
        "fx_op": "call_function",
        "target": torch.sqrt,
        "input_mapping": ["input"],
        "attribute_mapping": [],
    },
    "Div": {
        "fx_op": "call_function",
        "target": torch.div,
        "input_mapping": ["input", "other"],
        "attribute_mapping": [],
    },
    "MatMul": {
        "fx_op": "call_function",
        "target": torch.matmul,
        "input_mapping": ["input", "other"],
        "attribute_mapping": [],
    },
    "Transpose": {
        "fx_op": "call_function",
        "target": torch.permute,
        "name_override": "Permute",
        "attribute_mapping": ["dims"],
        "attribute_transform": [lambda x: [i for i in x]],
        "input_mapping": ["input"],
    },
    "Max": {
        "fx_op": "call_function",
        "target": torch.max,
        "input_mapping": ["input"],
        "attribute_mapping": [],
    },
    "Softmax": {
        "fx_op": "call_function",
        "target": torch.nn.functional.softmax,
        "input_mapping": ["input"],
        "attribute_mapping": ["dim"],
        "attribute_transform": [None],
    },
    "Relu": {
        "fx_op": "call_function",
        "target": torch.relu,
        "input_mapping": ["input"],
        "attribute_mapping": [],
    },
}
