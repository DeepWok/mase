import torch.nn.functional as F

MASE_TYPES = [
    "module",
    "module_related_func",
    "builtin_func",
    "implicit_func",
    "placeholder",
    "get_attr",
    "output",
]


MASE_IMPLICIT_FUNCS = [
    # possibly are just constants
    "size",
    "view",
    # possibly are just memory ops or tensor reshapes
    "flatten",
    "unsqueeze",
    "transpose",
    "permute",
    "reshape",
    "contiguous",
    # possibly should be built-in funcs
    # softmax should be listed before max
    "softmax",
    "max",
    "cumsum",
    "erf",
    # possibly can just safely ignore?
    "dropout",
    "eq",
    "ge",
    "where",
    "_assert",
    "getattr",
    "getitem",
    "long",
    "type_as",
    "clamp",
    "abs",
    "stack",
    "cast",
    "shape",
    "gather",
    "slice",
    "cat",
]

MASE_MODULE_RELATED_FUNCS = [
    "adaptive_avg_pool1d",
    "adaptive_avg_pool2d",
    "adaptive_max_pool1d",
    "adaptive_max_pool2d",
    "avg_pool1d",
    "avg_pool2d",
    "batch_norm",
    "conv1d",
    "conv2d",
    "layer_norm",
    "linear",
    "max_pool1d",
    "max_pool2d",
    "relu",
]

MASE_MODULES = [
    "batch_norm1d",
    "batch_norm2d",
]
MASE_BUILTIN_FUNCS = [
    "mul",
    "sub",
    "add",
    "matmul",
    "bmm",
    "mean",
    "pow",
    "sqrt",
    "div",
]


MASE_TYPE_MAP = {
    "adaptive_avg_pool1d": {"type": "module_related_func"},
    "adaptive_avg_pool2d": {"type": "module_related_func"},
    "adaptive_max_pool1d": {"type": "module_related_func"},
    "adaptive_max_pool2d": {"type": "module_related_func"},
    "avg_pool1d": {"type": "module_related_func"},
    "avg_pool2d": {"type": "module_related_func"},
    "batch_norm": {"type": "module_related_func"},
    "batch_norm1d": {"type": "module"},
    "batch_norm2d": {"type": "module"},
    "conv1d": {"type": "module_related_func"},
    "conv2d": {"type": "module_related_func"},
    "layer_norm": {"type": "module_related_func"},
    "linear": {"type": "module_related_func"},
    "max_pool1d": {"type": "module_related_func"},
    "max_pool2d": {"type": "module_related_func"},
    "relu": {"type": "module_related_func"},
    "sub": {"type": "builtin_func"},
    "add": {"type": "builtin_func"},
    "size": {"type": "implicit_func"},
    "view": {"type": "implicit_func"},
    "placeholder": {"type": "placeholder"},
    "get_attr": {"type": "get_attr"},
    "output": {"type": "output"},
}

MASE_HARDWARE_TOOLCHAIN = [
    "INTERNAL_RTL",
    "EXTERNAL_RTL",
    "INTERNAL_HLS",
    "EXTERNAL_HLS",
    "MLIR_HLS",
]
