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
    "t",
    "transpose",
    "reshape",
    "contiguous",
    # possibly should be built-in funcs
    "max",
    "softmax",
    "cumsum",
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
]

# use this list to print out MASE_MODULE_RELATED_FUNCS when new functions are added
# module_related_funcs = [
#     F.adaptive_avg_pool1d,
#     F.adaptive_avg_pool2d,
#     F.adaptive_max_pool1d,
#     F.adaptive_max_pool2d,
#     F.avg_pool1d,
#     F.avg_pool2d,
#     F.batch_norm,
#     F.conv1d,
#     F.conv2d,
#     F.layer_norm,
#     F.linear,
#     F.max_pool1d,
#     F.max_pool2d,
#     F.relu,
# ]
# print(sorted([f.__name__ for f in module_related_funcs]))

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
]


MASE_TYPE_MAP = {
    "adaptive_avg_pool1d": {"type": "module_related_func"},
    "adaptive_avg_pool2d": {"type": "module_related_func"},
    "adaptive_max_pool1d": {"type": "module_related_func"},
    "adaptive_max_pool2d": {"type": "module_related_func"},
    "avg_pool1d": {"type": "module_related_func"},
    "avg_pool2d": {"type": "module_related_func"},
    "batch_norm": {"type": "module_related_func"},
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
