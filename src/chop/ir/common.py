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
    # Memory ops and tensor reshapes
    "to",
    "bool",
    "int",
    "flatten",
    "squeeze",
    "unsqueeze",
    "transpose",
    "permute",
    "reshape",
    "contiguous",
    "dropout",
    "eq",
    "ne",
    "gemm",
    "ge",
    "where",
    "_assert",
    "getattr",
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
    "split",
    "tile",
    "expand",
    "full",
    "ones",
    "dim",
    "finfo",
    "masked_fill",
    "masked_fill_",
    "index_select",
    "detach",
    "tensor",
    "cumsum",  # Added for Wave2Vec
    "flip",  # Added for Wave2Vec
    "repeat",  # Added for Wave2Vec
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
    "identity",
    "hardshrink",
    "silu",
    "elu",
    "sigmoid",
    "softshrink",
    "logsigmoid",
    "softmax",
    "selu",
    "tanh",
    "gelu",
    "softplus",
    "softsign",
]

MASE_MODULES = [
    "batch_norm1d",
    "batch_norm2d",
]

MASE_BUILTIN_FUNCS = [
    "mul",
    "baddbmm",
    "addmm",
    "sub",
    "add",
    "matmul",
    "bmm",
    "mean",
    "pow",
    "sqrt",
    "div",
    "softmax",
    "max",
    "cumsum",
    "erf",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "greater",
    "gt",
    "less",
    "le",  # less or equal
    "lt",
    "sigmoid",
    "not",
    "min",
    "neg",
    "log",
    "arange",
    "range",
    "gelu",
    "scaled_dot_product_attention",
    "embedding",
    "df_split",
    "getitem",
    "zeros",  # Added for Wave2Vec
    "setitem",  # Added for Wave2Vec
    "invert",  # Added for Wave2Vec
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
    "hardshrink": {"type": "module_related_func"},
    "silu": {"type": "module_related_func"},
    "elu": {"type": "module_related_func"},
    "sigmoid": {"type": "module_related_func"},
    "softshrink": {"type": "module_related_func"},
    "logsigmoid": {"type": "module_related_func"},
    "softmax": {"type": "module_related_func"},
    "selu": {"type": "module_related_func"},
    "tanh": {"type": "module_related_func"},
    "gelu": {"type": "module_related_func"},
    "softsign": {"type": "module_related_func"},
    "softplus": {"type": "module_related_func"},
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
