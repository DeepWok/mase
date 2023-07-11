MASE_TYPES = [
    "module",
    "module_related_func",
    "builtin_func",
    "implicit_func",
    "placeholder",
    "get_attr",
    "output",
]


MASE_IMPLICIT_FUNCS = ["size", "view", "flatten"]
MASE_MODULE_RELATED_FUNCS = ["relu", "linear"]
MASE_MODULES = [
    "adaptiveavgpool1d",
    "adaptiveavgpool2d",
    "adaptivemaxpool1d",
    "adaptivemaxpool2d",
    "avgpool1d",
    "avgpool2d",
    "batchnorm1d",
    "batchnorm2d",
    "conv1d",
    "conv2d",
    "layernorm",
    "linear",
    "maxpool1d",
    "maxpool2d",
    "relu",
]
MASE_BUILTIN_FUNCS = [
    "mul",
    "sub",
    "add",
    "matmul",
    "bmm",
]


MASE_TYPE_MAP = {
    "linear": {
        "type": "module",
    },
    "conv2d": {
        "type": "module",
    },
    "relu": {
        "type": ("module", "module_related_func"),
    },
    "mul": {
        "type": "builtin_func",
    },
    "sub": {
        "type": "builtin_func",
    },
    "add": {
        "type": "builtin_func",
    },
    "size": {
        "type": "implicit_func",
    },
    "view": {
        "type": "implicit_func",
    },
    "placeholder": {
        "type": "placeholder",
    },
    "get_attr": {
        "type": "get_attr",
    },
    "output": {
        "type": "output",
    },
}

MASE_HARDWARE_TOOLCHAIN = [
    "INTERNAL_RTL",
    "EXTERNAL_RTL",
    "INTERNAL_HLS",
    "EXTERNAL_HLS",
    "MLIR_HLS",
]
