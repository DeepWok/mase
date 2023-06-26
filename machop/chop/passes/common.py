MASE_TYPES = [
    "module",
    "module_related_funcs",
    "builtin_funcs",
    "implicit_funcs",
    "placeholder",
    "get_attr",
    "output",
]


MASE_IMPLICIT_FUNCS = ["size", "view"]
MASE_MODULE_RELATED_FUNCS = ["relu"]
MASE_MODULE = ["linear", "conv2d", "conv1d"]
MASE_BUILDIN_FUNCS = ["mul", "sub", "add", "flatten"]


MASE_TYPE_MAP = {
    "linear": {
        "type": "module",
    },
    "conv2d": {
        "type": "module",
    },
    "relu": {
        "type": ("module", "module_related_funcs"),
    },
    "mul": {
        "type": "builtin_funcs",
    },
    "sub": {
        "type": "builtin_funcs",
    },
    "add": {
        "type": "builtin_funcs",
    },
    "size": {
        "type": "implicit_funcs",
    },
    "view": {
        "type": "implicit_funcs",
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
