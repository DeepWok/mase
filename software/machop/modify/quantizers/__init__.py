from .ops import LinearInteger, ReLUInteger, Conv2dInteger, AddInteger, LinearBlockFP
from .functions import integer_add

possible_ops = [LinearInteger, ReLUInteger, Conv2dInteger, AddInteger, LinearBlockFP]
possible_functions = [integer_add]


ops_map = {
    "linear": {
        "integer": LinearInteger,
        "blockfp": LinearBlockFP,
    },
    "relu": {
        "integer": ReLUInteger,
    },
    "conv2d": {
        "integer": Conv2dInteger,
    },
    "add": {
        "integer": AddInteger,
    },
}

functions_map = {
    "add": {
        "integer": integer_add,
    }
}