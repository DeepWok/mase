from typing import Dict, Tuple

from .functions import integer_add
from .ops import AddInteger, Conv2dInteger, LinearInteger, LinearMSFP, ReLUInteger

possible_ops = [LinearInteger, ReLUInteger, Conv2dInteger, AddInteger, LinearMSFP]
possible_functions = [integer_add]


ops_map = {
    "linear": {
        "integer": LinearInteger,
        "msfp": LinearMSFP,
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
