from .ops import LinearInteger, ReLUInteger, Conv2dInteger, AddInteger, LinearBlockFP

possible_ops = [LinearInteger, ReLUInteger, Conv2dInteger, AddInteger, LinearBlockFP]

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
