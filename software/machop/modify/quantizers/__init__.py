from .ops import (
    LinearInteger, ReLUInteger, Conv2dInteger, 
    LinearBlockFP)

possible_ops = [LinearInteger, ReLUInteger, Conv2dInteger]

ops_map = {
    'linear': {
        'integer': LinearInteger,
        'blockfp': LinearBlockFP,
    },
    'relu': {
        'integer': ReLUInteger,
    },
    'conv2d': {
        'integer': Conv2dInteger,
    },
}