from .ops import LinearInteger, ReLUInteger, Conv2dInteger

possible_ops = [LinearInteger, ReLUInteger, Conv2dInteger]

ops_map = {
    'linear': {
        'integer': LinearInteger,
    },
    'relu': {
        'integer': ReLUInteger,
    },
    'conv2d': {
        'integer': Conv2dInteger,
    },
}
