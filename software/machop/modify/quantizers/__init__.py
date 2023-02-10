from .ops import LinearInteger, ReLUInteger

possible_ops = [LinearInteger, ReLUInteger]

ops_map = {
    'linear': {
        'integer': LinearInteger,
    },
    'relu': {
        'integer': ReLUInteger,
    }
}