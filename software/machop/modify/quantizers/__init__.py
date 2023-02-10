from .ops import LinearInteger, ReLUInteger

ops_map = {
    'linear': {
        'integer': LinearInteger,
    },
    'relu': {
        'integer': ReLUInteger,
    }
}