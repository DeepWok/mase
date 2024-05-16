from .modules import (
    quantized_module_map,
    BertSelfAttentionInteger,
    BertSelfAttentionHeadInteger,
)
from .functional import quantized_func_map, fixed_softermax
