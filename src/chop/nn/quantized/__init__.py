from .modules import (
    quantized_module_map,
    BertSelfAttentionInteger,
    ViTSelfAttentionInteger,
    BertSelfAttentionHeadInteger,
    ViTSelfAttentionHeadInteger,
    LinearInteger,
    LayerNormInteger,
    GELUInteger,
    SiLUInteger,
    RMSNormInteger,
)
from .functional import quantized_func_map, fixed_softermax
