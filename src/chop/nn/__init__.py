from .quantized import quantized_module_map
from .attention import attention_module_map

MASE_LEAF_LAYERS = tuple(quantized_module_map.values())
