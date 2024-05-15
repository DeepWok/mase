from .mase_op_wrapper import torch_arange, torch_ones, torch_zeros
from ..transforms.quantize import quantized_module_map, quantized_func_map

MASE_LEAF_FUNCTIONS = (
    # tensor constructors
    torch_arange,
    torch_ones,
    torch_zeros,
)  # + tuple(quantized_func_map.keys()) # add this if there is a case where quantized module is traced again

MASE_LEAF_LAYERS = () + tuple(
    quantized_module_map.values()
)  # add this if there is a case where quantized module is traced again
