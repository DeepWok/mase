from .block_fp import block_fp_quantizer
from .block_log import block_log_quantizer
from .block_minifloat import block_minifloat_quantizer
from .integer import integer_quantizer, integer_floor_quantizer
from .binary import binary_quantizer, residual_sign_quantizer
from .ternary import ternary_quantizer
from .log import log_quantizer
from .minifloat import minifloat_denorm_quantizer, minifloat_ieee_quantizer
from .quantizers_for_hw import integer_quantizer_for_hw

quantizer_map = {
    "log": log_quantizer,
    "block_log": block_log_quantizer,
    "minifloat_denorm": minifloat_denorm_quantizer,
    "minifloat_ieee": minifloat_ieee_quantizer,
    "block_minifloat": block_minifloat_quantizer,
    "block_fp": block_fp_quantizer,
    "integer": integer_quantizer,
    "binary": binary_quantizer,
    "ternary": ternary_quantizer,
}
