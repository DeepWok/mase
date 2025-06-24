from .morr_linear import AllPassMORRCirculantLinear
from .morr_conv2d import AllPassMORRCirculantConv2d

# from ..triton_modules.morr_linear_mem import TritonMemMORRLinear


optical_module_map = {
    "linear_morr": AllPassMORRCirculantLinear,
    "conv2d_morr": AllPassMORRCirculantConv2d,
    # "linear_morr_triton": TritonMemMORRLinear,
}
