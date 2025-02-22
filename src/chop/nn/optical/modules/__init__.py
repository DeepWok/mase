from .morr_linear import AllPassMORRCirculantLinear
from .morr_conv2d import AllPassMORRCirculantConv2d

optical_module_map = {
    "linear_morr": AllPassMORRCirculantLinear,
    "conv2d_morr": AllPassMORRCirculantConv2d,
}
