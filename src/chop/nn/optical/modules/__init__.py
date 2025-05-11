from .morr_linear import AllPassMORRCirculantLinear
from .morr_conv2d import AllPassMORRCirculantConv2d
from .morr_custom_linear import AllPassMORRLinear
from ..triton_modules.morr_linear import TritonMORRLinear
from ..triton_modules.morr_linear_mem import TritonMemMORRLinear
from .morr_transformer.morr_bert import BertMORRSelfAttention


optical_module_map = {
    "linear_morr": AllPassMORRCirculantLinear,
    "conv2d_morr": AllPassMORRCirculantConv2d,
    "linear_morr_full": AllPassMORRLinear,
    "linear_morr_triton": TritonMORRLinear,
    "linear_morr_triton_mem": TritonMemMORRLinear,
    "bert_self_attention_morr": BertMORRSelfAttention,
}
