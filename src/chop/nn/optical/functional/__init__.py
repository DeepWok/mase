from .mrr import (
    MORRConfig_20um_MQ,
    MRRConfig_5um_HQ,
    MRRConfig_5um_MQ,
    MRRConfig_5um_LQ,
    MORRConfig_10um_MQ,
)

from .compute import (
    im2col_2d,
    toeplitz,
)

from .general import (
    logger,
)

from .initializer import (
    morr_uniform_,
)

from .quantize import (
    input_quantize_fn,
    weight_quantize_fn,
)

from .mrr_op import (
    mrr_roundtrip_phase_to_tr_func,
    mrr_roundtrip_phase_to_tr_fused,
)





# """
# Description:
# Author: Jiaqi Gu (jqgu@utexas.edu)
# Date: 2021-06-09 01:40:22
# LastEditors: Jiaqi Gu (jqgu@utexas.edu)
# LastEditTime: 2021-06-09 01:40:22
# """

# import importlib
# import os

# # automatically import any Python files in this directory
# for file in sorted(os.listdir(os.path.dirname(__file__))):
#     if file.endswith(".py") and not file.startswith("_"):
#         source = file[: file.find(".py")]
#         module = importlib.import_module("torchonn.layers." + source)
#         if "__all__" in module.__dict__:
#             names = module.__dict__["__all__"]
#         else:
#             # import all names that do not begin with _
#             names = [x for x in module.__dict__ if not x.startswith("_")]
#         globals().update({k: getattr(module, k) for k in names})
