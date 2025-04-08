import os, sys, logging, traceback, pdb

from chop.tools import get_logger
from chop.tools.logger import set_logging_verbosity
def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)
sys.excepthook = excepthook

from a_cx_mxint_quant.module_level_tranform import vit_module_level_quantize
from utils import acc_cal

import os
import timm
import json
from utils import init_dataset
from chop.tools import get_logger
from chop.tools.logger import set_logging_verbosity

from a_cx_mxint_quant import DEIT_TINY_IMAGENET_ACC, DEIT_TINY_IMAGENET_ACC_100ITER
from pathlib import Path
import chop.dataset
imagenet_dir = Path("/data/datasets/imagenet_pytorch/")
chop.dataset.DATASET_CACHE_DIR = imagenet_dir
logger = get_logger(__name__)
set_logging_verbosity("info")

exponent_width = 8

logger = get_logger(__name__)
set_logging_verbosity("info")

from quant_aware_search import iterative_search

checkpoint = "deit_tiny_patch16_224"

# quant_config = {
#     "by": "type",
#     "layer_norm": {
#         "config": {
#             "quant_type": "mxint",
#             "data_in_width": 8,
#             "data_in_exponent_width": 8,
#             "data_in_parallelism": [1,32],
#             "weight_width": 8,
#             "weight_exponent_width": 8,
#             "weight_parallelism": [1,32],
#             "bias_width": 8,
#             "bias_exponent_width": 8,
#             "bias_parallelism": [1,32],
#             "data_out_width": 8,
#             "data_out_exponent_width": 8,
#             "data_out_parallelism": [1,32],
#             "enable_internal_width": True,
#             "norm_in_int_width": 8,
#             "norm_in_frac_width": 8,
#             "norm_out_int_width": 8,
#             "norm_out_frac_width": 8,
#             "enable_mxint_var": False,
#         }
#     }
# }
quant_config = {
    "by": "type",
    "gelu": {
        "config": {
            "quant_type": "mxint",
            "data_in_width": 8,
            "data_in_exponent_width": 8,
            "data_in_parallelism": [1,32],
            "weight_width": 8,
            "weight_exponent_width": 8,
            "weight_parallelism": [1,32],
            "bias_width": 8,
            "bias_exponent_width": 8,
            "bias_parallelism": [1,32],
            "data_out_width": 8,
            "data_out_exponent_width": 8,
            "data_out_parallelism": [1,32],
            "enable_internal_width": False
            # "norm_in_int_width": 8,
            # "norm_in_frac_width": 8,
            # "norm_out_int_width": 8,
            # "norm_out_frac_width": 8,
            # "enable_mxint_var": False,
        }
    }
}
import timm

model = timm.create_model(checkpoint, pretrained=True)
datamodule = init_dataset("imagenet", 32, checkpoint)
# loop through search_args
quant_acc = DEIT_TINY_IMAGENET_ACC_100ITER
# Check if search results file already exists
# quant_type = quant_config["att"]["config"]["quant_type"]
qmodel = vit_module_level_quantize(model, quant_config)
acc = acc_cal(qmodel, datamodule.test_dataloader())