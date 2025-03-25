import os, sys, logging, traceback, pdb

from chop.tools import get_logger
from chop.tools.logger import set_logging_verbosity
def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)
sys.excepthook = excepthook

logger = get_logger(__name__)
set_logging_verbosity("info")

exponent_width = 8

logger = get_logger(__name__)
set_logging_verbosity("info")

from quant_aware_search import iterative_search

from a_cx_mxint_quant.module_level_tranform import vit_module_level_quantize
from utils import acc_cal

import timm
import json
from utils import init_dataset
from chop.tools import get_logger
from chop.tools.logger import set_logging_verbosity
from pathlib import Path
imagenet_dir = Path("/data/datasets/imagenet_pytorch/")

# Override the DATASET_CACHE_DIR with our target directory
import chop.dataset
chop.dataset.DATASET_CACHE_DIR = imagenet_dir
checkpoint = "deit_tiny_patch16_224"
default_quant_config = {
    "data_in_width": 8,
    "data_in_exponent_width": exponent_width,
    "data_in_parallelism": (1, 32),
    "data_out_width": 8,
    "data_out_exponent_width": exponent_width,
    "data_out_parallelism": (1, 32),
    "weight_width": 6,
    "weight_exponent_width": exponent_width,
    "weight_parallelism": (1, 32), # Note: weight_parallelism is only used in layer_norm
    "bias_width": 6,
    "bias_exponent_width": exponent_width,
    "bias_parallelism": (1, 32), # Note: bias_parallelism is only used in layer_norm
    "enable_internal_width": False,
}
quant_config = {
    "by": "type",
    "gelu": {
        "config": {
            **default_quant_config,
            # "enable_mxint_softmax": True,
            # "enable_mxint_exp": True,
            # "exp_width": 2,
            # "enable_mxint_exp_sum": False,
            # "enable_mxint_division": False,
        }
    }
}
def main():
    model = timm.create_model(checkpoint, pretrained=True)
    datamodule = init_dataset("imagenet", 32, checkpoint)
    vit_module_level_quantize(model, quant_config)
    acc = acc_cal(model, datamodule.test_dataloader())
    logger.info(f"Accuracy: {acc}")
if __name__ == "__main__":
    main()