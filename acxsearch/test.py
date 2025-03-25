
from a_cx_mxint_quant.module_level_tranform import vit_module_level_quantize
from utils import acc_cal

import os
import timm
import json
from utils import init_dataset
from chop.tools import get_logger
from chop.tools.logger import set_logging_verbosity

from a_cx_mxint_quant import DEIT_TINY_IMAGENET_ACC, DEIT_TINY_IMAGENET_ACC_100ITER

logger = get_logger(__name__)
set_logging_verbosity("debug")


from pathlib import Path
imagenet_dir = Path("/data/datasets/imagenet_pytorch/")

# Override the DATASET_CACHE_DIR with our target directory
import chop.dataset
chop.dataset.DATASET_CACHE_DIR = imagenet_dir

quant_config = {
    "by": "type",
    "gelu": {
        "config": {
            "quant_type": "mxint",
            "data_in_width": 8,
            "data_in_exponent_width": 8,
            "data_in_parallelism": (1, 32),
            "data_out_width": 8,
            "data_out_exponent_width": 8,
            "data_out_parallelism": (1, 32),
            "weight_width": 6,
            "weight_exponent_width": 8,
            "weight_parallelism": (1, 32), # Note: weight_parallelism is only used in layer_norm
            "bias_width": 6,
            "bias_exponent_width": 8,
            "bias_parallelism": (1, 32), # Note: bias_parallelism is only used in layer_norm
            "enable_internal_width": True,
            "hash_in_int_width": 3,
            "hash_in_frac_width": 1,
            "hash_out_int_width": 16,
            "hash_out_frac_width": 16,
        }
    }
}
checkpoint = "deit_tiny_patch16_224"
model = timm.create_model(checkpoint, pretrained=True)
datamodule = init_dataset("imagenet", 32, checkpoint)
qmodel = vit_module_level_quantize(model, quant_config)
acc = acc_cal(qmodel, datamodule.test_dataloader())
logger.info(f"original_acc: {DEIT_TINY_IMAGENET_ACC}, final_acc: {acc}")
