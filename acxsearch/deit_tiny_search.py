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

from quant_aware_search import iterative_search
from pathlib import Path
checkpoint = "deit_tiny_patch16_224"

imagenet_dir = Path("/data/datasets/imagenet_pytorch/")

# Override the DATASET_CACHE_DIR with our target directory
import chop.dataset
chop.dataset.DATASET_CACHE_DIR = imagenet_dir

exponent_width = 8
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

def search_gelu():
    quant_config = {
        "by": "type",
        "gelu": {
            "config": {
                **default_quant_config,
                "enable_internal_width": True,
                "hash_in_int_width": 16,
                "hash_in_frac_width": 16,
                "hash_out_int_width": 16,
                "hash_out_frac_width": 16,
            }
        },
    }
    collected_results = {
        "hash_in_int_width": 4,
        "hash_in_frac_width": 3,
        "hash_out_int_width": 3,
        "hash_out_frac_width": 13,
    }
    _ = iterative_search(
        checkpoint, 
        "gelu",
        {
        "hash_in_int_width": [2, 3, 4, 5, 6],
        "hash_in_frac_width": [3,4,5,6,7],
        "hash_out_int_width": [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
        "hash_out_frac_width": [3,4,5,6,7,8,9,10,11,12,13,14,15,16],
        },
        quant_config, 
    )

def search_layer_norm():
    collected_results = {
        "enable_internal_width": True,
        "norm_in_int_width": 1,
        "norm_in_frac_width": 7,
        "norm_out_int_width": 5,
        "norm_out_frac_width": 6,
        "enable_mean_sqrt_noise": True,
        "var_int_width": 1,
        "var_frac_width": 13,
        "enable_mxint_var": True,
        "isqrt_in_width": 5,
        "isqrt_out_int_width": 2,
        "isqrt_out_frac_width": 6,
        ## Notice the result from 0.721 to 0.718 is due to the var_frac_width
    }
    quant_config = {
        "by": "type",
        "layer_norm": {
            "config": {
                **default_quant_config,
                **collected_results,
                "enable_mxint_var": True,
            }
        },
    }
    _ = iterative_search(
        checkpoint, 
        "layer_norm",
        {
            # "enable_internal_width": [True],
            # "enable_mean_sqrt_noise": [True, False],
            # "enable_mxint_var": [False],
            # "var_int_width": [1,2,3,4,5,6,7,8],
            # "var_frac_width": [11,12,13,14,15,16],
            "isqrt_in_width": [1,2,3,4,5,6,7,8],
            "isqrt_out_int_width": [1,2,3,4,5,6,7,8],
            "isqrt_out_frac_width": [1,2,3,4,5,6,7,8],
        },
        quant_config, 
    )

def search_softmax():
    collected_results = {
        "enable_mxint_softmax": True,
        "enable_mxint_exp": True,
        "exp_width": 2, #if search the best result will be at 6 bit, but won't be a big difference, 2 bit can keep the loss < 0.3 %
        "enable_mxint_exp_sum": False,
        "enable_mxint_division": False,
    }
    quant_config = {
        "by": "type",
        "attention": {
            "config": {
                **default_quant_config,
                **collected_results,
                "enable_mxint_exp_sum": False,
                "enable_mxint_division": False,
                "exp_exponent_width": 8,
                "exp_sum_width": 8,
                "exp_sum_exponent_width": 8,
                "data_out_width": 8,
                "data_out_exponent_width": 8,
            }
        },
    }
    _ = iterative_search(
        checkpoint, 
        "attention",
        {
            "enable_mxint_softmax": [True],
            # "exp_width": [1,2,3,4,5,6,7,8],
            # "exp_exponent_width": [1,2,3,4,5,6,7,8],
            # "exp_sum_width": [1,2,3,4,5,6,7,8],
            # "exp_sum_exponent_width": [1,2,3,4,5,6,7,8],
            # "data_out_width": [1,2,3,4,5,6,7,8],
            # "data_out_exponent_width": [1,2,3,4,5,6,7,8],
        },
        quant_config, 
    )


if __name__ == "__main__":
    # search_gelu()
    # search_layer_norm()
    search_softmax()