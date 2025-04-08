import os, sys, logging, traceback, pdb

from chop.tools import get_logger
from chop.tools.logger import set_logging_verbosity
def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)
sys.excepthook = excepthook

logger = get_logger(__name__)
# Set global logging level to debug

from quant_aware_search import iterative_search
from pathlib import Path
checkpoint = "deit_base_patch16_224"
save_dir = "deit_base_saved_results"


# Override the DATASET_CACHE_DIR with our target directory
import chop.dataset
imagenet_dir = Path("/data/datasets/imagenet_pytorch/")
chop.dataset.DATASET_CACHE_DIR = imagenet_dir

exponent_width = 8
default_quant_config = {
    "quant_type": "mxint",
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
    "enable_internal_width": True,
}

def search_gelu(fixed_op, target_op):
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
                **fixed_op,
            }
        },
    }
    acc_list = iterative_search(
        checkpoint, 
        "gelu",
        {
            **target_op
        },
        quant_config, 
    )
    return acc_list

def search_layer_norm(fixed_op, target_op):
    # collected_results = {
    #     "enable_internal_width": True,
    #     "norm_in_int_width": 1,
    #     "norm_in_frac_width": 7,
    #     "norm_out_int_width": 5,
    #     "norm_out_frac_width": 6,
    #     "enable_mean_sqrt_noise": True,
    #     "var_int_width": 1,
    #     "var_frac_width": 13,
    #     "enable_mxint_var": True,
    #     "isqrt_in_width": 5,
    #     "isqrt_out_int_width": 2,
    #     "isqrt_out_frac_width": 6,
    #     ## Notice the result from 0.721 to 0.718 is due to the var_frac_width
    # }
    quant_config = {
        "by": "type",
        "layer_norm": {
            "config": {
                **default_quant_config,
                # **collected_results,
                "norm_in_int_width": 8,
                "norm_in_frac_width": 8,
                "norm_out_int_width": 8,
                "norm_out_frac_width": 8,
                "enable_internal_width": True,
                "enable_mean_sqrt_noise": True,
                "var_int_width": 4,
                "var_frac_width": 16,
                "enable_mxint_var": True,
                "isqrt_in_width": 16,
                "isqrt_out_int_width": 8,
                "isqrt_out_frac_width": 8,
                **fixed_op,
            }
        },
    }
    acc_list = iterative_search(
        checkpoint, 
        "layer_norm",
        {
            **target_op
        },
        quant_config, 
    )
    return acc_list

def search_softmax(fixed_op, target_op):
    quant_config = {
        "by": "type",
        "attention": {
            "config": {
                **default_quant_config,
                "enable_mxint_exp_sum": False,
                "enable_mxint_division": False,
                "enable_mxint_exp": True,
                "exp_width": 16,
                "exp_exponent_width": 16,
                "exp_sum_width": 16,
                "exp_sum_exponent_width": 16,
                "data_out_width": 16,
                "data_out_exponent_width": 16,
                **fixed_op,
            }
        },
    }
    acc_list = iterative_search(
        checkpoint, 
        "attention",
        {  
            **target_op
        },
        quant_config, 
    )
    return acc_list

default_int_quant_config = {
    "quant_type": "int",
    "in_width": 8,
    "in_range_momentum": 0.995,
    "out_width": 16,
}
def search_int(search_op):
    layer_norm_collected_results = {
        # lossless
        "in_width": 10,
        "in_range_momentum": 0.995,
    }
    softmax_collected_results = {
        # "in_width": 10,
        # "in_range_momentum": 0.995,
        # "out_width": 10,
    }
    gelu_collected_results = {
        # "in_width": 10,
        # "in_range_momentum": 0.995,
        # "out_width": 10,
    }
    collected_results = {
    }
    quant_config = {
        "by": "type",
        search_op: {
            "config": {
                **default_int_quant_config,
                **collected_results,
            }
        },
    }
    acc_list = iterative_search(
        checkpoint, 
        search_op,
        {
            "in_width": [8,9,10,11,12],
            "in_range_momentum": [0.95, 0.99, 0.995],
            "out_width": [8,9,10,11,12],
        },
        quant_config, 
    )

from utils import save_accuracy_list
def search_top(): 
    acc_list_all = []
    # Graph 1: GELU
    # 1. we need to use hash table to hash the total bit width
    # for int_width in [1,2,3,4]:
    #     fixed_op = {"bound": int_width}
    #     target_op = {"hash_bits": [1,2,3,4,5,6,7,8]}
    #     acc_list = search_gelu(fixed_op, target_op)
    #     acc_list_all.append(acc_list)
    # save_accuracy_list(acc_list_all, directory=save_dir, base_filename="gelu_mxint_clipping_search")

    for int_width in [3]:
        fixed_op = {"bound": int_width}
        target_op = {"hash_bits": [1,2,3,4,5,6,7,8]}
        acc_list = search_gelu(fixed_op, target_op)
        acc_list_all.append(acc_list)
    save_accuracy_list(acc_list_all, directory=save_dir, base_filename="gelu_mxint_search")

    # acc_list_all = []
    # for int_width in [1,2,3]:
    #     fixed_op = {"var_int_width": int_width}
    #     target_op = {"var_frac_width": [8,9,10,11,12,13,14,15]}
    #     acc_list = search_layer_norm(fixed_op, target_op)
    #     acc_list_all.append(acc_list)
    # save_accuracy_list(acc_list_all, directory=save_dir, base_filename="layer_norm_var_search")

    # acc_list_all = []
    
    # for int_width in [8]:
    #     fixed_op = {"isqrt_exponent_width": int_width}
    #     target_op = {"isqrt_in_width": [1,2,3,4,5,6,7,8]}
    #     acc_list = search_layer_norm(fixed_op, target_op)
    #     acc_list_all.append(acc_list)
    # save_accuracy_list(acc_list_all, directory=save_dir, base_filename="layer_norm_isqrt_search")

    # acc_list_all = []
    # for int_width in [8]:
    #     fixed_op = {"exp_exponent_width": int_width}
    #     target_op = {"exp_width": [1,2,3,4,5,6,7,8]}
    #     acc_list = search_softmax(fixed_op, target_op)
    #     acc_list_all.append(acc_list)
    # save_accuracy_list(acc_list_all, directory=save_dir, base_filename="attention_exp_search")

if __name__ == "__main__":
    set_logging_verbosity("info")  
    # acc_list_all = []
    # for int_width in [1,2,3,4,5]:
    #     acc_list = search_gelu(int_width=int_width)
    #     acc_list_all.append(acc_list)
    search_top()
    # search_layer_norm()
    # search_softmax()
    # search_int("layer_norm")
    # search_int("attention")
    # search_int("gelu")