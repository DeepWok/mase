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

checkpoint = "deit_tiny_patch16_224"

def search_gelu():
    exponent_width = 8
    quant_config = {
        "by": "type",
        "gelu": {
            "config": {
                "data_in_width": 8,
                "data_in_exponent_width": exponent_width,
                "data_in_parallelism": (1, 32),
                "data_out_width": 8,
                "data_out_exponent_width": exponent_width,
                "data_out_parallelism": (1, 32),
                "enable_internal_width": True,
                "hash_in_int_width": 16,
                "hash_in_frac_width": 16,
                "hash_out_int_width": 16,
                "hash_out_frac_width": 16,
                "hash_in_int_width": 16,
                "hash_in_frac_width": 16,
                "hash_out_int_width": 16,
                "hash_out_frac_width": 16,
            }
        },
    }
    _, = iterative_search(
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

# def search_softmax(quant_config):
#     quant_config["softmax"]["config"]["enable_internal_width"] = True
#     _, = iterative_search(
#         checkpoint, 
#         "softmax",

if __name__ == "__main__":
    search_gelu()