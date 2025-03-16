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
quant_config = iterative_search(
    checkpoint, 
    "gelu",
    {
        "enable_internal_width": [True],
        "hash_in_int_width": [2, 3, 4, 5, 6],
        "hash_in_frac_width": [6,7,8,9,10,11,12,13,14,15,16],
        "hash_out_int_width": [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
        "hash_out_frac_width": [8,9,10,11,12,13,14,15,16],
    }
)

