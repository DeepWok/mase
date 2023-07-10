import logging
import math

import torch
from torch.fx import GraphModule, Tracer
from torch.fx import wrap as fx_wrap

logger = logging.getLogger(__name__)

# ----------------------------------------
# Currently tensor constructors cannot be traced.
# see https://pytorch.org/docs/stable/fx.html#miscellanea
# ----------------------------------------

# def mark_as_tensor_constructor(func):
#     global CUSTOM_LEAF_FUNCTIONS
#     global MY_TENSOR_CONSTRUCTORS
#     if func in CUSTOM_LEAF_FUNCTIONS:
#         logger.warning(f"Function {func} was already marked as leaf function")
#     else:
#         MY_TENSOR_CONSTRUCTORS.append(func)
#     return func


# @mark_as_tensor_constructor
def torch_zeros(*args, **kwargs):
    return torch.zeros(*args, **kwargs)


# @mark_as_tensor_constructor
def torch_ones(*args, **kwargs):
    return torch.ones(*args, **kwargs)


# @mark_as_tensor_constructor
def torch_arange(*args, **kwargs):
    return torch.arange(*args, **kwargs)
