"""
Some of the functions here are taken from the Quantizer Modifier File we had before
"""

from functools import partial
from typing import Dict

import torch
from chop.tools.utils import copy_weights
from torch import nn

from chop.nn.backward.modules import custom_module_map
from chop.nn.quantized.functional import quantized_func_map
from chop.nn.backward.functional import quantized_grad_func_map


SUPPORTED_FORWARD_TRANSFORMS = {"quantize": quantized_func_map}
SUPPORTED_BACKWARD_TRANSFORMS = {"quantize": quantized_grad_func_map}


def attach_forward_fn(q_fn: torch.autograd.Function, mase_op: str, q_fn_cfg: dict):
    """
    Attach a forward function
    """
    name = q_fn_cfg["name"]
    pass_name = q_fn_cfg["pass"]
    bypass = q_fn_cfg.get("bypass", False)
    if bypass:
        return
    forward_fn_name = f"{mase_op}_{name}"

    if pass_name in SUPPORTED_FORWARD_TRANSFORMS:
        if forward_fn_name not in SUPPORTED_FORWARD_TRANSFORMS[pass_name]:
            raise ValueError(
                f"Forward function {forward_fn_name} not recognized in pass {pass_name}"
            )
        forward_fn = SUPPORTED_FORWARD_TRANSFORMS[pass_name][forward_fn_name]
    else:
        raise ValueError(f"Transform pass {pass_name} not recognized")

    q_fn.forward = partial(forward_fn, config=q_fn_cfg)


def attach_backward_fn(q_fn: torch.autograd.Function, mase_op: str, q_fn_cfg: dict):
    """
    Attach a custom backward function
    """
    name = q_fn_cfg["name"]
    pass_name = q_fn_cfg["pass"]
    backward_fn_name = f"{mase_op}_{name}"
    bypass = q_fn_cfg.get("bypass", False)
    if bypass:
        return

    if pass_name in SUPPORTED_BACKWARD_TRANSFORMS:
        if backward_fn_name not in SUPPORTED_BACKWARD_TRANSFORMS[pass_name]:
            raise ValueError(
                f"Backward function {backward_fn_name} not recognized in pass {pass_name}"
            )
        backward_fn = SUPPORTED_BACKWARD_TRANSFORMS[pass_name][backward_fn_name]
    else:
        raise ValueError(f"Transform pass {pass_name} not recognized")

    q_fn.backward = partial(backward_fn, config=q_fn_cfg)


def create_new_module(
    mase_op: str,
    original_module: nn.Module,
    config: dict,
    node_meta: dict,
):
    original_module_cls = type(original_module)

    if mase_op == "linear":
        new_module_cls = custom_module_map["linear"]
        use_bias = original_module.bias is not None
        new_module = new_module_cls(
            in_features=original_module.in_features,
            out_features=original_module.out_features,
            bias=use_bias,
            config=config,
        )
        copy_weights(original_module.weight, new_module.weight)
        if use_bias:
            copy_weights(original_module.bias, new_module.bias)
        # Edit forward function
        attach_forward_fn(new_module.linear_autograd_fn, mase_op, config["forward"])
        # Edit backward function
        attach_backward_fn(new_module.linear_autograd_fn, mase_op, config["backward"])
    else:
        raise NotImplementedError(
            f"Unsupported module class {original_module_cls} to modify"
        )
    return new_module
