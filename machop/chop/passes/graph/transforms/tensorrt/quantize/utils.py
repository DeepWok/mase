import os
from datetime import datetime as dt
from glob import glob
from copy import copy, deepcopy
import logging
import numpy as np
import pytorch_quantization.calib as calib
import pytorch_quantization.nn as qnn
import tensorrt as trt
import torch as t
import torch.nn.functional as F
from cuda import cudart
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from torch.autograd import Variable            
import torch
from typing import Dict
from chop.tools.utils import copy_weights, init_LinearLUT_weight, init_Conv2dLUT_weight
from torch import nn

from ....utils import (
    get_mase_op,
    get_mase_type,
    get_node_actual_target,
    get_parent_name,
    get_similar_node_actual_target,
    match_a_pattern,
    get_node_target_by_name,
)

QUANTIZEABLE_OP = (
    # "add",
    # "bmm",
    # "conv1d",
    "conv2d",
    # "matmul",
    # "mul",
    "linear",
    # "relu",
    # "sub",
)

class FakeQuantizer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create_quantized_module(self,
        mase_op: str,
        original_module: nn.Module,
        config: dict
    ):
        original_module_cls = type(original_module)
        #TODO implement more module support: https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/index.html#quantized-modules
        try:
            if mase_op == "linear":
                use_bias = original_module.bias is not None

                new_module = qnn.QuantLinear(
                    in_features=original_module.in_features,
                    out_features=original_module.out_features,
                    bias=use_bias
                )
                new_module.set_default_quant_desc_input(QuantDescriptor(calib_method=config["input"]["calibrator"], axis=config["input"]["quantize_axis"]))
                new_module.set_default_quant_desc_weight(QuantDescriptor(calib_method=config["weight"]["calibrator"], axis=config["weight"]["quantize_axis"]))

                copy_weights(original_module.weight, new_module.weight)
                if use_bias:
                    copy_weights(original_module.bias, new_module.bias)
        
            elif mase_op in ("conv2d"):
                use_bias = original_module.bias is not None
                new_module = qnn.QuantConv2d(
                    in_channels=original_module.in_channels,
                    out_channels=original_module.out_channels,
                    kernel_size=original_module.kernel_size,
                    stride=original_module.stride,
                    padding=original_module.padding,
                    dilation=original_module.dilation,
                    groups=original_module.groups,
                    bias=use_bias,
                    padding_mode=original_module.padding_mode,
                )

                new_module.set_default_quant_desc_input(QuantDescriptor(calib_method=config["input"]["calibrator"], axis=config["input"]["quantize_axis"]))
                new_module.set_default_quant_desc_weight(QuantDescriptor(calib_method=config["weight"]["calibrator"], axis=config["weight"]["quantize_axis"]))

                copy_weights(original_module.weight, new_module.weight)
                if use_bias:    
                    copy_weights(original_module.bias, new_module.bias)

            else:
                raise NotImplementedError(
                    f"Unsupported module class {original_module_cls} to modify"
                )
            
        except KeyError:
            raise Exception(f"Config/TOML not configured correctly for layer {original_module_cls}. Please check documentation for what must be defined.")
        
        return new_module
    
    def get_config(self, name: str):
        """Retrieve specific configuration from the instance's config dictionary or return default."""
        try:
            config = self.config.get(name, 'default')
        except KeyError:
            raise Exception(f"Please check Config/TOML file. Default config must be defined.")
        return config

    def fake_quantize_by_type(self, graph):
            """
            This method applies fake quantization to the graph based on the type of each node.
            """
            for node in graph.fx_graph.nodes:
                if get_mase_op(node) not in QUANTIZEABLE_OP:
                    continue
                node_config = self.get_config(get_mase_op(node))
                if not node_config['config']['quantize']:
                    continue
                if node.op == "call_module":
                    original_module = get_node_actual_target(node)
                    new_module = self.create_quantized_module(
                        get_mase_op(node),
                        original_module,
                        node_config,
                    )
                    parent_name, name = get_parent_name(node.target)
                    setattr(graph.modules[parent_name], name, new_module)
            return graph

    def fake_quantize_by_name(self, graph):
        """
        This method applies fake quantization to the graph based on the name of each node.
        """
        #TODO implement fake quantize_by_name
        return graph
