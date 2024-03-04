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

    def fake_quantize_by_type(self, graph):
            """
            This method applies fake quantization to the graph based on the type of each node.
            """
            for node in graph.fx_graph.nodes:
                if self.get_mase_operation(node) not in QUANTIZEABLE_OP:
                    continue
                node_config = self.get_node_config(self.get_mase_operation(node))
                if node_config["name"] is None:
                    continue
                if node.op == "call_module":
                    original_module = self.get_actual_node_target(node)
                    new_module = self.create_quantized_module(
                        self.get_mase_operation(node),
                        original_module,
                        node_config,
                    )
                    parent_name, name = self.get_parent_name(node.target)
                    setattr(graph.modules[parent_name], name, new_module)
            return graph

    def fake_quantize_by_name(self, graph):
        """
        This method applies fake quantization to the graph based on the name of each node.
        """
        for node in graph.fx_graph.nodes:
            if self.get_mase_operation(node) not in QUANTIZEABLE_OP:
                continue
            node_config = self.get_node_config(node.name)
            if node_config["name"] is None:
                continue
            if node.op == "call_module":
                original_module = self.get_actual_node_target(node)
                new_module = self.create_quantized_module(
                    self.get_mase_operation(node),
                    original_module,
                    node_config,
                )
                parent_name, name = self.get_parent_name(node.target)
                setattr(graph.modules[parent_name], name, new_module)
            else:
                raise ValueError(f"Unsupported node type for quantization: {self.get_mase_type(node)}")
        return graph
