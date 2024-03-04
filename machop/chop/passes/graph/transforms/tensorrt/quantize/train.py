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


def tensorrt_train_transform_pass(graph, pass_args=None):
    """ Performs Quantized Aware Training """
    by = pass_args["by"]
    trainer = FineTuning(pass_args)
    match by:
        case "type":
            graph = trainer.fake_quantize_by_type(graph)
        case "name":
            graph = trainer.fake_quantize_by_name(graph)
        case "regex_name":
            ...
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')

    # link the model with graph
    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)
    return graph, {}


class FineTuning:
    def __init__(self, config):
        self.config = config

    def get_node_config(self, name: str):
        """Retrieve specific node configuration from the instance's config dictionary or return default."""
        return self.config.get(name, self.config['default'])['config']