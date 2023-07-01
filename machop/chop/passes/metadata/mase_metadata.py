import logging
import math
import os
import time

import torch
import torch.fx
from torch import nn

from ..utils import get_module_by_name

logger = logging.getLogger(__name__)


class MaseMetadata:
    """
    The metadata of a Mase node in a Mase graph describes the constraints of the
    node for any static analysis or possible transformation. The metadata has a
    tree structure, e.g.
    - common
      - mase_op -> str : the mase op of the node, e.g. placeholder, linear, relu
      - mase_type -> str : the mase type of the node, e.g. module, builtin_func, module_related_func
      - args -> {}
         - $name : name of the arg
           - type : type of the arg, e.g. fixed point or float
           - precision : format of the type, e.g. (10, 5)
           - size : size of the arg
           - from : node
      - results -> {}
         - $name : name of the result
           - type : type of the result, e.g. fixed point or float
           - precision : format of the type, e.g. (10, 5)
           - size : size of the result
    - software
    - hardware
      - is_implicit -> bool : whether the node is mapped on hardware or software annotation only
      - verilog_parameters -> {} : parameters need for customise the hardware module
      - toolchain -> str : tool chain for code generation, must be INTERNAL, EXTERNAL or HLS
      - module -> str : the name of the used hardware module
      - interface_parameters -> {}
         - name : name of the parameters
           - storage : the hardware interface implemented, must be BRAM
           - transpose : whether the data needs to be transposed before emitting
      - dependence_files -> [] : the dependent files for the generated module
    ...
    """

    # Hardware dict
    known_types = ["fixed", "float", "NA"]
    known_toolchain = ["INTERNAL", "EXTERNAL", "HLS"]
    known_storage = ["BRAM"]

    def __init__(
        self,
        node=None,
        model=None,
        fx_graph=None,
    ):
        # Top-level model
        self.model = model
        # The target module in the model
        self.module = get_module_by_name(model, node.target)
        # The type of the module
        self.type = type(self.module)
        # The fx node of the module in the fx graph of the model
        self.node = node
        self.graph = fx_graph
        # layers that we have in RTL
        self.internal_layers = {nn.Linear: "linear", nn.ReLU: "relu"}

        self.parameters = {
            "common": {},
            "software": {},
            "hardware": {},
        }
