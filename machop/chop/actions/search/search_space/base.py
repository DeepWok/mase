import logging
import torch
import copy
import toml

from torch import nn
from chop.passes.utils import get_mase_op, get_mase_type

# from ..graph.mase_tracer import mase_symbolic_trace
# from ..graph.utils import get_module_by_target
# from .plt_wrapper import get_model_wrapper


class SearchSpaceBase:
    def __init__(
        self,
        model_name,
        model,
        mg,
        config,
    ) -> None:
        self.model_name = model_name
        self.model = model

        self.mg = mg
        self.graph_search_space()
        self.config = config

    def build_search_space(self):
        raise NotImplementedError()

    def graph_search_space(self):
        # This is a simple traverse
        node_info = {}
        for node in self.mg.fx_graph.nodes:
            node_info[node.name] = {
                "mase_type": get_mase_type(node),
                "mase_op": get_mase_op(node),
            }
        self.graph_info = node_info
