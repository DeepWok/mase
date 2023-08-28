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
        accelerator,
    ) -> None:
        self.model_name = model_name
        self.model = model

        self._set_accelerator(accelerator)
        # if we are handling mase graph, lets add some info
        # and perform a simple traverse as a test
        self.mg = mg
        if self.mg is not None:
            self.graph_search_space()

        self.config = config
        self.use_mg = mg is not None

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

    def _set_accelerator(self, accelerator):
        if accelerator == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif accelerator == "gpu":
            self.device = torch.device("cuda:0")
        elif accelerator == "cpu":
            self.device = torch.device("cpu")
        else:
            raise RuntimeError(f"Unsupported accelerator {accelerator}")
