import torch
import logging
import copy

from typing import Tuple
from machop.graph.mase_graph import MaseGraph
from machop.graph.utils import (
    get_module_by_name, 
    isinstance_but_not_subclass, 
    get_parent_name, check_func_type)
from functools import partial


class MaseModifyGraph(MaseGraph):
    def __init__(self, model=None):
        super().__init__(model)
        self.modified_model = copy.deepcopy(model)
        self.logs = {}

    def modify(
            self, target, replacement_fn, replacement_fn_kwargs={}):
        modules = dict(self.modified_model.named_modules())
        for node in self.fx_graph.nodes:
            if node.op == "call_module":
                layer = get_module_by_name(self.model, node.target)
                # module replacement is an experimental phase feature in torch.fx
                # https://github.com/pytorch/pytorch/blob/master/torch/fx/experimental/optimization.py
                if isinstance_but_not_subclass(layer, target):
                    new_layer = replacement_fn(layer)
                    parent_name, name = get_parent_name(node.target)
                    modules[node.target] = new_layer
                    setattr(modules[parent_name], name, new_layer)
                    self.logs[node.target] = new_layer.config
            if node.op == 'call_function':
                if check_func_type(node, target):
                    with self.fx_graph.inserting_after(node):
                        kwargs = {**replacement_fn_kwargs, **node.kwargs}
                        new_node = self.fx_graph.call_function(
                            replacement_fn, node.args, kwargs)
                        node.replace_all_uses_with(new_node)
                    self.fx_graph.erase_node(node)
                    # FIXME: this is very hacky, and it seems like names will get replaced ...
                    self.logs[new_node.name] = replacement_fn_kwargs.get('config', None)
        self.fx_graph.lint()
        self.modules = modules
        self.modified_model = torch.fx.GraphModule(modules, self.fx_graph)
