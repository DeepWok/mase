import logging
import os
import pickle
import pprint
from copy import deepcopy
from typing import Dict, Tuple

import toml
import torch
from tabulate import tabulate
from torch import nn
from torch.fx import Graph, GraphModule
from torch.fx.proxy import TraceError

from ..graph.dummy_inputs import _get_default_args
from ..graph.mase_tracer import MaseTracer
from ..graph.utils import (
    check_func_type,
    get_module_by_name,
    get_node_by_name,
    get_parent_name,
    isinstance_but_not_subclass,
)
from .quantizers import functions_map, ops_map, possible_ops
from .utils import copy_weights, load_checkpoint_into_model, load_model

pp = pprint.PrettyPrinter(depth=4)


def create_dummy_inputs(model: nn.Module):
    # TODO: Create dummy inputs for different models
    return {}


class Modifier:
    modifiable_layers = ["linear", "relu", "conv2d"]
    modifiable_functions = ["add"]

    def __init__(
        self,
        model: nn.Module,
        config: str,
        dummy_inputs: Dict = {},
        save_dir: str = None,
        load_name: str = None,
    ) -> None:
        """
        save_dir is expected to be "${no_hyphen_model_name}/software/modify-sw"
        """
        # keep a copy of original model only for comparison
        self.original_model = deepcopy(model)
        # load modify config from toml
        if not config.endswith("toml"):
            raise ValueError("Config File must be a toml file")
        self.config = toml.load(config)
        # create dummy inputs
        self.dummy_inputs = dummy_inputs
        # save_dir for comparison report and modified model
        self.save_dir = save_dir

        # load checkpoint if load_name is given
        if load_name:
            model = load_checkpoint_into_model(load_name, model)
        # create the graph of model
        self.graph: Graph = MaseTracer().trace(model, dummy_inputs)

        self.graph_module = GraphModule(model, self.graph)

        self.modify()
        self.compare_model()
        self.save_modified_model()

    def compare_model(self):
        headers = ["Name", "Op", "Original", "Modified", "Changed?"]
        original_named_modules = dict(self.original_model.named_modules())
        original_graph = MaseTracer().trace(self.original_model, self.dummy_inputs)

        modified_named_modules = dict(self.graph_module.named_modules())

        tabular_rows = []
        for node in self.graph_module.graph.nodes:
            if node.op == "call_module":
                original_module = original_named_modules[node.target]
                modified_module = modified_named_modules[node.target]
                changed = type(original_module) != type(modified_module)
                row = [
                    f"`{node.name}`",
                    f"`{node.op}`",
                    "`{}`".format(str(type(original_module))),
                    "`{}`".format(str(type(modified_module))),
                    changed,
                ]
                tabular_rows.append(row)
            elif node.op == "call_function":
                original_node = get_node_by_name(original_graph, node.name)
                changed = original_node.target == node.target
                row = [
                    f"`{node.name}`",
                    f"`{node.op}`",
                    f"`{original_node.target}`",
                    f"`{node.target}`",
                    changed,
                ]
                tabular_rows.append(row)

        report = tabulate(tabular_rows, headers=headers, tablefmt="github")
        print("A tabular summary of modified funcs and modules")
        print(report)

        if self.save_dir:
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)
            report_path = os.path.join(self.save_dir, "modify-sw-report.md")

            with open(report_path, "w+") as f:
                f.write(report)
                logging.info(
                    f"The tabular summary of modify-sw is saved to {report_path}"
                )

    def load_model(self, model):
        """
        The model can be:
            1. partially modified
            2. non-modified model

        The checkpoint can be:
            1. wrapped model saved by pl.Trainer
            2. user's ckpt
            3. ckpt saved by modifier
        """
        model = load_checkpoint_into_model(self.load_name, model)
        logging.info("Checkpoint loaded before software modification")
        return model

    def save_modified_model(self):
        if self.save_dir:
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)
            modified_ckpt_path = os.path.join(self.save_dir, "modified_model.ckpt")
            modified_pkl_path = os.path.join(self.save_dir, "modified_model.pkl")

            torch.save(
                {"state_dict": self.graph_module.state_dict()}, modified_ckpt_path
            )
            with open(modified_pkl_path, "wb") as f:
                pickle.dump(self.graph_module, file=f)

            logging.info(
                f"Modified model is saved as {modified_ckpt_path} and {modified_pkl_path}"
            )

    def modify(self):
        default_config = self.config.pop("default", None)
        if default_config is None:
            raise ValueError("Default config is not provided")

        for name in self.modifiable_layers:
            if name in self.config:
                layer_config = self.config[name]
                replace_fn = getattr(self, f"replace_module_{name}")
                replace_fn(layer_config)
            else:
                replace_fn = getattr(self, f"replace_module_{name}")
                replace_fn(default_config)

        for name in self.modifiable_functions:
            if name in self.config:
                fn_config = self.config[name]
                replace_fn = getattr(self, f"replace_func_{name}")
                replace_fn(fn_config)

    def _traverse_graph_to_replace_nodes(
        self, target: type, replacement_fn, replacement_fn_kwargs={}
    ):
        named_modules = dict(self.graph_module.named_modules())

        for node in self.graph.nodes:
            if node.op == "call_module":
                layer = get_module_by_name(self.graph_module, node.target)
                if isinstance_but_not_subclass(layer, target):
                    new_layer = replacement_fn(layer)
                    parent_name, name = get_parent_name(node.target)
                    named_modules[node.target] = new_layer
                    setattr(named_modules[parent_name], name, new_layer)
            if node.op == "call_function":
                if check_func_type(node, target):
                    with self.graph.inserting_after(node):
                        kwargs = {**replacement_fn_kwargs, **node.kwargs}
                        new_node = self.graph.call_function(
                            replacement_fn, node.args, kwargs
                        )
                        node.replace_all_uses_with(new_node)
                    self.graph.erase_node(node)
        self.graph.lint()
        self.graph_module.recompile()
        self.graph_module = GraphModule(self.graph_module, self.graph)

    def replace_func_add(self, config):
        replacement_fn = functions_map["add"][config["name"]]
        target = torch.add
        self._traverse_graph_to_replace_nodes(
            target, replacement_fn, replacement_fn_kwargs={"config", config}
        )

    def replace_module_linear(self, config):
        replace_cls = ops_map["linear"][config["name"]]
        target = nn.Linear

        # This shows how to use the replace function
        # First, we have to define a custom replacement_fn
        # We then call the replace function with the model, target layer, and replacement_fn
        def replacement_fn(child):
            # Check https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
            # for details about the class definition of child
            use_bias = child.bias is not None
            # class instantiation
            my_linear = replace_cls(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=use_bias,
                config=config,
            )

            # grab pretrained weights
            copy_weights(child.weight, my_linear.weight)
            if use_bias:
                copy_weights(child.bias, my_linear.bias)
            return my_linear

        self._traverse_graph_to_replace_nodes(target, replacement_fn)

    def replace_module_conv2d(self, config):
        replace_cls = ops_map["conv2d"][config["name"]]
        target = nn.Conv2d

        def replacement_fn(child):
            use_bias = child.bias is not None
            my_conv = replace_cls(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=use_bias,
                padding_mode=child.padding_mode,
                config=config,
            )
            # grab pretrained weights
            copy_weights(child.weight, my_conv.weight)
            if use_bias:
                copy_weights(child.weight, my_conv.weight)
            return my_conv

        self._traverse_graph_to_replace_nodes(target, replacement_fn)

    def replace_module_relu(self, config):
        replace_cls = ops_map["relu"][config["name"]]
        target = nn.ReLU

        def replacement_fn(child):
            return replace_cls(inplace=child.inplace, config=config)

        self._traverse_graph_to_replace_nodes(target, replacement_fn)
