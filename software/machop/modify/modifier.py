import logging
import operator
import os
import pickle
import pprint
from copy import deepcopy
from typing import Dict, List

import toml
import torch
from tabulate import tabulate
from torch import nn
from torch.fx import Graph, GraphModule

from ..graph.mase_tracer import MaseTracer
from ..graph.utils import (
    check_func,
    get_module_by_name,
    get_parent_name,
    isinstance_but_not_subclass,
)
from ..utils import load_pt_pl_or_pkl_checkpoint_into_pt_model
from .quantizers import functions_map, ops_map
from .utils import copy_weights

pp = pprint.PrettyPrinter(depth=4)

logger = logging.getLogger(__name__)


# TODO: try layer-wise replacement instead of current class-wise replacement
class Modifier:
    def __init__(
        self,
        model: nn.Module,
        config: str,
        dummy_inputs: Dict = {},
        save_dir: str = None,
        load_name: str = None,
        load_type: str = None,
        modifiable_layers: List[str] = ["linear", "relu", "conv2d"],
        modifiable_functions: List[str] = ["add", "relu", "matmul", "bmm"],
        do_comparison: bool = True,
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
        # module/ layers to be modified

        if "modifiable_layers" in self.config:
            self.modifiable_layers = self.config["modifiable_layers"]
        else:
            self.modifiable_layers = modifiable_layers
            self.config["modifiable_layers"] = modifiable_layers
        if "modifiable_functions" in self.config:
            self.modifiable_functions = self.config["modifiable_functions"]
        else:
            self.modifiable_functions = modifiable_functions
            self.config["modifiable_functions"] = modifiable_functions
        # create dummy inputs
        self.dummy_inputs = dummy_inputs
        # save_dir for comparison report and modified model
        self.save_dir = save_dir

        # load checkpoint if load_name is given
        if load_name:
            model = self.load_model(load_name, load_type=load_type, model=model)
        # create the graph of model
        self.graph: Graph = MaseTracer().trace(model, dummy_inputs)

        self.graph_module = GraphModule(model, self.graph)

        self.modify()
        if do_comparison:
            self.compare_model()
        # breakpoint()
        self.save_modified_model()

    def compare_model(self):
        headers = ["Old Name", "Name", "Op", "Original", "Modified", "Changed?"]
        original_named_modules = dict(self.original_model.named_modules())
        original_graph = MaseTracer().trace(self.original_model, self.dummy_inputs)

        modified_named_modules = dict(self.graph_module.named_modules())

        tabular_rows = []

        for node, old_node in zip(self.graph.nodes, original_graph.nodes):
            if node.op == "call_module":
                original_module = original_named_modules[node.target]
                modified_module = modified_named_modules[node.target]
                changed = type(original_module) != type(modified_module)
                row = [
                    f"`{old_node.name}`",
                    f"`{node.name}`",
                    f"`{node.op}`",
                    "`{}`".format(str(type(original_module))),
                    "`{}`".format(str(type(modified_module))),
                    changed,
                ]
                tabular_rows.append(row)
            elif node.op == "call_function":
                changed = old_node.target != node.target
                row = [
                    f"`{old_node.name}`",
                    f"`{node.name}`",
                    f"`{node.op}`",
                    f"`{old_node.target}`",
                    f"`{node.target}`",
                    changed,
                ]
                tabular_rows.append(row)

        report = tabulate(tabular_rows, headers=headers, tablefmt="github")
        logger.info("A tabular summary of modified funcs and modules")
        print(report)

        if self.save_dir:
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)
            report_path = os.path.join(self.save_dir, "modify-sw-report.md")

            with open(report_path, "w+") as f:
                f.write(report)
            logger.info(f"The tabular summary of modify-sw is saved to {report_path}")

    def load_model(self, load_name, load_type, model):
        """
        The checkpoint can be:
            1. wrapped model ckpt saved by pl.Trainer
            2. user's ckpt
        """
        # model = load_checkpoint_into_model(self.load_name, model)
        model = load_pt_pl_or_pkl_checkpoint_into_pt_model(
            load_name=load_name, load_type=load_type, model=model
        )
        logger.info(f"Checkpoint {load_name} loaded before software modification")
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

            logger.info(
                f"Modified model is saved as {modified_ckpt_path} and {modified_pkl_path}"
            )

    def modify(self):
        default_config = self.config.pop("default", None)
        if default_config is None:
            raise ValueError("Default config is not provided")

        for layer in self.modifiable_layers:
            if layer in self.config:
                layer_config = self.config[layer]
                replace_fn = getattr(self, f"replace_module_{layer}")
                replace_fn(layer_config)
            else:
                replace_fn = getattr(self, f"replace_module_{layer}")
                replace_fn(default_config)

        # TODO: replace F.relu, torch.matmul, F.linear
        # TODO: deal with Node.target == 'call_method'
        for func in self.modifiable_functions:
            if func in self.config:
                fn_config = self.config[func]
                replace_fn = getattr(self, f"replace_func_{func}")
                replace_fn(fn_config)
            else:
                replace_fn = getattr(self, f"replace_func_{func}")
                replace_fn(default_config)

        # copy original model attributes to modified model
        # some of them are useful like HuggingFace `transformers.PretrainedModel.name_or_path`
        for attr_name in filter(
            lambda attr_name: not attr_name.startswith("__"), dir(self.original_model)
        ):
            if not hasattr(self.graph_module, attr_name):
                setattr(
                    self.graph_module,
                    attr_name,
                    getattr(self.original_model, attr_name),
                )

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
                if check_func(node, target):
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

    # -------------------------------------------------------------------------
    # Replace Function
    # -------------------------------------------------------------------------
    def replace_func_add(self, config):
        replacement_fn = functions_map["add"][config["name"]]
        # torch.add(x1, x2)
        target = torch.add
        self._traverse_graph_to_replace_nodes(
            target, replacement_fn, replacement_fn_kwargs={"config": config}
        )
        # x1 + x2
        target = operator.add
        self._traverse_graph_to_replace_nodes(
            target, replacement_fn, replacement_fn_kwargs={"config": config}
        )

    def replace_func_relu(self, config):
        replacement_fn = functions_map["relu"][config["name"]]
        target = torch.nn.functional.relu
        self._traverse_graph_to_replace_nodes(
            target, replacement_fn, replacement_fn_kwargs={"config": config}
        )

    def replace_func_matmul(self, config):
        replacement_fn = functions_map["matmul"][config["name"]]
        target = torch.matmul
        self._traverse_graph_to_replace_nodes(
            target, replacement_fn, replacement_fn_kwargs={"config": config}
        )

    def replace_func_bmm(self, config):
        replacement_fn = functions_map["bmm"][config["name"]]
        target = torch.bmm
        self._traverse_graph_to_replace_nodes(
            target, replacement_fn, replacement_fn_kwargs={"config": config}
        )

    # -------------------------------------------------------------------------
    # Replace nn.Module
    # -------------------------------------------------------------------------
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
