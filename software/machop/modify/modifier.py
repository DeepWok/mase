import os

import torch
import logging
import toml
import pprint
import pickle

from tabulate import tabulate
from torch import nn
from .quantizers import ops_map, possible_ops, functions_map
from collections import OrderedDict
from .utils import load_model
from .mase_modify_graph import MaseModifyGraph
from functools import partial

pp = pprint.PrettyPrinter(depth=4)


class Modifier:
    modifiable_layers = ["linear", "relu", "conv2d"]
    modifiable_functions = ["add"]

    def __init__(
        self,
        model=None,
        config=None,
        save_name=None,
        load_name=None,
        interactive=False,
        silent=False,
    ):
        self.model = model
        self.graph = MaseModifyGraph(model)


        if load_name is not None:
            self.model = load_model(load_path=load_name, plt_model=self.model)
        # load config as toml
        if not config.endswith(".toml"):
            raise ValueError("Config file must be a toml file")
        config = toml.load(config)

        if not silent:
            logging.info(f"Config loaded!")
            pp.pprint(config)

        self.config = config
        self._pre_modify_check()

        if not silent:
            logging.info(f"Model architecture")
            print(self.model)

        # The core function that makes the modification
        self.modify()
        self.compare_model()

        if not silent:
            logging.info("Model modified")
            logging.info(f"Model architecture, post modification")
            # print(self.model)
            self._post_modify_check()
            self._print_out_diff()

        if save_name is not None:
            self.save(save_name)

        if interactive:
            import pdb
            pdb.set_trace()

    def modify(self):
        default_config = self.config.pop("default", None)
        if default_config is None:
            raise ValueError("Default config is not provided")

        # modify modules
        for name in self.modifiable_layers:
            # use config setup if we have a config
            if name in self.config:
                replace_config = self.config[name]
                replace_fn = getattr(self, f"replace_{name}")
                replace_fn(self.model, replace_config)
            # otherwise all modifiable layers are changed based on default config
            else:
                replace_fn = getattr(self, f"replace_{name}")
                replace_fn(self.model, default_config)
        # modify funcitons
        for name in self.modifiable_functions:
            # use config setup if we have a config
            if name in self.config:
                replace_config = self.config[name]
                replace_fn = getattr(self, f"replace_{name}")
                replace_fn(self.model, replace_config)
            # otherwise all modifiable layers are changed based on default config
            else:
                replace_fn = getattr(self, f"replace_{name}")
                replace_fn(self.model, default_config)
        self.modified_model = self.graph.modified_model


    def _pre_modify_check(self):
        modules = self.model.modules()
        if any([isinstance(m, tuple(possible_ops)) for m in modules]):
            raise ValueError(
                "The model already contains modified classes, why are we modifying again?"
            )
        self._pre_modify_values = self.model.state_dict()

    def _post_modify_check(self):
        self._post_modify_values = OrderedDict()
        for name, module in self.modified_model.named_modules():
            if hasattr(module, "get_quantized_weight"):
                value = module.get_quantized_weight()
                self._post_modify_values[name] = value

    def _print_out_diff(self):
        # printout 10 numbers in each tensor
        logging.info("A pintout, each tensor only outputs 5 values")
        logging.info("Pre-modification values")
        for k, v in self._pre_modify_values.items():
            print(k)
            print(v.flatten()[:5])

        logging.info("Post-modification values")
        for k, v in self._post_modify_values.items():
            print(k)
            print(v.flatten()[:5])

    def replace_add(self, model, config):
        replacement_fn = functions_map["add"][config["name"]]
        target = torch.add 
        replacement_fn = replacement_fn
        self.graph.modify(target, replacement_fn, replacement_fn_kwargs={'config': config})

    def replace_linear(self, model, config):
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
            # WARNING: need to test on the gpu land!
            my_linear.weight = child.weight.cpu()
            if use_bias:
                my_linear.bias = child.bias.cpu()
            return my_linear

        self.graph.modify(target, replacement_fn)

    def replace_conv2d(self, model, config):
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
            # WARNING: need to test on the gpu land!
            my_conv.weight = child.weight.cpu()
            if use_bias:
                my_conv.bias = child.bias.cpu()
            return my_conv

        self.graph.modify(target, replacement_fn)

    def replace_relu(self, model, config):
        replace_cls = ops_map["relu"][config["name"]]
        target = nn.ReLU

        def replacement_fn(child):
            return replace_cls(
                inplace=child.inplace, 
                config=config)
        self.graph.modify(target, replacement_fn)


    def compare_model(self):
        original_modules = dict(self.model.named_modules())
        trace = torch.fx.symbolic_trace(self.model)
        trace.graph.lint()
        trace = trace
        orignal_fx_graph = trace.graph

        modified_modules = self.graph.modules
        modified_fx_graph = self.graph.fx_graph
        headers = ['Name', 'Op', 'Original', 'Modified', 'Changed?']
        rows = []

        for node in self.graph.fx_graph.nodes:
            if node.op == "call_module":
                original_module = original_modules[node.target]
                modified_module = modified_modules[node.target]
                changed = type(original_module) != type(modified_module)
                row = [
                    node.target, node.op, 
                    str(type(original_module)),
                    str(type(modified_module)),
                    changed]
                rows.append(row)
        print(f"A tabular summary of modified modules")
        print(tabulate(rows, headers=headers))

        print(f"A tabular summary of the modified graph")
        modified_fx_graph.print_tabular()

    # # A generic replacement function that works for any layer
    # # This function traverses the modules in a model recursively
    # # And replaces the target layer with the replacement layer
    # def replace(self, model, target, replacement_fn):
    #     for child_name, child in model.named_children():
    #         if isinstance(child, target):
    #             setattr(model, child_name, replacement_fn(child))
    #         else:
    #             self.replace(child, target, replacement_fn)

    def save(self, name):
        if not os.path.isdir("modified_ckpts"):
            os.mkdir("modified_ckpts")

        save_name = f"modified_ckpts/{name}.original.ckpt"
        model_save_name = f"modified_ckpts/{name}.original.pkl"
        torch.save(self.model.state_dict(), save_name)
        with open(model_save_name, "wb") as f:
            pickle.dump(self.model, file=f)

        save_name = f"modified_ckpts/{name}.modified.ckpt"
        model_save_name = f"modified_ckpts/{name}.modified.pkl"
        torch.save(self.modified_model.state_dict(), save_name)
        with open(model_save_name, "wb") as f:
            pickle.dump(self.modified_model, file=f)

        logging.info(f"Original and modified model are saved as {save_name}.")

        if not os.path.isdir("modified_tomls"):
            os.mkdir("modified_tomls")
        save_name = f"modified_tomls/{name}_modified.toml"
        with open(save_name, 'w') as f:
            toml.dump(self.graph.logs, f)
        logging.info(f"Modified toml log is saved as {save_name}.")

