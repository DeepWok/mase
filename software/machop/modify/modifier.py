import logging
import operator
import os
import pickle
from copy import copy, deepcopy
from typing import Callable, Dict, List, Union

import toml
import torch
import torch.nn.functional as F
from machop.utils import copy_weights
from tabulate import tabulate
from torch import nn
from torch.fx import GraphModule, Node

from ..graph.mase_interpreter import clear_node_metadata_software
from ..graph.mase_tracer import (
    MaseTracer,
    clear_user_custom_leaf_modules,
    mark_as_user_custom_leaf_module,
)
from ..graph.utils import get_module_by_name, get_parent_name

# from .quantizers import functions_map, layers_map
from .quantizers import (
    FUNC_MAP_NEO,
    METHOD_MAP_NEO,
    MODULE_CLS_MAP_NEO,
    QUANTIZED_FUNC_CLASSES,
    QUANTIZED_MODULE_CLASSES,
)

logger = logging.getLogger(__name__)


MODULE_CLASS_NAME_TO_MODULE_CLASS = {
    "linear": nn.Linear,
    "relu": nn.ReLU,
    "conv1d": nn.Conv1d,
    "conv2d": nn.Conv2d,
    "avgpool2d": nn.AvgPool2d,
    "adaptiveavgpool2d": nn.AdaptiveAvgPool2d,
}

MODULE_CLASS_TO_MODULE_CLASS_NAME = {
    nn.Linear: "linear",
    nn.ReLU: "relu",
    nn.Conv1d: "conv1d",
    nn.Conv2d: "conv2d",
    nn.AvgPool2d: "avgpool2d",
    nn.AdaptiveAvgPool2d: "adaptiveavgpool2d",
}

FUNCTION_NAME_TO_FUNCTIONS = {
    "add": (operator.add, torch.add),
    "relu": (F.relu,),
    "matmul": (operator.matmul, torch.matmul),
    "bmm": (torch.bmm,),
}

FUNCTION_TO_FUNCTION_NAME = {
    operator.add: "add",
    torch.add: "add",
    F.relu: "relu",
    operator.matmul: "matmul",
    torch.matmul: "matmul",
    torch.bmm: "bmm",
}

METHOD_NAMES = ["add", "relu", "matmul", "bmm"]


def dummy_create_new_module_fn(original_model: nn.Module, config: Dict):
    raise NotImplementedError(
        f"Module {original_model} is not a built-in supported module class to modify"
    )


class Modifier:
    def __init__(
        self,
        model: nn.Module,
        config_path: Union[str, dict],
        dummy_inputs_for_fx: Dict = {},
        module_classes_to_modify: Dict[str, Dict] = None,
        function_classes_to_modify: Dict[str, Dict] = None,
        method_classes_to_modify: Dict[str, Dict] = None,
        module_nodes_to_modify: Dict[str, Dict] = None,
        function_nodes_to_modify: Dict[str, Dict] = None,
        method_nodes_to_modify: Dict[str, Dict] = None,
        custom_leaf_module_classes: List[type] = [],
        create_new_custom_module_fn: Callable = dummy_create_new_module_fn,
        save_dir: str = None,
        silent: bool = False,
    ) -> None:
        """
        Note that modules_to_modify will overwrites module_classes_to_modify

        A config file includes:
        - `module_cls_to_modify` (required): a mapping of "module class name -> new module class config"
        - `function_cls_to_modify` (required): a mapping of "function class name -> new function config"
        - `method_cls_to_modify (required): a mapping of "method class name -> substitute function_config"
        - `module_nodes_to_modify` (optional): a mapping of "module node.target -> new module config"
        - `function_nodes_to_modify` (optional): a mapping of "function node.name -> new function config"
        - `method_nodes_to_modify` (optional): a mapping of "function node.name -> substitute function config"

        A `create_new_custom_module` follows the interface:

            ```
            def self.create_new_custom_module(
                original_module: torch.nn.Module,
                config=sub_config: dict
            ) -> torch.nn.Module:
                # construct new module using original module and sub_config
                return new_module
            ```

        Args:
        - config_path (str): path to toml config file
        - dummy_inputs_for_fx: inputs provided for model.forward to freeze some dynamic control flows
        - module_cls_to_modify (Dict[str, Dict]): overwrites `model_classes_to_modify` in config
        - function_cls_to_modify (Dict[str, Dict]): overwrites `functions_to_modify` in config
        - method_cls_to_modify (Dict[str, Dict]): overwrites `methods_to_modify` in config
        - module_nodes_to_modify (Dict[str, Dict]): overwrites `modules_to_modify` in config.
        - function_nodes_to_modify
        - method_nodes_to_modify
        - create_new_custom_module (Callable): call this function to create new layers if a module name in modules_to_modify corresponds to non-built-in module class
        - save_dir is expected to be "${no_hyphen_model_name}/software/modify-sw"
        """

        assert isinstance(model, torch.nn.Module)
        # keep a copy of original model only for comparison
        self.original_model = deepcopy(model)

        self.config = None
        self.dummy_inputs = dummy_inputs_for_fx
        self.module_classes_to_modify = {}
        self.function_classes_to_modify = {}
        self.method_classes_to_modify = {}
        self.module_nodes_to_modify = {}
        self.function_nodes_to_modify = {}
        self.method_nodes_to_modify = {}
        self.create_new_custom_module = create_new_custom_module_fn
        self.save_dir = save_dir
        self.graph = None
        self.graph_module = None

        self.silent = silent

        # load, parse, and save config
        self.load_config(config_path)
        self.overwrite_config(
            module_classes_to_modify,
            function_classes_to_modify,
            method_classes_to_modify,
            module_nodes_to_modify,
            function_nodes_to_modify,
            method_nodes_to_modify,
        )

        # build fx.Graph
        self.build_graph(model, dummy_inputs_for_fx, custom_leaf_module_classes)
        # check if target to modify exists in the graph
        self.check_modifiable()
        self._save_config()

    def load_config(self, config_path: Union[str, dict]):
        if config_path is None:
            self.config = None
            return
        if type(config_path) == dict:
            self.config = config_path
            return
        if not config_path.endswith("toml"):
            raise ValueError("Config File must be a toml file")
        self.config = toml.load(config_path)

    def overwrite_config(
        self,
        module_classes_to_modify: Dict[str, Dict],
        function_classes_to_modify: Dict[str, Dict],
        method_classes_to_modify: Dict[str, Dict],
        module_nodes_to_modify: Dict[str, Dict],
        function_nodes_to_modify: Dict[str, Dict],
        method_nodes_to_modify: Dict[str, Dict],
    ):
        if self.config is None:
            return
        # assert (
        #     "module_classes_to_modify" in self.config
        # ), "Cannot find `module_classes_to_modify` in config"
        # assert (
        #     "functions_to_modify" in self.config
        # ), "Cannot find `functions_to_modify` in config"
        # assert (
        #     "methods_to_modify" in self.config
        # ), "Cannot find `methods_to_modify` in config"
        if module_classes_to_modify is not None:
            self.config["module_classes_to_modify"] |= module_classes_to_modify
        if function_classes_to_modify is not None:
            self.config["function_classes_to_modify"] |= function_classes_to_modify
        if method_classes_to_modify is not None:
            self.config["method_classes_to_modify"] |= method_classes_to_modify
        if module_nodes_to_modify is not None:
            self.config["module_nodes_to_modify"] |= module_nodes_to_modify
        if function_nodes_to_modify is not None:
            self.config["function_nodes_to_modify"] |= function_nodes_to_modify
        if method_nodes_to_modify is not None:
            self.config["method_nodes_to_modify"] |= method_nodes_to_modify
        if (
            len(self.config.get("module_classes_to_modify", [])) == 0
            and len(self.config.get("function_classes_to_modify", [])) == 0
            and len(self.config.get("method_classes_to_modify", [])) == 0
            and len(self.config.get("module_nodes_to_modify", [])) == 0
            and len(self.config.get("function_node_to_modify", [])) == 0
            and len(self.config.get("method_node_to_modify", [])) == 0
        ):
            logger.warning("No modify config found!")

    def build_graph(self, model, dummy_inputs, custom_leaf_module_classes):
        clear_user_custom_leaf_modules()
        for custom_cls in custom_leaf_module_classes:
            mark_as_user_custom_leaf_module(custom_cls)
        graph = MaseTracer().trace(model, dummy_inputs)
        self.graph = graph
        self.graph_module = GraphModule(model, graph)

    def check_modifiable(self):
        """
        Check if model class, function class, and method class are supported to modify
        Check if module node name, function node name, and method node name exists in model's graph

        update corresponding class list
        """
        if self.config is None:
            return

        # check module cls
        if "module_classes_to_modify" in self.config:
            for module_cls_name, module_cls_config in self.config[
                "module_classes_to_modify"
            ].items():
                assert (
                    module_cls_name in MODULE_CLASS_NAME_TO_MODULE_CLASS
                ), f"Unsupported module class: {module_cls_name}"
                module_cls = MODULE_CLASS_NAME_TO_MODULE_CLASS[module_cls_name]
                self.module_classes_to_modify |= {module_cls: module_cls_config}
        # check func cls
        if "function_classes_to_modify" in self.config:
            for func_name, func_config in self.config[
                "function_classes_to_modify"
            ].items():
                assert (
                    func_name in FUNCTION_NAME_TO_FUNCTIONS
                ), f"Unsupported function {func_name}"
                for func in FUNCTION_NAME_TO_FUNCTIONS[func_name]:
                    self.function_classes_to_modify |= {func: func_config}
        # check method cls
        if "method_classes_to_modify" in self.config:
            for method_name, method_config in self.config[
                "method_classes_to_modify"
            ].items():
                assert method_name in METHOD_NAMES, f"Unsupported method {method_name}"
                self.method_classes_to_modify |= {method_name: method_config}
        # check module node.target
        if "module_nodes_to_modify" in self.config:
            module_names = list(
                map(
                    lambda x: x.target,
                    filter(lambda x: x.op == "call_module", self.graph.nodes),
                )
            )
            for name in self.config["module_nodes_to_modify"]:
                assert name in module_names
            self.module_nodes_to_modify = self.config["module_nodes_to_modify"]
        # check func node.name
        if "function_nodes_to_modify" in self.config:
            function_names = list(
                map(
                    lambda x: x.name,
                    filter(lambda x: x.op == "call_function", self.graph.nodes),
                )
            )
            for name in self.config["function_nodes_to_modify"]:
                assert name in function_names
            self.function_nodes_to_modify = self.config["function_nodes_to_modify"]
        # check method node.name
        if "method_nodes_to_modify" in self.config:
            method_names = list(
                map(
                    lambda x: x.name,
                    filter(lambda x: x.op == "call_method", self.graph.nodes),
                )
            )
            for name in self.config["method_nodes_to_modify"]:
                assert name in method_names
            self.method_nodes_to_modify = self.config["method_nodes_to_modify"]

    def _save_config(self):
        if self.save_dir:
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)
            config_path = os.path.join(self.save_dir, "modify-sw-config.toml")
            with open(config_path, "w+") as f:
                toml.dump(self.config, f)
            logger.info(f"Modify-sw config is saved to {config_path}")

    def modify(self) -> GraphModule:
        is_training_mode = self.graph_module.training
        self.graph_module.eval()
        clear_node_metadata_software(self.graph_module)
        self._modify()
        self._copy_attributes()
        if is_training_mode:
            self.graph_module.train()
        self.compare_model()
        self._save_modified_model_to_pkl()
        return self.graph_module

    def compare_model(self):
        original_named_modules = dict(self.original_model.named_modules())
        original_graph = MaseTracer().trace(self.original_model, self.dummy_inputs)

        modified_named_modules = dict(self.graph_module.named_modules())

        complete_comparison = []
        modify_hits_and_misses = {}
        total_modules = 0
        total_functions = 0
        total_methods = 0

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
                complete_comparison.append(row)

                if str(type(original_module)) in modify_hits_and_misses:
                    modify_hits_and_misses[str(type(original_module))]["count"] += 1
                else:
                    modify_hits_and_misses[str(type(original_module))] = {
                        "count": 1,
                        "changed": changed,
                        "op": old_node.op,
                    }
                total_modules += 1

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
                complete_comparison.append(row)

                if str(old_node.target) in modify_hits_and_misses:
                    modify_hits_and_misses[str(old_node.target)]["count"] += 1
                else:
                    modify_hits_and_misses[str(old_node.target)] = {
                        "count": 1,
                        "changed": changed,
                        "op": old_node.op,
                    }
                total_functions += 1

            elif node.op == "call_method":
                changed = old_node.target != node.target
                row = [
                    f"`{old_node.name}`",
                    f"`{node.name}`",
                    f"`{node.op}`",
                    f"`{old_node.target}`",
                    f"`{node.target}`",
                    changed,
                ]
                complete_comparison.append(row)

                if str(old_node.target) in modify_hits_and_misses:
                    modify_hits_and_misses[str(old_node.target)]["count"] += 1
                else:
                    modify_hits_and_misses[str(old_node.target)] = {
                        "count": 1,
                        "changed": changed,
                        "op": old_node.op,
                    }
                total_methods += 1

        headers = ["Old Name", "Name", "Op", "Original", "Modified", "Changed?"]
        report = tabulate(complete_comparison, headers=headers, tablefmt="github")
        self.original_model = None

        if not self.silent:
            logger.info("Histogram of modified model")
        headers = ["Original OP", "Num", "Changed?"]
        histogram_rows = []
        for op_name, d in modify_hits_and_misses.items():
            row = [op_name, d["count"], d["changed"]]
            histogram_rows.append(row)
        histogram_rows = sorted(histogram_rows, key=lambda x: x[1], reverse=True)
        histogram_rows.append(["All 'call_module'", total_modules, "-"])
        histogram_rows.append(["All 'call function'", total_functions, "-"])
        histogram_rows.append(["All 'call method'", total_methods, "-"])
        if not self.silent:
            print(tabulate(histogram_rows, headers=headers, tablefmt="github"))

        if self.save_dir:
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)
            report_path = os.path.join(self.save_dir, "modify-sw-report.md")

            with open(report_path, "w+") as f:
                f.write(report)
            logger.info(f"The tabular summary of modify-sw is saved to {report_path}")

            profile_path = os.path.join(self.save_dir, "modify-sw-histogram.toml")
            with open(profile_path, "w+") as f:
                toml.dump(modify_hits_and_misses, f)
            logger.info(
                f"The histogram summary of modify-sw is saved to {profile_path}"
            )

    def _save_modified_model_to_pkl(self):
        if self.save_dir:
            modified_pkl_path = os.path.join(self.save_dir, "modified_model.pkl")
            with open(modified_pkl_path, "wb") as f:
                pickle.dump(self.graph_module, file=f)
            logger.info(f"Modified model is saved at {modified_pkl_path}")

    def _modify(self):
        assert "default" in self.config, "Please provide `default` config."
        default_config = self.config["default"]

        # Traverse all nodes to perform node replacement
        named_modules = dict(self.graph_module.named_modules())
        for node in self.graph.nodes:
            # Coarse-grained + fine-grained module replacement
            if node.op == "call_module":
                module_name = node.target
                module = get_module_by_name(self.graph_module, module_name)
                module_cls = type(module)

                if module_name in self.module_nodes_to_modify:
                    # fetch config for module class
                    sub_config = self.module_nodes_to_modify[module_name]
                    # .named_modules() may include repeated nodes,
                    # if this node has been modified, just skip it
                    if module_cls in QUANTIZED_MODULE_CLASSES:
                        continue
                elif module_cls in self.module_classes_to_modify:
                    # fetch config for this layer
                    sub_config = self.module_classes_to_modify[module_cls]
                else:
                    continue

                if sub_config["name"] == "NA":
                    continue
                elif sub_config["name"] == "default":
                    sub_config = default_config

                try:
                    new_module = _create_new_module(
                        original_module=module,
                        config=sub_config,
                    )
                except NotImplementedError:
                    # print(f"Custom module {module}, {type(module)}")
                    # create new layer using user provided module
                    new_module = self.create_new_custom_module(
                        original_module=module, config=sub_config
                    )
                # replace the old module with the new one
                parent_name, name = get_parent_name(module_name)
                setattr(named_modules[parent_name], name, new_module)
                node.meta["software"]["modify-sw"] = {
                    "config": getattr(new_module, "config", sub_config)
                }
            # Coarse-grained + fine-grained function replacement
            elif node.op == "call_function":
                function_node = node.name
                function_cls = node.target

                if function_node in self.function_nodes_to_modify:
                    sub_config = self.function_nodes_to_modify[function_node]
                    if function_cls in QUANTIZED_FUNC_CLASSES:
                        continue
                elif function_cls in self.function_classes_to_modify:
                    sub_config = self.function_classes_to_modify[function_cls]
                else:
                    continue

                if sub_config["name"] == "NA":
                    continue
                elif sub_config["name"] == "default":
                    sub_config = default_config
                new_function, args, kwargs = _get_new_func_args_and_kwargs(
                    original_node=node, config=sub_config
                )
                with self.graph.inserting_after(node):
                    new_node = self.graph.call_function(
                        new_function, args=args, kwargs=kwargs
                    )
                    new_node.name = node.name
                    new_node.meta = copy(node.meta)
                    new_node.meta["software"]["modify-sw"] = {"config": sub_config}
                    node.replace_all_uses_with(new_node)
                self.graph.erase_node(node)
            # Coarse-grained + fine-grained Tensor method replacement
            # Note that Modifier replace methods with functions
            elif node.op == "call_method":
                # Replace method with function
                method_node = node.name
                method_name = node.target

                if method_node in self.method_nodes_to_modify:
                    sub_config = self.method_nodes_to_modify[method_node]
                elif method_name in self.method_classes_to_modify:
                    sub_config = self.method_classes_to_modify[method_name]
                else:
                    continue

                if sub_config["name"] == "NA":
                    continue
                elif sub_config["name"] == "default":
                    sub_config = default_config
                (
                    substitute_func,
                    args,
                    kwargs,
                ) = _get_substitute_function_args_and_kwargs(
                    original_node=node, config=sub_config
                )
                with self.graph.inserting_after(node):
                    new_node = self.graph.call_function(
                        substitute_func, args=args, kwargs=kwargs
                    )
                    new_node.name = node.name
                    new_node.meta = copy(node.meta)
                    new_node.meta["software"]["modify-sw"] = {"config": sub_config}
                    node.replace_all_uses_with(new_node)
                self.graph.erase_node(node)
        self.graph.lint()
        self.graph_module.recompile()
        self.graph_module = GraphModule(self.graph_module, self.graph)
        # fmt: off
        if len(self.module_nodes_to_modify) + len(self.function_nodes_to_modify) + len(self.method_nodes_to_modify)> 0:
            granularity = "fine-grained"
        else:
            granularity = "Coarse-grained"
        # fmt: on
        if not self.silent:
            logger.info(f"{granularity} software modification done")

    def _copy_attributes(self):
        for attr_name in filter(
            lambda attr_name: not attr_name.startswith("__"), dir(self.original_model)
        ):
            if not hasattr(self.graph_module, attr_name):
                setattr(
                    self.graph_module,
                    attr_name,
                    getattr(self.original_model, attr_name),
                )

    @classmethod
    def create_empty_config_template(
        cls,
        model: nn.Module,
        dummy_inputs={},
        custom_leaf_module_classes: List[type] = [],
        save_path: str = None,
    ) -> Dict:
        config = {
            "module_classes_to_modify": {},
            "function_classes_to_modify": {},
            "method_classes_to_modify": {},
            "module_nodes_to_modify": {},
            "function_nodes_to_modify": {},
            "method_nodes_to_modify": {},
        }
        config["default"] = {
            "name": "integer",
            "weight_width": 8,
            "weight_frac_width": 3,
            "data_in_width": 8,
            "data_in_frac_width": 5,
            "bias_width": 8,
            "bias_frac_width": 5,
        }

        for module_cls in sorted(MODULE_CLASS_NAME_TO_MODULE_CLASS.keys()):
            config["module_classes_to_modify"][module_cls] = {"name": "default"}

        for func_cls in sorted(FUNCTION_NAME_TO_FUNCTIONS.keys()):
            config["function_classes_to_modify"][func_cls] = {"name": "default"}

        for method_cls in sorted(METHOD_NAMES):
            config["method_classes_to_modify"][method_cls] = {"name": "default"}

        is_training_mode = model.training
        model.eval()
        clear_user_custom_leaf_modules()
        for custom_cls in custom_leaf_module_classes:
            mark_as_user_custom_leaf_module(custom_cls)
        graph = MaseTracer().trace(model, dummy_inputs)
        for node in graph.nodes:
            if node.op == "call_module":
                model_cls = type(get_module_by_name(model, node.target))
                name = (
                    "default"
                    if model_cls in MODULE_CLASS_TO_MODULE_CLASS_NAME
                    else "NA"
                )
                config["module_nodes_to_modify"][node.target] = {"name": name}
            elif node.op == "call_function":
                name = "default" if node.target in FUNCTION_TO_FUNCTION_NAME else "NA"
                config["function_nodes_to_modify"][node.name] = {"name": name}
            elif node.op == "call_method":
                name = "default" if node.target in METHOD_NAMES else "NA"
                config["method_nodes_to_modify"][node.name] = {"name": name}

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if save_path is not None:
            with open(save_path, "w+") as f:
                toml.dump(config, f)
            logger.info(f"Modify-sw-config template saved at {save_path}")
        if is_training_mode:
            model.train()
        return config


def _create_new_module(original_module: nn.Module, config: Dict):
    original_module_cls = type(original_module)

    if original_module_cls is nn.Linear:
        new_module_cls = MODULE_CLS_MAP_NEO[original_module_cls][config["name"]]
        use_bias = original_module.bias is not None
        new_module = new_module_cls(
            in_features=original_module.in_features,
            out_features=original_module.out_features,
            bias=use_bias,
            config=config,
        )

        copy_weights(original_module.weight, new_module.weight)
        if use_bias:
            copy_weights(original_module.bias, new_module.bias)
    elif original_module_cls in (nn.Conv1d, nn.Conv2d):
        new_module_cls = MODULE_CLS_MAP_NEO[original_module_cls][config["name"]]
        use_bias = original_module.bias is not None
        new_module = new_module_cls(
            in_channels=original_module.in_channels,
            out_channels=original_module.out_channels,
            kernel_size=original_module.kernel_size,
            stride=original_module.stride,
            padding=original_module.padding,
            dilation=original_module.dilation,
            groups=original_module.groups,
            bias=use_bias,
            padding_mode=original_module.padding_mode,
            config=config,
        )
        copy_weights(original_module.weight, new_module.weight)
        if use_bias:
            copy_weights(original_module.weight, new_module.weight)

    elif original_module_cls is nn.ReLU:
        new_module_cls = MODULE_CLS_MAP_NEO[original_module_cls][config["name"]]
        new_module = new_module_cls(inplace=original_module.inplace, config=config)
    elif original_module_cls is nn.AvgPool2d:
        new_module_cls = MODULE_CLS_MAP_NEO[original_module_cls][config["name"]]
        new_module = new_module_cls(
            kernel_size=original_module.kernel_size,
            stride=original_module.stride,
            padding=original_module.padding,
            ceil_mode=original_module.ceil_mode,
            count_include_pad=original_module.count_include_pad,
            divisor_override=original_module.divisor_override,
            config=config,
        )
    elif original_module_cls is nn.AdaptiveAvgPool2d:
        new_module_cls = MODULE_CLS_MAP_NEO[original_module_cls][config["name"]]
        new_module = new_module_cls(
            output_size=original_module.output_size, config=config
        )
    else:
        raise NotImplementedError(
            f"Unsupported module class {original_module_cls} to modify"
        )

    return new_module


def _get_new_func_args_and_kwargs(original_node: Node, config: Dict):
    original_func = original_node.target
    new_func = FUNC_MAP_NEO[original_func][config["name"]]
    if original_func in (operator.add, torch.add):
        if len(original_node.all_input_nodes) >= 1:
            additional_kwargs = {"config": config}
        else:
            additional_kwargs = {}
            new_func = operator.add if original_func == operator.add else torch.add
    elif original_func in (F.relu,):
        additional_kwargs = {"config": config}
    elif original_func in (torch.matmul, operator.matmul):
        additional_kwargs = {"config": config}
    elif original_func in (torch.bmm,):
        additional_kwargs = {"config": config}
    else:
        raise NotImplementedError(f"Unsupported function {original_func} to modify")

    args = original_node.args
    kwargs = original_node.kwargs | additional_kwargs

    return new_func, args, kwargs


def _get_substitute_function_args_and_kwargs(original_node, config: Dict):
    original_method_name = original_node.target
    substitute_func = METHOD_MAP_NEO[original_method_name][config["name"]]
    if original_method_name in ("add",):
        additional_kwargs = {"config": config}
    elif original_method_name in ("relu",):
        additional_kwargs = {"config": config}
    elif original_method_name in ("matmul",):
        additional_kwargs = {"config": config}
    elif original_method_name in ("bmm",):
        additional_kwargs = {"config": config}
    else:
        raise NotImplementedError(
            f"Unsupported method {original_method_name} to modify"
        )

    args = original_node.args
    kwargs = original_node.kwargs | additional_kwargs

    return substitute_func, args, kwargs


def is_modifiable(node: Node, model: nn.Module) -> bool:
    if node.op == "call_module":
        module_cls = type(get_module_by_name(model, node.target))
        return module_cls in MODULE_CLASS_TO_MODULE_CLASS_NAME
    elif node.op == "call_function":
        return node.target in FUNCTION_TO_FUNCTION_NAME
    elif node.op == "call_method":
        return node.target in METHOD_NAMES
    else:
        return False
