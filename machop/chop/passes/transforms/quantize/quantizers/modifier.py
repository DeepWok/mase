import json
import logging
import operator
import os
import pickle
from copy import copy, deepcopy
from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd
import toml
import torch
import torch.nn.functional as F
from chop.utils import copy_weights
from tabulate import tabulate
from torch import nn
from torch.fx import GraphModule, Node

from ..graph.mase_interpreter import clear_node_metadata_software
from ..graph.mase_tracer import (
    MaseTracer,
    clear_user_custom_leaf_modules,
    is_leaf_module_to_trace,
    mark_as_user_custom_leaf_module,
)
from ..graph.utils import get_module_by_name, get_module_by_target, get_parent_name

# from .quantizers import functions_map, layers_map
from .bblog import FUNC_MAP, METHOD_MAP
from .bblog import MODULE_CLS_MAP as _MODULE_CLS_MAP
from .bblog import QUANTIZED_FUNC_CLASSES, QUANTIZED_MODULE_CLASSES

logger = logging.getLogger(__name__)

# -----------------------------------------
# (1). MODULE_CLS_MAP: original module class -> name -> modified module class
# For example: nn.Linear -> "integer" -> LinearInteger
# -----------------------------------------
MODULE_CLS_MAP = deepcopy(_MODULE_CLS_MAP)

# -----------------------------------------
# (2.1) MODULE_CLS_NAME_TO_MODULE_CLS: name -> original class
# For example, "linear" -> nn.Linear
# this "linear" is used in modify-sw-config to specify coarse-grained module class to modify
# -----------------------------------------
_MODULE_CLS_NAME_TO_MODULE_CLS = {
    "linear": nn.Linear,
    "relu": nn.ReLU,
    "conv1d": nn.Conv1d,
    "conv2d": nn.Conv2d,
    "avgpool2d": nn.AvgPool2d,
    "adaptiveavgpool2d": nn.AdaptiveAvgPool2d,
}
MODULE_CLS_NAME_TO_MODULE_CLS = deepcopy(_MODULE_CLS_NAME_TO_MODULE_CLS)
# (2.2) This reversed mapping is used to check the modify-sw-config before modify-sw
_MODULE_CLS_TO_MODULE_CLS_NAME = {
    nn.Linear: "linear",
    nn.ReLU: "relu",
    nn.Conv1d: "conv1d",
    nn.Conv2d: "conv2d",
    nn.AvgPool2d: "avgpool2d",
    nn.AdaptiveAvgPool2d: "adaptiveavgpool2d",
}

MODULE_CLS_TO_MODULE_CLS_NAME = deepcopy(_MODULE_CLS_TO_MODULE_CLS_NAME)

# -----------------------------------------
# (3) MODIFIED_MODULE_CLASSES: a list of modified class
# For example, [LinearInteger, Conv2dInteger]
# This list is used in modify-sw to avoid modifying the same module multiple times
# -----------------------------------------
_MODIFIED_MODULE_CLASSES = [] + QUANTIZED_MODULE_CLASSES
MODIFIED_MODULE_CLASSES = deepcopy(_MODIFIED_MODULE_CLASSES)

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

MODIFIED_FUNC_CLASSES = [] + QUANTIZED_FUNC_CLASSES

METHOD_NAMES = ["add", "relu", "matmul", "bmm"]


def register_original_to_name_to_modified_for_modify_sw(
    original_cls_to_name_to_modified_cls: Dict[type, Dict[str, type]],
):
    """
    update (1) MODULE_CLS_MAP: original module class -> name -> modified module class

    - `original_cls_to_name_to_modified_cls` example: nn.Linear -> "integer" -> LinearInteger
    """
    global MODULE_CLS_MAP
    MODULE_CLS_MAP.update(original_cls_to_name_to_modified_cls)


def reset_original_to_name_to_modified_for_modify_sw():
    """
    reset (1) MODULE_CLS_MAP to built-in MODULE_CLS_MAP
    """
    global MODULE_CLS_MAP
    MODULE_CLS_MAP = deepcopy(_MODULE_CLS_MAP)


def register_cls_name_to_cls_for_modify_sw(cls_name_to_cls: Dict[str, type]):
    """
    update (2.1) and (2.2)
    - (2.1) MODULE_CLS_NAME_TO_MODULE_CLS: maps string cls_name in modify-sw-config to cls,
    (2.1) maps module_class name in modify-sw-config toml to real module class
    - (2.2) MODULE_CLS_TO_MODULE_CLS_NAME: does the reverse. This is used to check if module_class names in modify-sw-config toml are supported.
    """
    global MODULE_CLS_NAME_TO_MODULE_CLS
    global MODULE_CLS_TO_MODULE_CLS_NAME
    cls_to_cls_name = {c: n for n, c in cls_name_to_cls.items()}
    MODULE_CLS_NAME_TO_MODULE_CLS.update(cls_name_to_cls)
    MODULE_CLS_TO_MODULE_CLS_NAME.update(cls_to_cls_name)


def reset_cls_name_to_cls_for_modify_sw():
    """
    reset (2.1) and (2.2) to built-in mapping
    """
    global MODULE_CLS_TO_MODULE_CLS_NAME
    global MODULE_CLS_NAME_TO_MODULE_CLS
    MODULE_CLS_TO_MODULE_CLS_NAME = deepcopy(_MODULE_CLS_TO_MODULE_CLS_NAME)
    MODULE_CLS_NAME_TO_MODULE_CLS = deepcopy(_MODULE_CLS_NAME_TO_MODULE_CLS)


def register_modified_cls_for_modify_sw(classes: List[type]):
    """
    (3) MODIFIED_MODULE_CLASSES: a list of modified class.
    - For example, [LinearInteger, Conv2dInteger]
    - This list is used in modify-sw to avoid modifying the same module multiple times
    """
    global MODIFIED_MODULE_CLASSES
    MODIFIED_MODULE_CLASSES = MODIFIED_MODULE_CLASSES + classes


def reset_modified_cls_for_modify_sw():
    """
    resnet (3) to built-in modified module classes
    """
    global MODIFIED_MODULE_CLASSES
    MODIFIED_MODULE_CLASSES = deepcopy(_MODIFIED_MODULE_CLASSES)


def dummy_create_new_module_fn(original_module: nn.Module, config: Dict):
    raise NotImplementedError(
        f"Module {original_module} is not a built-in supported module class to modify"
    )


class Modifier:
    def __init__(
        self,
        model: nn.Module,
        config_path: Union[str, dict],
        dummy_inputs_for_fx: Dict = {},
        custom_module_cls_to_trace=[],
        custom_module_cls_map={},
        custom_module_cls_name_to_cls={},
        custom_modified_module_cls=[],
        custom_module_create_fn: Callable = dummy_create_new_module_fn,
        save_dir: str = None,
        silent: bool = False,
    ) -> None:
        """
        ---
        Args:
        - config_path (str): path to toml config file
        - dummy_inputs_for_fx: inputs provided for model.forward to freeze some dynamic control flows
        - create_new_custom_module_fn (Callable): call this function to create new layers if a module name in modules_to_modify corresponds to non-built-in module class
        - save_dir is expected to be "${no_hyphen_model_name}/software/modify-sw"

        ---
        Note that modules_to_modify will overwrites module_classes_to_modify

        ---
        A config file includes:
        - `module_cls_to_modify` (required): a mapping of "module class name -> new module class config"
        - `function_cls_to_modify` (required): a mapping of "function class name -> new function config"
        - `method_cls_to_modify (required): a mapping of "method class name -> substitute function_config"
        - `module_nodes_to_modify` (optional): a mapping of "module node.target -> new module config"
        - `function_nodes_to_modify` (optional): a mapping of "function node.name -> new function config"
        - `method_nodes_to_modify` (optional): a mapping of "function node.name -> substitute function config"

        ---
        A `create_new_custom_module_fn` follows the interface:

            ```
            def self.create_new_custom_module(
                original_module: torch.nn.Module,
                config=sub_config: dict
            ) -> torch.nn.Module:
                # construct new module using original module and sub_config
                return new_module
            ```

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

        self.custom_module_cls_to_trace = deepcopy(custom_module_cls_to_trace)
        self.custom_module_cls_map = deepcopy(custom_module_cls_map)
        self.custom_module_cls_name_to_cls = deepcopy(custom_module_cls_name_to_cls)
        self.custom_modified_module_cls = deepcopy(custom_modified_module_cls)
        self.custom_module_create_fn = custom_module_create_fn
        self.save_dir = save_dir
        self.graph = None
        self.graph_module = model

        self.silent = silent

        self.load_config(config_path)

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

    def _reset_custom_leaf_module_mapping(self):
        clear_user_custom_leaf_modules()
        reset_original_to_name_to_modified_for_modify_sw()
        reset_cls_name_to_cls_for_modify_sw()
        reset_modified_cls_for_modify_sw()

    def _register_custom_leaf_module_mapping(self):
        for cls in self.custom_module_cls_to_trace:
            if not is_leaf_module_to_trace(cls):
                mark_as_user_custom_leaf_module(cls)
        register_original_to_name_to_modified_for_modify_sw(self.custom_module_cls_map)
        register_cls_name_to_cls_for_modify_sw(self.custom_module_cls_name_to_cls)
        register_modified_cls_for_modify_sw(self.custom_modified_module_cls)
        if not self.silent:
            if len(self.custom_module_cls_to_trace) == 0:
                logger.info("No custom leaf module class is additionally registered")
            else:
                logger.info(
                    "{} custom leaf module classes are registered".format(
                        len(self.custom_module_cls_to_trace)
                    )
                )

    def build_graph(
        self,
    ):
        self._reset_custom_leaf_module_mapping()
        self._register_custom_leaf_module_mapping()
        graph = MaseTracer().trace(self.graph_module, self.dummy_inputs)
        self.graph = graph
        self.graph_module = GraphModule(self.graph_module, graph)

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
                    module_cls_name in MODULE_CLS_NAME_TO_MODULE_CLS
                ), f"Unsupported module class: {module_cls_name}"
                module_cls = MODULE_CLS_NAME_TO_MODULE_CLS[module_cls_name]
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
            for module_target, sub_config in self.config[
                "module_nodes_to_modify"
            ].items():
                assert (
                    module_target in module_names
                ), f"`{module_target}` is not a valid module node name"
            self.module_nodes_to_modify = self.config["module_nodes_to_modify"]
        # check func node.name
        if "function_nodes_to_modify" in self.config:
            function_names = list(
                map(
                    lambda x: x.name,
                    filter(lambda x: x.op == "call_function", self.graph.nodes),
                )
            )
            for module_target in self.config["function_nodes_to_modify"]:
                assert module_target in function_names
            self.function_nodes_to_modify = self.config["function_nodes_to_modify"]
        # check method node.name
        if "method_nodes_to_modify" in self.config:
            method_names = list(
                map(
                    lambda x: x.name,
                    filter(lambda x: x.op == "call_method", self.graph.nodes),
                )
            )
            for module_target in self.config["method_nodes_to_modify"]:
                assert module_target in method_names
            self.method_nodes_to_modify = self.config["method_nodes_to_modify"]

    def _save_config(self):
        if self.save_dir:
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)
            config_path = os.path.join(self.save_dir, "modify-sw-config.toml")
            with open(config_path, "w+") as f:
                toml.dump(self.config, f)
            if not self.silent:
                logger.info(f"Modify-sw config is saved to {config_path}")

    def modify(self) -> GraphModule:
        assert self.config is not None
        is_training_mode = self.graph_module.training

        self.build_graph()
        self.check_modifiable()

        self.graph_module.eval()
        clear_node_metadata_software(self.graph_module)
        self._modify()
        self._copy_attributes()
        if is_training_mode:
            self.graph_module.train()
        if not self.silent:
            self.compare_model()
        self._save_config()
        self._save_modified_model_to_pkl()
        return self.graph_module

    def compare_model(self):
        # original_named_modules = dict(self.original_model.named_modules())
        original_graph = MaseTracer().trace(self.original_model, self.dummy_inputs)

        # modified_named_modules = dict(self.graph_module.named_modules())

        df = pd.DataFrame(
            columns=[
                "Original name",
                "New name",
                "OP",
                "Original type",
                "New type",
                "Changed",
            ]
        )

        for i, (node, old_node) in enumerate(
            zip(self.graph.nodes, original_graph.nodes)
        ):
            if node.op == "call_module":
                # original_module = original_named_modules[node.target]
                # modified_module = modified_named_modules[node.target]
                original_module = get_module_by_target(
                    self.original_model, old_node.target
                )
                modified_module = get_module_by_target(self.graph_module, node.target)

                original_name = "`{}`".format(old_node.target)
                new_name = "`{}`".format(node.target)
                op = "`call_module`"
                original_type = "`{}`".format(str(type(original_module)))
                new_type = "`{}`".format(str(type(modified_module)))
                changed = type(original_module) != type(modified_module)
            elif node.op == "call_function":
                original_name = "`{}`".format(old_node.name)
                new_name = "`{}`".format(node.name)
                op = "`call_function`"
                original_type = "`{}`".format(old_node.target)
                new_type = "`{}`".format(node.target)
                changed = old_node.target != node.target
            elif node.op == "call_method":
                original_name = "`{}`".format(old_node.name)
                new_name = "`{}`".format(node.name)
                op = "`call_method`"
                original_type = "`{}`".format(old_node.target)
                new_type = "`{}`".format(node.target)
                changed = old_node.target != node.target
            else:
                continue
            df.loc[i] = [original_name, new_name, op, original_type, new_type, changed]

        df = df.reset_index(drop=True)

        report = tabulate(df, headers="keys", tablefmt="github")
        if self.save_dir:
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)
            report_md_path = os.path.join(self.save_dir, "modify-sw-report.md")
            with open(report_md_path, "w+") as f:
                f.write(report)

            logger.info(
                f"The tabular summary of modify-sw is saved to {report_md_path}"
            )

        histogram_df = df.groupby(["Original type"]).agg(
            OP=pd.NamedAgg(column="OP", aggfunc="first"),
            Total=pd.NamedAgg(column="Changed", aggfunc="count"),
            Changed=pd.NamedAgg(column="Changed", aggfunc=lambda x: np.sum(x)),
            Unchanged=pd.NamedAgg(
                column="Changed", aggfunc=lambda x: np.sum(1 - np.array(x))
            ),
        )
        histogram_df = histogram_df.sort_values("Total", ascending=False)
        tmp_df = histogram_df.groupby("OP").agg("sum")
        if "`call_method`" in tmp_df.index:
            histogram_df.loc["call_module"] = [
                "-",
                tmp_df.loc["`call_module`", :]["Total"],
                tmp_df.loc["`call_module`", :]["Changed"],
                tmp_df.loc["`call_module`", :]["Unchanged"],
            ]
        if "`call_function`" in tmp_df.index:
            histogram_df.loc["call_function"] = [
                "-",
                tmp_df.loc["`call_function`", :]["Total"],
                tmp_df.loc["`call_function`", :]["Changed"],
                tmp_df.loc["`call_function`", :]["Unchanged"],
            ]
        if "`call_method`" in tmp_df.index:
            histogram_df.loc["call_method"] = [
                "-",
                tmp_df.loc["`call_method`", :]["Total"],
                tmp_df.loc["`call_method`", :]["Changed"],
                tmp_df.loc["`call_method`", :]["Unchanged"],
            ]

        if self.save_dir:
            histogram_path = os.path.join(self.save_dir, "modify-sw-histogram.md")
            histogram = tabulate(histogram_df, headers="keys", tablefmt="github")
            with open(histogram_path, "w+") as f:
                f.write(histogram)
            logger.info(
                f"The histogram summary of modify-sw is saved to {histogram_path}"
            )
        if not self.silent:
            print(histogram)

    def _save_modified_model_to_pkl(self):
        if self.save_dir:
            modified_pkl_path = os.path.join(self.save_dir, "modified_model.pkl")
            with open(modified_pkl_path, "wb") as f:
                pickle.dump(self.graph_module, file=f)
            if not self.silent:
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
                    if (
                        module_cls
                        in MODIFIED_MODULE_CLASSES
                        # or module_cls in self.new_custom_leaf_module_classes
                    ):
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
                    new_module = self.custom_module_create_fn(
                        original_module=module, config=sub_config
                    )
                except NotImplementedError:
                    if is_modifiable(node, self.graph_module):
                        new_module = _create_new_module(
                            original_module=module, config=sub_config
                        )
                    else:
                        logger.warning(
                            "Module {}: {} is not modifiable.".format(
                                node.target, type(module)
                            )
                        )
                    # new_module = _create_new_module(
                    #     original_module=module, config=sub_config
                    # )

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
                    if function_cls in MODIFIED_FUNC_CLASSES:
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
        # original_custom_leaf_module_classes: List[type] = [],
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

        for module_cls in sorted(MODULE_CLS_NAME_TO_MODULE_CLS.keys()):
            config["module_classes_to_modify"][module_cls] = {"name": "default"}

        for func_cls in sorted(FUNCTION_NAME_TO_FUNCTIONS.keys()):
            config["function_classes_to_modify"][func_cls] = {"name": "default"}

        for method_cls in sorted(METHOD_NAMES):
            config["method_classes_to_modify"][method_cls] = {"name": "default"}

        is_training_mode = model.training
        model.eval()
        graph = MaseTracer().trace(model, dummy_inputs)
        for node in graph.nodes:
            if node.op == "call_module":
                model_cls = type(get_module_by_name(model, node.target))
                assert model_cls is not None
                if model_cls in MODULE_CLS_TO_MODULE_CLS_NAME:
                    name = "default"
                    original_module_cls = MODULE_CLS_TO_MODULE_CLS_NAME[model_cls]
                else:
                    name = "NA"
                    original_module_cls = "NA"

                config["module_nodes_to_modify"][node.target] = {
                    "name": name,
                    "original_module_cls": original_module_cls,
                }
            elif node.op == "call_function":
                name = "default" if node.target in FUNCTION_TO_FUNCTION_NAME else "NA"
                config["function_nodes_to_modify"][node.name] = {"name": name}
            elif node.op == "call_method":
                name = "default" if node.target in METHOD_NAMES else "NA"
                config["method_nodes_to_modify"][node.name] = {"name": name}

        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if save_dir != "" and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(save_path, "w+") as f:
                toml.dump(config, f)
            logger.info(f"Modify-sw-config template saved at {save_path}")
        if is_training_mode:
            model.train()
        return config


def _create_new_module(original_module: nn.Module, config: Dict):
    original_module_cls = type(original_module)

    if original_module_cls is nn.Linear:
        new_module_cls = MODULE_CLS_MAP[original_module_cls][config["name"]]
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
        new_module_cls = MODULE_CLS_MAP[original_module_cls][config["name"]]
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
        new_module_cls = MODULE_CLS_MAP[original_module_cls][config["name"]]
        new_module = new_module_cls(inplace=original_module.inplace, config=config)
    elif original_module_cls is nn.AvgPool2d:
        new_module_cls = MODULE_CLS_MAP[original_module_cls][config["name"]]
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
        new_module_cls = MODULE_CLS_MAP[original_module_cls][config["name"]]
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
    new_func = FUNC_MAP[original_func][config["name"]]
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
    substitute_func = METHOD_MAP[original_method_name][config["name"]]
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
        return module_cls in MODULE_CLS_TO_MODULE_CLS_NAME
    elif node.op == "call_function":
        return node.target in FUNCTION_TO_FUNCTION_NAME
    elif node.op == "call_method":
        return node.target in METHOD_NAMES
    else:
        return False
