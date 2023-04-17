# ----------------------------
# MaseInterpreter
# - MaseInterpreter.forward_to_interpret triggers hook functions before/after run each node
# - Hook functions should only update node.meta, rather than replace nodes
# ----------------------------

from collections import defaultdict
from logging import getLogger
from typing import Any, Callable, Optional, Tuple

import torch
from torch.fx import Interpreter
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node

logger = getLogger(__name__)


def _dummy_hook_before_forward(
    graph_module: GraphModule,
    fetch_module_by_target: Optional[Callable] = None,
):
    """
    `fetch_module_by_target` is provided by MaseInterpreter
    """
    return graph_module


def _dummy_hook_before_call_function(node: Node, function: Callable, args, kwargs):
    """ """


def _dummy_hook_before_call_module(node: Node, module: torch.nn.Module, args, kwargs):
    """ """


def _dummy_hook_before_call_method(node: Node, method_name: str, args, kwargs):
    """ """


def _dummy_hook_after_call_function(node: Node, function: Callable, output):
    """ """


def _dummy_hook_after_call_module(node: Node, module: torch.nn.Module, output):
    """ """


def _dummy_hook_after_call_method(node: Node, method_name: str, output):
    """ """


def _dummy_hook_after_forward(
    graph_module: GraphModule, fetch_module_by_target: Optional[Callable] = None
):
    """ """
    return graph_module


def clear_node_metadata_software(graph_module: GraphModule):
    for node in graph_module.graph.nodes:
        meta = node.meta
        node.meta = {
            "common": meta.get("common", {}),
            "software": {"modify-sw": {}},
            "hardware": meta.get("hardware", {}),
        }


def clear_node_metadata_hardware(graph_module: GraphModule):
    for node in graph_module.graph.nodes:
        meta = node.meta
        node.meta = {
            "common": meta.get("common", {}),
            "software": meta.get("software", {}),
            "hardware": {},
        }


def clear_node_metadata_common(graph_module: GraphModule):
    for node in graph_module.graph.nodes:
        meta = node.meta
        node.meta = {
            "common": {},
            "software": meta.get("software", {}),
            "hardware": meta.get("hardware", {}),
        }


class MaseInterpreter(Interpreter):
    """
    ---
    Hook function signature requirement:
    - hooks_before/after_forward should be a list of function receiving (graph_module, fetch_module_by_target), where `fetch_module_by_target` returns module instances given "call_function" node's node.target
    - hook_before_call_function/module/method should receive params (node, function/module/method_name, args, kwargs)
    - hook_after_call_function/module/method should receive params (node, function/module/method_name, output)
    """

    def __init__(
        self,
        graph_module: GraphModule,
        hooks_before_forward: Tuple[Callable] = (_dummy_hook_before_forward,),
        hook_before_call_function: Callable = _dummy_hook_before_call_function,
        hook_after_call_function: Callable = _dummy_hook_after_call_function,
        hook_before_call_module: Callable = _dummy_hook_before_call_module,
        hook_after_call_module: Callable = _dummy_hook_after_call_module,
        hook_before_call_method: Callable = _dummy_hook_before_call_method,
        hook_after_call_method: Callable = _dummy_hook_after_call_method,
        hooks_after_forward: Tuple[Callable] = (_dummy_hook_after_forward,),
        garbage_collect_values: bool = True,
    ):
        """
        ---
        Hook function signature requirement:
        - hooks_before/after_forward should be a list of function receiving (graph_module, fetch_module_by_target), where `fetch_module_by_target` returns module instances given "call_function" node's node.target
        - hook_before_call_function/module/method should receive params (node, function/module/method_name, args, kwargs)
        - hook_after_call_function/module/method should receive params (node, function/module/method_name, output)
        """
        super().__init__(graph_module, garbage_collect_values)

        self.hooks_before_forward = hooks_before_forward

        self.hook_before_call_function = hook_before_call_function
        self.hook_after_call_function = hook_after_call_function

        self.hook_before_call_module = hook_before_call_module
        self.hook_after_call_module = hook_after_call_module

        self.hook_before_call_method = hook_before_call_method
        self.hook_after_call_method = hook_after_call_method

        self.hooks_after_forward = hooks_after_forward

    def run_node(self, n: Node) -> Any:
        with self._set_current_node(n):
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)
            if n.op == "call_function":
                function = n.target
                self.hook_before_call_function(
                    node=n, function=function, args=args, kwargs=kwargs
                )
                output = self.call_function(n.target, args, kwargs)

                self.hook_after_call_function(
                    node=n,
                    function=function,
                    output=output,
                )
                return output
            elif n.op == "call_module":
                module = self.fetch_attr(n.target)
                self.hook_before_call_module(
                    node=n, module=module, args=args, kwargs=kwargs
                )
                output = self.call_module(n.target, args=args, kwargs=kwargs)
                self.hook_after_call_module(node=n, module=module, output=output)
                return output
            elif n.op == "call_method":
                # args[0] is the `self` object for this method call
                self.hook_before_call_method(
                    node=n, method_name=n.target, args=args, kwargs=kwargs
                )
                output = self.call_method(n.target, args=args, kwargs=kwargs)
                self.hook_after_call_method(node=n, method_name=n.target, output=output)
                return output
            else:
                return getattr(self, n.op)(n.target, args, kwargs)

    def interpret_before_forward(self) -> GraphModule:
        for hook in self.hooks_before_forward:
            self.module = hook(
                graph_module=self.module, fetch_module_by_target=self.fetch_attr
            )
            assert isinstance(
                self.module, GraphModule
            ), f"hook {hook} didn't return a graph module"
        return self.module

    def interpret_after_forward(self) -> GraphModule:
        for hook in self.hooks_after_forward:
            hook(graph_module=self.module, fetch_module_by_target=self.fetch_attr)
            assert isinstance(
                self.module, GraphModule
            ), f"hook {hook} didn't return a graph module"
        return self.module

    def interpret_without_forward(self) -> GraphModule:
        """
        Traverse graph to run pass assigned both before and after forward propagation. No input args and this no forward propagation
        """
        self.module = self.interpret_before_forward()
        self.module = self.interpret_after_forward()
        return self.module

    def interpret_with_forward(self, *args) -> GraphModule:
        """
        Takes a model input, run pass before forward, during forward, and after forward
        """
        assert len(args) > 0, "Input must be provided for forward_to_interpretation"
        self.module = self.interpret_before_forward()
        outputs = self.run(*args)  # forward
        self.module = self.interpret_after_forward()
        return self.module
