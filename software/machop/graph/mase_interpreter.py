# ----------------------------
# MaseInterpreter
# - MaseInterpreter.forward_and_interpret triggers hook functions before/after run each node
# - Hook functions should only update node.meta, rather than replace nodes
# ----------------------------
# Cheng Zhang

from collections import defaultdict
from logging import getLogger
from typing import Any, Optional

from torch.fx import Interpreter
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node

logger = getLogger(__name__)


def dummy_hook_before_call_function(node, function, args, kwargs):
    logger.debug(
        f"""
    node: {node},
    meta: {node.meta}
    function: {function},
    args: {args},
    kwargs: {kwargs}
    """
    )


def dummy_hook_before_call_module(node, module, args, kwargs):
    logger.debug(
        f"""
    node: {node}
    meta: {node.meta},
    module: {module},
    args: {args},
    kwargs: {kwargs}
    """
    )


def dummy_hook_before_call_method(node, method_name, args, kwargs):
    logger.debug(
        f"""
    node: {node}
    meta: {node.meta},
    method_name: {method_name},
    args: {args},
    kwargs: {kwargs}
    """
    )


def dummy_hook_after_call_function(node, function, output):
    logger.debug(
        f"""
    node: {node},
    meta: {node.meta},
    function: {function},
    output: {output}
    """
    )


def dummy_hook_after_call_module(node, module, output):
    logger.debug(
        f"""
    node: {node},
    meta: {node.meta},
    module: {module},
    output: {output}
    """
    )


def dummy_hook_after_call_method(node, method_name, output):
    logger.debug(
        f"""
    node: {node},
    meta: {node.meta},
    method_name: {method_name},
    output: {output}
    """
    )


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
    def __init__(
        self,
        graph_module: GraphModule,
        hook_before_call_function=dummy_hook_before_call_function,
        hook_after_call_function=dummy_hook_after_call_function,
        hook_before_call_module=dummy_hook_before_call_module,
        hook_after_call_module=dummy_hook_after_call_module,
        hook_before_call_method=dummy_hook_before_call_method,
        hook_after_call_method=dummy_hook_after_call_method,
        garbage_collect_values: bool = True,
    ):
        """
        ---
        Hook function signature requirement:
        - hook_before_call_function/module/method should accept params (meta, function/module, args, kwargs)
        - hook_before_call_function/module/method should accept params (meta, function/module, output)
        """
        super().__init__(graph_module, garbage_collect_values)

        self.hook_before_call_function = hook_before_call_function
        self.hook_after_call_function = hook_after_call_function

        self.hook_before_call_module = hook_before_call_module
        self.hook_after_call_module = hook_after_call_module

        self.hook_before_call_method = hook_before_call_method
        self.hook_after_call_method = hook_after_call_method

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
                # breakpoint()
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

    def forward_and_interpret(self, *args):
        """
        Takes a model input, run forward and trigger hook functions to update Node.meta
        """
        assert len(args) > 0, "Input must be provided for this interpretation"
        return self.run(*args)
