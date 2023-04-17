# -----------------------------------------
# The code is copied and modified from
# https://github.com/pytorch/pytorch/blob/bd07f8d2e03d2d53865584d04945c9f36d4be019/torch/fx/experimental/optimization.py
# -----------------------------------------
import copy
from typing import Any, Dict, Iterable, Tuple, Type

import torch
import torch.fx as fx
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


def _parent_name(target: str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit(".", 1)
    return parent[0] if parent else "", name


# Works for length 2 patterns with 2 modules
def matches_module_pattern(
    pattern: Iterable[Type], node: fx.Node, modules: Dict[str, Any]
):
    if len(node.args) == 0:
        return False
    nodes: Tuple[Any, fx.Node] = (node.args[0], node)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, fx.Node):
            return False
        if current_node.op != "call_module":
            return False
        if not isinstance(current_node.target, str):
            return False
        if current_node.target not in modules:
            return False
        if type(modules[current_node.target]) is not expected_type:
            return False
    return True


def replace_node_module(
    node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module
):
    assert isinstance(node.target, str)
    parent_name, name = _parent_name(node.target)
    modules[node.target] = new_module
    setattr(modules[parent_name], name, new_module)


def fuse_conv_bn_pass(graph_module: fx.GraphModule) -> fx.GraphModule:
    """
    Fuses convolution/BN layers for inference purposes. Will deepcopy your
    model by default, but can modify the model inplace as well.
    """
    patterns = [
        (nn.Conv1d, nn.BatchNorm1d),
        (nn.Conv2d, nn.BatchNorm2d),
        (nn.Conv3d, nn.BatchNorm3d),
    ]
    modules = dict(graph_module.named_modules())
    new_graph = copy.deepcopy(graph_module.graph)

    for pattern in patterns:
        for node in new_graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                if not bn.track_running_stats:
                    continue
                fused_conv = fuse_conv_bn_eval(conv, bn)
                replace_node_module(node.args[0], modules, fused_conv)
                node.replace_all_uses_with(node.args[0])
                new_graph.erase_node(node)
    return fx.GraphModule(graph_module, new_graph)
