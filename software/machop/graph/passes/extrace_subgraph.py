from typing import Dict, List

import torch.fx as fx
import torch.nn as nn


def extract_subgraph(
    orig_module: nn.Module,
    nodes: List[fx.Node],
    inputs: List[fx.Node],
    outputs: List[fx.Node],
):
    """
    Given lists of nodes from an existing graph that represent a subgraph, returns a submodule that executes that subgraph.
    """
    new_graph = fx.Graph()
    env: Dict[fx.Node, fx.Node] = {}
    for input in inputs:
        new_node = new_graph.placeholder(input.name)
        env[input] = new_node
    for node in nodes:
        new_node = new_graph.node_copy(node, lambda x: env[x])
        env[node] = new_node
    new_graph.output([env[output] for output in outputs])
    new_graph.lint()
    return fx.GraphModule(orig_module, new_graph)
