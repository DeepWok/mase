import torch
import torch.fx as fx
from torch.fx._symbolic_trace import _assert_is_none

NODE_TARGETS_TO_REMOVE = (torch._assert, _assert_is_none)


def remove_assert(
    graph_module: fx.GraphModule,
    fetch_module_by_target=None,
    node_targets_to_remove=NODE_TARGETS_TO_REMOVE,
):
    """
    This pass removes non-synthesizable nodes in the mase graph before hardware synthesis

    !: This can be dangerous
    """
    graph = graph_module.graph
    nodes_to_remove = []
    for node in graph.nodes:
        if node.target in node_targets_to_remove:
            nodes_to_remove.append(node)
    for node in nodes_to_remove:
        graph.erase_node(node)

    graph.eliminate_dead_code()
    return fx.GraphModule(graph_module, graph)
