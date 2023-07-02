import logging

import torch
import torch.nn as nn
from torch.fx._symbolic_trace import _assert_is_none

logger = logging.getLogger(__name__)


# Currently target fx graph - need to change to mase_graph
def remove_nonsynthesizable_nodes_pass(mase_graph):
    """
    This pass removes non-synthesizable nodes in the mase graph before hardware synthesis
    """
    nodes_to_remove = []
    # nonsynthesizable_nodes = {torch._assert, nn.Dropout, nn.functional.dropout}
    nonsynthesizable_nodes = {torch._assert, _assert_is_none}
    for node in mase_graph.fx_graph.nodes:
        if (
            node.meta["mase"].type in nonsynthesizable_nodes
            or node.target in nonsynthesizable_nodes
        ):
            assert len(node.users.keys()) == 0
            nodes_to_remove.append(node)

    for node in nodes_to_remove:
        mase_graph.fx_graph.erase_node(node)

    mase_graph.fx_graph.eliminate_dead_code()

    return mase_graph
