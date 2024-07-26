import torch

from chop.tools import get_logger
from chop.nn.functional.tensor import (
    torch_size,
    torch_expand,
    torch_view,
    torch_contiguous,
    torch_reshape,
    torch_split,
    torch_permute,
    torch_transpose,
)

logger = get_logger(__name__)
logger.setLevel("DEBUG")


REPLACE_METHODS = {
    "size": torch_size,
    "reshape": torch_reshape,
    "expand": torch_expand,
    "split": torch_split,
    "view": torch_view,
    "permute": torch_permute,
    "transpose": torch_transpose,
    "contiguous": torch_contiguous,
}


def replace_method_with_function(mg, pass_args={}):
    """Replaces call_method calls with call_function calls in the graph.

    Args:
        graph (MaseGraph): The input graph.

    Returns:
        MaseGraph: The graph with method calls replaced with function calls.
    """
    for node in mg.fx_graph.nodes:
        if node.op != "call_method":
            continue

        if node.target in REPLACE_METHODS:

            with mg.fx_graph.inserting_after(node):
                logger.debug(f"Replacing {node.target} with function call.")
                new_node = mg.fx_graph.call_function(
                    REPLACE_METHODS[node.target],
                    node.args,
                    node.kwargs,
                )
                node.replace_all_uses_with(new_node)
            mg.fx_graph.erase_node(node)

        else:
            raise NotImplementedError(
                f"Method {node.target} not implemented in replace_method_with_function."
            )

    mg.model.recompile()

    return mg, {}
