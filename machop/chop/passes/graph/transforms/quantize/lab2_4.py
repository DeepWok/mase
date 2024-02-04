import logging
import os

import numpy as np
import pandas as pd
from tabulate import tabulate

from ...utils import get_mase_op, get_mase_type, get_node_actual_target


logger = logging.getLogger(__name__)


def graph_iterator_compare_nodes(
    ori_graph, graph, save_path=None, silent=False
) -> None:
    """List all nodes in the graph and compare the original and quantized nodes."""

    def get_type_str(node):
        if node.op == "call_module":
            return type(get_node_actual_target(node)).__name__
        elif get_mase_type(node) in [
            "builtin_func",
            "module_related_func",
            "patched_func",
        ]:
            return get_node_actual_target(node).__name__
        elif get_mase_type(node) in ["implicit_func"]:
            actual_target = get_node_actual_target(node)
            if isinstance(actual_target, str):
                return actual_target
            else:
                return actual_target.__name__
        else:
            return node.target

    headers = [
        "Graph 1 name",
        "Graph 2 name",
        "MASE_TYPE",
        "Mase_OP",
        "Graph 1 type",
        "Graph 2 type",
        "Change",
    ]
    rows = []
    for ori_n, n in zip(ori_graph.fx_graph.nodes, graph.fx_graph.nodes):
        rows.append(
            [
                ori_n.name,
                n.name,
                get_mase_type(n),
                get_mase_op(n),
                get_type_str(ori_n),
                get_type_str(n),
                type(get_node_actual_target(n)) != type(get_node_actual_target(ori_n)),
            ]
        )
    df = pd.DataFrame(rows, columns=headers)    
    logger.info("Compare nodes:")
    logger.info("\n" + tabulate(df, headers=headers, tablefmt="orgtbl"))

    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(tabulate(rows, headers=headers))

   
    if save_path is not None:
        df.to_csv(save_path)


def compare_graphs_analysis_pass(
    ori_graph, graph, save_dir: str = None
) -> None:
    """
    Compares the two graphs analysis pass.

    Args:
        ori_graph: The original graph.
        graph: The modified graph.
        save_dir (optional): The directory to save the summary files. Defaults to None.
    """

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    table_path = os.path.join(save_dir, "quantize_table.csv") if save_dir else None
    
    graph_iterator_compare_nodes(ori_graph, graph, save_path=table_path, silent=False)
