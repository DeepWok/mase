import logging
import os
from copy import deepcopy

import numpy as np
import pandas as pd
from chop.passes.utils import get_mase_op, get_mase_type, get_node_actual_target
from tabulate import tabulate

logger = logging.getLogger(__name__)


def graph_iterator_compare_nodes(
    ori_graph, graph, save_path=None, silent=False
) -> pd.DataFrame:
    """List all nodes in the graph and compare the original and quantized nodes."""

    def get_type_str(node):
        if node.op == "call_module":
            return type(get_node_actual_target(node)).__name__
        elif get_mase_type(node) in [
            "builtin_func",
            "module_related_func",
        ]:
            return get_node_actual_target(node).__name__
        elif get_mase_type(node) in ["implicit_func"]:
            return get_node_actual_target(node)
        else:
            return node.target

    headers = [
        "Ori name",
        "New name",
        "MASE_TYPE",
        "Mase_OP",
        "Original type",
        "Quantized type",
        "Changed",
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
    if not silent:
        logger.debug("Compare nodes:")
        logger.debug("\n" + tabulate(rows, headers=headers, tablefmt="orgtbl"))
    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(tabulate(rows, headers=headers))

    df = pd.DataFrame(rows, columns=headers)
    if save_path is not None:
        df.to_csv(save_path)

    return df


def graph_iterator_node_histogram(ori_graph, graph, save_path: str = None):
    """Group nodes by their types and count the number of nodes in each group."""
    df = graph_iterator_compare_nodes(ori_graph, graph, save_path=None, silent=True)
    histogram_df = df.groupby(["Original type"]).agg(
        OP=pd.NamedAgg(column="Mase_OP", aggfunc="first"),
        Total=pd.NamedAgg(column="Changed", aggfunc="count"),
        Changed=pd.NamedAgg(column="Changed", aggfunc=lambda x: np.sum(x)),
        Unchanged=pd.NamedAgg(
            column="Changed", aggfunc=lambda x: np.sum(1 - np.array(x))
        ),
    )
    logger.info("Quantized graph histogram:")
    logger.info("\n" + tabulate(histogram_df, headers="keys", tablefmt="orgtbl"))
    if save_path is not None:
        histogram_df.to_csv(save_path)


# def graph_iterator_compare_nodes(*args, **kwargs):
#     # TODO: remove this function when the add_common_metadata is fixed
#     pass


# def graph_iterator_node_histogram(*args, **kwargs):
#     # TODO: remove this function when the add_common_metadata is fixed
#     pass


def quantize_summary_analysis_pass(ori_graph, graph, save_dir: str = None) -> None:
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    table_path = os.path.join(save_dir, "quantize_table.csv") if save_dir else None
    histogram_path = (
        os.path.join(save_dir, "quantize_histogram.csv") if save_dir else None
    )
    graph_iterator_compare_nodes(ori_graph, graph, save_path=table_path, silent=False)
    graph_iterator_node_histogram(ori_graph, graph, save_path=histogram_path)
