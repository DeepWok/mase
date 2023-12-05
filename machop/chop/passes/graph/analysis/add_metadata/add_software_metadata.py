import logging
from ...utils import get_mase_op, get_mase_type
from .software_metadata_layers import SOFTWARE_PARAM_ANALYSIS_LAYERS


logger = logging.getLogger(__name__)


def add_software_metadata_analysis_pass(graph, pass_args=None):
    """add software metadata

    :param graph: a MaseGraph
    :type graph: MaseGraph
    :param pass_args: this pass does not need any arguments, defaults to None
    :type pass_args: _type_, optional
    :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
    :rtype: tuple(MaseGraph, Dict)
    """

    for node in graph.fx_graph.nodes:
        mase_op = get_mase_op(node)
        mase_type = get_mase_type(node)

        if mase_op in SOFTWARE_PARAM_ANALYSIS_LAYERS[mase_type]:
            SOFTWARE_PARAM_ANALYSIS_LAYERS[mase_type][mase_op](node.meta["mase"])
        else:
            logger.warning(
                f"mase_type `{mase_type}`, mase_op `{mase_op}` not found in SOFTWARE_PARAM_ANALYSIS_LAYERS. Using default analysis layer"
            )
            SOFTWARE_PARAM_ANALYSIS_LAYERS[mase_type]["default"](node.meta["mase"])
    return graph, {}
