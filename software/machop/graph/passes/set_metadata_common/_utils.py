from logging import getLogger
from typing import List

from torch.fx import Node

logger = getLogger(__name__)


def _get_next_available_dtype_info(node: Node):
    for next_node, _ in node.users.items():
        if next_node.op in ("placeholder", "get_attr", "output"):
            return _get_next_available_dtype_info(next_node)
        else:
            # breakpoint()
            if "data_in" in next_node.meta["common"]["args"]:
                if (
                    next_node.meta["common"]["args"]["data_in"]["type"] != "NA"
                    and next_node.meta["common"]["args"]["data_in"]["precision"] != "NA"
                ):
                    return next_node.meta["common"]["args"]["data_in"]
                else:
                    return _get_next_available_dtype_info(next_node)
            elif "data_in_0" in next_node.meta["common"]["args"]:
                if (
                    next_node.meta["common"]["args"]["data_in_0"]["type"] != "NA"
                    and next_node.meta["common"]["args"]["data_in_0"]["precision"]
                    != "NA"
                ):
                    return next_node.meta["common"]["args"]["data_in_0"]
                else:
                    return _get_next_available_dtype_info(next_node)
            else:
                # breakpoint()
                raise RuntimeError(
                    f"No data_in/data_in_0 keys in Node {next_node}({next_node.op}: {next_node.target})"
                )
    logger.debug(
        f"Node {node} is a dead node, and no available dtype & precision info can be fetched from this node."
    )
    return None


def _get_prev_available_dtype_info(node: Node):
    for prev_node in node.all_input_nodes:
        if prev_node.op in ("placeholder", "get_attr", "output"):
            return _get_next_available_dtype_info(prev_node)
        else:
            if (
                prev_node.meta["common"]["results"]["data_out"]["type"] != "NA"
                and prev_node.meta["common"]["results"]["data_out"]["precision"] != "NA"
            ):
                return prev_node.meta["common"]["results"]["data_out"]
            else:
                return _get_prev_available_dtype_info(prev_node)
    logger.debug(f"Node {node} has no input nodes!!!")
    return None
