from logging import getLogger

from tabulate import tabulate

logger = getLogger(__name__)

from ._set_common_dtype import (
    _set_dtype_after_call_function,
    _set_dtype_after_call_method,
    _set_dtype_before_call_function,
    _set_dtype_before_call_method,
    _set_dtype_before_call_module,
    _set_dtype_of_nodes_depending_on_neighbors,
)
from ._set_common_empty import (
    _set_empty_metadata_before_call_function,
    _set_empty_metadata_before_call_method,
    _set_empty_metadata_before_call_module,
)
from ._set_common_size import (
    _set_arg_size_before_call_function,
    _set_arg_size_before_call_method,
    _set_arg_size_before_call_module,
    _set_result_size_after_call_function,
    _set_result_size_after_call_method,
    _set_result_size_after_call_module,
)


def set_metadata_common_before_call_function(node, function, args, kwargs):
    _set_empty_metadata_before_call_function(node, function, args, kwargs)
    _set_arg_size_before_call_function(node, function, args, kwargs)
    _set_dtype_before_call_function(node, function, args, kwargs)


def set_metadata_common_after_call_function(node, function, output):
    _set_result_size_after_call_function(node, function, output)
    _set_dtype_after_call_function(node, function, output)


def set_metadata_common_before_call_module(node, module, args, kwargs):
    _set_empty_metadata_before_call_module(node, module, args, kwargs)
    _set_arg_size_before_call_module(node, module, args, kwargs)
    _set_dtype_before_call_module(node, module, args, kwargs)


def set_metadata_common_after_call_module(node, module, output):
    _set_result_size_after_call_module(node, module, output)


def set_metadata_common_before_call_method(node, method_name, args, kwargs):
    _set_empty_metadata_before_call_method(node, method_name, args, kwargs)
    _set_arg_size_before_call_method(node, method_name, args, kwargs)
    _set_dtype_before_call_method(node, method_name, args, kwargs)


def set_metadata_common_after_call_method(node, method_name, output):
    _set_result_size_after_call_method(node, method_name, output)
    _set_dtype_after_call_method(node, method_name, output)


def set_and_check_metadata_common_without_forward(graph_module, fetch_module_by_target):
    node_list = graph_module.graph.nodes

    # zero_user_nodes = []
    for node in node_list:
        if node.op in ("call_function", "call_module", "call_method"):
            if node.op == "call_function":
                real_target = node.target
            elif node.op == "call_module":
                real_target = fetch_module_by_target(node.target)
            else:
                real_target = node.target
            _set_dtype_of_nodes_depending_on_neighbors(node, real_target)

    return graph_module
    #     if len(node.users) == 0:
    #         zero_user_nodes.append(node)
    # logger.info("zero_user_nodes:")
    # print(zero_user_nodes)
    # breakpoint()

    # rows = []
    # meta_data_profile = {}
    # for node in node_list:
    #     if not node.op in ("call_function", "call_module", "call_method"):
    #         continue
    #     common_meta_args = node.meta["common"]["args"]
    #     common_meta_data_out = node.meta["common"]["results"]["data_out"]
    #     unavailable_list = []
    #     for arg_name, common_meta in common_meta_args.items():
    #         if (
    #             common_meta["type"] == "NA"
    #             or common_meta["precision"] == "NA"
    #             or common_meta["precision_format"] == "NA"
    #             or common_meta["size"] == "NA"
    #         ):
    #             unavailable_list.append(arg_name)

    #     if (
    #         common_meta_data_out["type"] == "NA"
    #         or common_meta_data_out["precision"] == "NA"
    #         or common_meta_data_out["precision_format"]
    #         or common_meta_data_out["size"] == "NA"
    #     ):
    #         unavailable_list.append("data_out")
