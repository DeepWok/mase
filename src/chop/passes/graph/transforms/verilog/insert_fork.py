import torch
import torch.nn as nn
from copy import copy, deepcopy
from chop.ir.graph import MaseMetadata


@torch.fx.wrap
def fork2(x):
    out = x
    return out


def insert_fork_transform_pass(graph, pass_args={}):
    """Insert hardware-explicit forks into the mase graph
    :param graph: a MaseGraph
    :type graph: MaseGraph
    :param pass_args: this pass requires additional arguments which is explained below, defaults to {}
    :type pass_args: _type_, optional
    :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
    :rtype: tuple(MaseGrap`h, Dict)
    """

    def generating_mase_metadata(new_node, node, quan_args):
        new_node.meta["mase"] = MaseMetadata(new_node, node.meta["mase"].model)
        new_node.meta["mase"].parameters["common"]["mase_type"] = "call_function"
        new_node.meta["mase"].parameters["common"]["mase_op"] = "fork2"
        inherited_metadata = deepcopy(
            node.meta["mase"]["common"]["results"]["data_out_0"]
        )
        if quan_args["config"]["name"] == "mxint_hardware":
            inherited_metadata["precision"] = [quan_args["config"]["data_in_width"], quan_args["config"]["data_in_exponent_width"]],
            inherited_metadata["type"] = "mxint_hardware"
        else:
            inherited_metadata["precision"] = quan_args
            inherited_metadata["type"] = "fixed"
        new_node.meta["mase"].parameters["common"]["args"] = {
            "data_in_0": inherited_metadata
        }
        new_node.meta["mase"].parameters["common"]["results"] = {
            "data_out_0": inherited_metadata,
            "data_out_1": inherited_metadata,
        }

        new_node.meta["mase"].parameters["hardware"]["is_implicit"] = False

    nodes_to_fork = []
    from chop.tools.utils import to_numpy_if_tensor, to_tensor_if_numpy
    from chop.passes.graph.transforms.utils import (
        metadata_value_type_cast_transform_pass,
    )

    graph, _ = metadata_value_type_cast_transform_pass(
        graph, pass_args={"fn": to_numpy_if_tensor}
    )
    for node in graph.fx_graph.nodes:
        user_count = 0
        for u in node.users.keys():
            user_count += 1
        if user_count > 1:
            nodes_to_fork.append(node)
    for node in nodes_to_fork:
        with graph.fx_graph.inserting_after(node):
            new_node = graph.fx_graph.call_function(fork2, args=(node,))
            node.replace_all_uses_with(new_node)
            new_node.args = (node,)
            by = pass_args.get("by", "type")
            if by == "type":
                generating_mase_metadata(new_node, node, quan_args=pass_args["fork2"])
            else:
                generating_mase_metadata(
                    new_node, node, quan_args=pass_args[new_node.name]
                )

    # test whether the new graph works
    insert_fifo_after_fork_pass(graph)
    graph, _ = metadata_value_type_cast_transform_pass(
        graph, pass_args={"fn": to_tensor_if_numpy}
    )
    graph.fx_graph.lint()
    return graph, None


@torch.fx.wrap
def fifo(x):
    out = x
    return out


def insert_fifo_after_fork_pass(graph, pass_args={}):
    def generating_mase_metadata(new_node, node, i):
        new_node.meta["mase"] = MaseMetadata(new_node, node.meta["mase"].model)
        new_node.meta["mase"].parameters["common"]["mase_type"] = "call_function"
        new_node.meta["mase"].parameters["common"]["mase_op"] = "fifo"
        inherited_metadata = deepcopy(
            node.meta["mase"]["common"]["args"][f"data_in_{i}"]
        )
        new_node.meta["mase"].parameters["common"]["args"] = {
            "data_in_0": inherited_metadata
        }
        new_node.meta["mase"].parameters["common"]["results"] = {
            "data_out_0": inherited_metadata
        }

        new_node.meta["mase"].parameters["hardware"]["is_implicit"] = False

    record_list = []
    for node in graph.fx_graph.nodes:
        if node.meta["mase"].parameters["common"]["mase_op"] == "fork2":
            for record_node in list(node.users):
                if record_node.meta["mase"].parameters["common"]["mase_op"] == "add":
                    record_list.append(record_node)
    for node in record_list:
        with graph.fx_graph.inserting_before(node):
            for i, arg in enumerate(list(node.args)):
                if arg.meta["mase"].parameters["common"]["mase_op"] == "fork2":
                    new_node = graph.fx_graph.call_function(fifo, args=(arg,))
                    generating_mase_metadata(new_node, node, i)
                    node_args = list(node.args)
                    node_args[i] = new_node
                    node.args = tuple(node_args)
    return graph, None


def insert_fifo_after_specified_modules(graph, pass_args={}):
    def generating_mase_metadata(new_node, node, parallelism):
        new_node.meta["mase"] = MaseMetadata(new_node, node.meta["mase"].model)
        new_node.meta["mase"].parameters["common"]["mase_type"] = "call_function"
        new_node.meta["mase"].parameters["common"]["mase_op"] = "fifo"
        inherited_metadata = deepcopy(
            node.meta["mase"]["common"]["results"][f"data_out_0"]
        )
        new_node.meta["mase"].parameters["common"]["args"] = {
            "data_in_0": inherited_metadata,
            "depth": inherited_metadata["shape"][-1] // parallelism,
        }
        new_node.meta["mase"].parameters["common"]["results"] = {
            "data_out_0": inherited_metadata
        }

        new_node.meta["mase"].parameters["hardware"]["is_implicit"] = False

    from chop.tools.utils import to_numpy_if_tensor, to_tensor_if_numpy
    from chop.passes.graph.transforms.utils import (
        metadata_value_type_cast_transform_pass,
    )

    graph, _ = metadata_value_type_cast_transform_pass(
        graph, pass_args={"fn": to_numpy_if_tensor}
    )
    record_list = []
    for node in graph.fx_graph.nodes:
        if (
            node.meta["mase"].parameters["common"]["mase_op"]
            in pass_args["insert_fifo"]
        ):
            record_list.append(node)
    for node in record_list:
        with graph.fx_graph.inserting_after(node):
            new_node = graph.fx_graph.call_function(fifo, args=(node,))
            node.replace_all_uses_with(new_node)
            new_node.args = (node,)
            generating_mase_metadata(new_node, node, pass_args["max_parallelism"])

    graph, _ = metadata_value_type_cast_transform_pass(
        graph, pass_args={"fn": to_tensor_if_numpy}
    )
    graph.fx_graph.lint()
    return graph, None
