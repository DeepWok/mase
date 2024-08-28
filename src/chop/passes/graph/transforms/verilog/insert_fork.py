import torch
import torch.nn as nn
from copy import copy, deepcopy
from chop.ir.graph import MaseMetadata


class ForkIdentity(torch.nn.Module):
    """
    Toy quantized FC model for digit recognition on MNIST
    """

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Identity()
        self.layer2 = nn.Identity()

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        return x1, x2


@torch.fx.wrap
def fork2(x):
    out = x
    return out


def updata_common_metadata(node, quan_args):
    node.meta["mase"].parameters["common"]["mase_type"] = "call_function"
    node.meta["mase"].parameters["common"]["mase_op"] = "fork2"

    common_results = node.meta["mase"].parameters["common"]["results"]["data_out_0"]
    common_results["precision"] = quan_args
    node.meta["mase"].parameters["common"]["results"]["data_out_0"] = common_results
    node.meta["mase"].parameters["common"]["results"]["data_out_1"] = common_results
    node.meta["mase"].parameters["common"]["args"]["data_in_0"] = common_results

    node.meta["mase"].parameters["hardware"]["is_implicit"] = False


def deepcopy_mase_metadata(new_node, node):
    new_node.meta["mase"] = MaseMetadata(new_node, node.meta["mase"].model)
    new_node.meta["mase"] = MaseMetadata(new_node, node.meta["mase"].model)
    new_node.meta["mase"].parameters["common"] = deepcopy(node.meta["mase"]["common"])
    new_node.meta["mase"].parameters["hardware"] = deepcopy(
        node.meta["mase"]["hardware"]
    )
    new_node.meta["mase"].parameters["software"] = deepcopy(
        node.meta["mase"]["software"]
    )


def insert_fork_transform_pass(graph, pass_args={}):
    """Insert hardware-explicit forks into the mase graph
    :param graph: a MaseGraph
    :type graph: MaseGraph
    :param pass_args: this pass requires additional arguments which is explained below, defaults to {}
    :type pass_args: _type_, optional
    :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
    :rtype: tuple(MaseGrap`h, Dict)
    """

    nodes_to_fork = []
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
            deepcopy_mase_metadata(new_node, node)
            updata_common_metadata(new_node, quan_args=pass_args[new_node.name])

    graph.fx_graph.lint()
    return graph, None
