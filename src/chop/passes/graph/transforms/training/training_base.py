from functools import partial
from chop.passes.graph.transforms.training.modify import create_new_module
import torch

from chop.ir.common import MASE_TYPE_MAP
from chop.passes.graph.utils import (
    get_mase_op,
    get_mase_type,
    get_node_actual_target,
    get_parent_name,
)


# empty for now
EDITABLE_OPS = [
    "linear",
]


def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]


def graph_iterator_by_type(graph, config: dict):
    for node in graph.fx_graph.nodes:
        mase_op = get_mase_op(node)
        if mase_op not in EDITABLE_OPS:
            continue
        node_config = get_config(config, mase_op)

        if node.op == "call_module":
            ori_module = get_node_actual_target(node)
            if mase_op == "linear":
                new_module = create_new_module(
                    get_mase_op(node),
                    ori_module,
                    node_config,
                    node.meta,
                )
                parent_name, name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], name, new_module)
            else:
                raise NotImplementedError("Not implemented yet")

        elif get_mase_type(node) in [
            "builtin_func",
            "module_related_func",
        ]:
            raise NotImplementedError("Not implemented yet")

    return graph


def training_base_pass(graph, pass_args: dict = {}):
    """
    Apply training transformation to the given graph.

    :param graph: The input graph to be transformed.
    :type graph: MaseGraph

    :param pass_args: Additional arguments for the transformation.
    :type pass_args: dict, optional

    .. code-block: python

            "by": "type",  # transform by type. NOTE: transformation based on type only for now
        quan_args = {
            "default": {"config": {"name": None}}, # default config, this would be used for any node that does not have a specific config
            "linear": {
                "config": {
                    "forward": { # specifed the forward pass of the custom module
                        "pass": "quantize", # catagory of the forward pass
                        "name": "integer",  # specific configurations of the pass
                        "weight_width": 10,
                        "weight_frac_width": 5,
                        "data_in_width": 10,
                        "data_in_frac_width": 5,
                        "bias_width": 10,
                        "bias_frac_width": 5,
                        "data_out_width": 10,
                        "data_out_frac_width": 5,
                    },
                    "backward": { # specifed the backward pass of the custom module
                        "pass": "quantize", # catagory of the backward pass
                        "name": "integer",  # specific configurations of the pass
                        "output_grad_width": 10,
                        "output_grad_frac_width": 5,
                        "data_in_width": 10,
                        "data_in_frac_width": 5,
                        "weight_width": 10,
                        "weight_frac_width": 5,
                        "bias_width": 10,
                        "bias_frac_width": 5,
                    },
                }
            },
        }

    :return: The transformed graph.
    :rtype: tuple
    :raises ValueError: If the quantize "by" argument is unsupported.


    - pass_args
        - by -> str : different transformation schemes choose from ["type"]
    """
    by = pass_args.pop("by")
    match by:
        case "type":
            graph = graph_iterator_by_type(graph, pass_args)
        # case "name":
        #     graph = graph_iterator_by_name(graph, pass_args)
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')

    # link the model with graph
    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)
    return graph, {}
