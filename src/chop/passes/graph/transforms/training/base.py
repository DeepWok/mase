from functools import partial
from chop.nn.backward.modules.linear import QLinear
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


def graph_iterator_by_type(graph, config: dict):
    for node in graph.fx_graph.nodes:
        mase_op = get_mase_op(node)
        if mase_op not in EDITABLE_OPS:
            continue

        node_config = config.get(mase_op, None)
        if node_config["name"] is None:
            continue

        if node.op == "call_module":
            ori_module = get_node_actual_target(node)
            if mase_op == "linear":
                new_module = create_new_module(
                    get_mase_op(node),
                    ori_module,
                    node_config,
                    node.meta,
                )
                new_module = QLinear(
                    ori_module.in_features,
                    ori_module.out_features,
                    bias=ori_module.bias is not None,
                    extra_params=node_config,
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
