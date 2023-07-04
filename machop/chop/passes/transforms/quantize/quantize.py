import os
from copy import copy, deepcopy

from chop.passes.utils import (
    get_mase_op,
    get_mase_type,
    get_parent_name,
    match_a_pattern,
    node_actual_target,
)
from chop.tools.logger import getLogger

from .modify import create_new_fn, create_new_module
from .quant_parsers import parse_node_config, relink_node_meta, update_quant_meta_param
from .summary import graph_iterator_compare_nodes, graph_iterator_node_histogram

logger = getLogger(__name__)

QUANTIZEABLE_OP = (
    "add",
    "bmm",
    "conv1d",
    "conv2d",
    "matmul",
    "mul",
    "linear",
    "relu",
    "sub",
)


def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]


def update_quant_meta_param(*args, **kwargs):
    # TODO: remove this function when the add_common_metadata is fixed
    pass


def graph_iterator_compare_nodes(*args, **kwargs):
    # TODO: remove this function when the add_common_metadata is fixed
    pass


def graph_iterator_node_histogram(*args, **kwargs):
    # TODO: remove this function when the add_common_metadata is fixed
    pass


def graph_iterator_quantize_by_type(graph, config: dict):
    for node in graph.fx_graph.nodes:
        if get_mase_op(node) not in QUANTIZEABLE_OP:
            continue
        node_config = get_config(config, get_mase_op(node))
        if node_config["name"] is None:
            continue
        node_config = parse_node_config(node_config, get_mase_op(node))
        if get_mase_type(node) == "module":
            ori_module = node_actual_target(node)
            new_module = create_new_module(get_mase_op(node), ori_module, node_config)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
            # update precision and type in meta.parameters["common"]
            update_quant_meta_param(node, node_config, get_mase_op(node))
        elif get_mase_type(node) in [
            "builtin_func",
            "module_related_func",
        ]:
            new_f, args, kwargs = create_new_fn(node, node_config)
            with graph.fx_graph.inserting_before(node):
                new_node = graph.fx_graph.call_function(new_f, args, kwargs)
                new_node.name = node.name
                new_node.meta["mase"] = copy(node.meta["mase"])
                # new_node.meta["mase"].node -> new_node
                relink_node_meta(new_node, model=graph.model)
                update_quant_meta_param(new_node, node_config, get_mase_op(node))
                node.replace_all_uses_with(new_node)
            graph.fx_graph.erase_node(node)
    return graph


def graph_iterator_quantize_by_name(graph, config: dict):
    for node in graph.fx_graph.nodes:
        if get_mase_op(node) not in QUANTIZEABLE_OP:
            continue
        node_config = get_config(config, node.name)
        if node_config["name"] is None:
            continue
        node_config = parse_node_config(node_config, get_mase_op(node))
        if get_mase_type(node) == "module":
            ori_module = node_actual_target(node)
            new_module = create_new_module(get_mase_op(node), ori_module, node_config)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
            update_quant_meta_param(node, node_config, get_mase_op(node))
        elif get_mase_type(node) in [
            "builtin_func",
            "module_related_func",
        ]:
            new_f, args, kwargs = create_new_fn(node, node_config)
            with graph.fx_graph.inserting_before(node):
                new_node = graph.fx_graph.call_function(new_f, args, kwargs)
                new_node.name = node.name
                new_node.meta["mase"] = copy(node.meta["mase"])
                relink_node_meta(new_node, model=graph.model)
                update_quant_meta_param(new_node, node_config, get_mase_op(node))
                node.replace_all_uses_with(new_node)
            graph.fx_graph.erase_node(node)
        else:
            raise ValueError(
                "Unsupported node type for quantisation: {}".format(get_mase_type(node))
            )
    return graph


def graph_iterator_quantize_by_regex_name(graph, config: dict):
    patterns = list(config.keys())
    for node in graph.fx_graph.nodes:
        if get_mase_op(node) not in QUANTIZEABLE_OP:
            continue
        matched_pattern = match_a_pattern(node.name, patterns)
        if not matched_pattern:
            node_config = get_config(config, "default")
        else:
            node_config = get_config(config, matched_pattern)
        if node_config["name"] is None:
            continue
        node_config = parse_node_config(node_config, get_mase_op(node))
        if get_mase_type(node) == "module":
            ori_module = graph.modules[node.target]
            new_module = create_new_module(get_mase_op(node), ori_module, node_config)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
            update_quant_meta_param(node, node_config, get_mase_op(node))
        elif get_mase_type(node) in [
            "builtin_func",
            "module_related_func",
        ]:
            new_f, args, kwargs = create_new_fn(node, node_config)
            with graph.fx_graph.inserting_before(node):
                new_node = graph.fx_graph.call_function(new_f, args, kwargs)
                new_node.name = node.name
                new_node.meta["mase"] = deepcopy(node.meta["mase"])
                relink_node_meta(new_node, model=graph.model)
                update_quant_meta_param(new_node, node_config, get_mase_op(node))
                node.replace_all_uses_with(new_node)
            graph.fx_graph.erase_node(node)
        else:
            raise ValueError(
                "Unsupported node type for quantisation:{}".format(get_mase_type(node))
            )
    return graph


# def quantize_transform_pass(graph, pass_args=None):
#     if "report" not in pass_args:
#         report = True
#         logger.warning(
#             "The `report` argument is not provided in quantize pass config. Generate the report by default, "
#             "but running report creates a copy of model, which may lead to memory overflow if the model is huge."
#         )
#     else:
#         report = pass_args["report"]
#     if report:
#         ori_graph = deepcopy(graph)

#     save_dir = pass_args.pop("report_to", None)
#     if save_dir is not None:
#         os.makedirs(save_dir, exist_ok=True)

#     by = pass_args.pop("by")
#     match by:
#         case "type":
#             graph = graph_iterator_quantize_by_type(graph, pass_args)
#         case "name":
#             graph = graph_iterator_quantize_by_name(graph, pass_args)
#         case "regex_name":
#             graph = graph_iterator_quantize_by_regex_name(graph, pass_args)
#         case _:
#             raise ValueError(f'Unsupported quantize "by": {by}')

#     table_path = os.path.join(save_dir, "quantize_table.csv") if save_dir else None
#     histogram_path = (
#         os.path.join(save_dir, "quantize_histogram.csv") if save_dir else None
#     )
#     if report:
#         graph_iterator_compare_nodes(
#             ori_graph, graph, save_path=table_path, silent=False
#         )
#         graph_iterator_node_histogram(ori_graph, graph, save_path=histogram_path)
#     return graph


def quantize_transform_pass(graph, pass_args=None):
    by = pass_args.pop("by")
    match by:
        case "type":
            graph = graph_iterator_quantize_by_type(graph, pass_args)
        case "name":
            graph = graph_iterator_quantize_by_name(graph, pass_args)
        case "regex_name":
            graph = graph_iterator_quantize_by_regex_name(graph, pass_args)
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')
    return graph


def quantize_summary_analysis_pass(ori_graph, graph, save_dir: str = None) -> None:
    table_path = os.path.join(save_dir, "quantize_table.csv") if save_dir else None
    histogram_path = (
        os.path.join(save_dir, "quantize_histogram.csv") if save_dir else None
    )
    graph_iterator_compare_nodes(ori_graph, graph, save_path=table_path, silent=False)
    graph_iterator_node_histogram(ori_graph, graph, save_path=histogram_path)
