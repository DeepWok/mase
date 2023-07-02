from copy import copy, deepcopy

from chop.tools.logger import getLogger

from ...utils import get_parent_name, match_a_pattern
from .modify import create_new_fn, create_new_module

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


def graph_iterator_quantize_by_type(graph, config: dict):
    for node in graph.fx_graph.nodes:
        if node.meta["mase"].parameters["common"]["mase_op"] not in QUANTIZEABLE_OP:
            continue
        node_config = get_config(
            config, node.meta["mase"].parameters["common"]["mase_op"]
        )
        if node_config["name"] is None:
            continue
        if node.meta["mase"].parameters["common"]["mase_type"] == "module":
            ori_module = node.meta["mase"].module
            new_module = create_new_module(
                node.meta["mase"].parameters["common"]["mase_op"],
                ori_module,
                node_config,
            )
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
            # TODO: update meta_data
        elif node.meta["mase"].parameters["common"]["mase_type"] in [
            "builtin_func",
            "module_related_func",
        ]:
            new_f, args, kwargs = create_new_fn(node, node_config)
            with graph.fx_graph.inserting_before(node):
                new_node = graph.fx_graph.call_function(new_f, args, kwargs)
                new_node.name = node.name
                new_node.meta["mase"] = copy(node.meta["mase"])
                # TODO: update meta_data
                node.replace_all_uses_with(new_node)
            graph.fx_graph.erase_node(node)
    return graph


def graph_iterator_quantize_by_name(graph, config: dict):
    for node in graph.fx_graph.nodes:
        if node.meta["mase"].parameters["common"]["mase_op"] not in QUANTIZEABLE_OP:
            continue
        node_config = get_config(config, node.name)
        if node_config["name"] is None:
            continue
        if node.meta["mase"].parameters["common"]["mase_type"] == "module":
            ori_module = node.meta["mase"].module
            new_module = create_new_module(
                node.meta["mase"].parameters["common"]["mase_op"],
                ori_module,
                node_config,
            )
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
            # TODO: update meta_data
        elif node.meta["mase"].parameters["common"]["mase_type"] in [
            "builtin_func",
            "module_related_func",
        ]:
            new_f, args, kwargs = create_new_fn(node, node_config)
            with graph.fx_graph.inserting_before(node):
                new_node = graph.fx_graph.call_function(new_f, args, kwargs)
                new_node.name = node.name
                new_node.meta["mase"] = copy(node.meta["mase"])
                # TODO: update meta_data
                node.replace_all_uses_with(new_node)
            graph.fx_graph.erase_node(node)
        else:
            raise ValueError(
                "Unsupported node type for quantisation: {}".format(
                    node.meta["mase"].parameters["common"]["mase_type"]
                )
            )
    return graph


def graph_iterator_quantize_by_regex_name(graph, config: dict):
    patterns = list(config.keys())
    for node in graph.fx_graph.nodes:
        if node.meta["mase"].parameters["common"]["mase_op"] not in QUANTIZEABLE_OP:
            continue
        matched_pattern = match_a_pattern(node.name, patterns)
        if not matched_pattern:
            node_config = get_config(config, "default")
        else:
            node_config = get_config(config, matched_pattern)
        if node_config["name"] in [None, "Python None", "NA"]:
            continue
        if node.meta["mase"].parameters["common"]["mase_type"] == "module":
            ori_module = graph.modules[node.target]
            new_module = create_new_module(
                node.meta["mase"].parameters["common"]["mase_op"],
                ori_module,
                node_config,
            )
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
            # TODO: update meta_data
        elif node.meta["mase"].parameters["common"]["mase_type"] in [
            "builtin_func",
            "module_related_func",
        ]:
            new_f, args, kwargs = create_new_fn(node, node_config)
            with graph.fx_graph.inserting_before(node):
                new_node = graph.fx_graph.call_function(new_f, args, kwargs)
                new_node.name = node.name
                new_node.meta["mase"] = deepcopy(node.meta["mase"])
                # TODO: update meta_data
                node.replace_all_uses_with(new_node)
            graph.fx_graph.erase_node(node)
        else:
            raise ValueError(
                "Unsupported node type for quantisation:{}".format(
                    node.meta["mase"].parameters["common"]["mase_type"]
                )
            )
    return graph


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
