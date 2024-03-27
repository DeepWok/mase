from copy import copy, deepcopy
import logging
from torch import nn

from chop.passes.graph.interface.save_and_load import load_mase_graph_interface_pass

from ...utils import (
    deepcopy_mase_graph,
    get_mase_op,
    get_mase_type,
    get_node_actual_target,
    get_parent_name,
    get_similar_node_actual_target,
    match_a_pattern,
    get_node_target_by_name,
)

from .modify import create_new_fn, create_new_module
from .quant_parsers import parse_node_config, relink_node_meta, update_quant_meta_param
from .summary import graph_iterator_compare_nodes, graph_iterator_node_histogram

logger = logging.getLogger(__name__)

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


# def update_quant_meta_param(*args, **kwargs):
#     # TODO: remove this function when the add_common_metadata is fixed
#     pass


def graph_iterator_quantize_by_type(graph, config: dict):
    # Some modules might need information from two graphs to be initilized
    if (
        config.get("baseline_weight_path") is not None
        and config.get("load_type") == "mz"
    ):
        bl_graph = deepcopy_mase_graph(graph)
        bl_graph = load_mase_graph_interface_pass(
            bl_graph, pass_args=config.get("baseline_weight_path")
        )
    else:
        bl_graph = None
    for node in graph.fx_graph.nodes:
        if get_mase_op(node) not in QUANTIZEABLE_OP:
            continue
        node_config = get_config(config, get_mase_op(node))
        if node_config["name"] is None:
            continue
        node_config = parse_node_config(node_config, get_mase_op(node))
        # if get_mase_type(node) == "module":
        if node.op == "call_module":
            ori_module = get_node_actual_target(node)
            successor_module = get_similar_node_actual_target(
                bl_graph, node.next
            )  # Certain modules will require information about their successor module to complete the initialization process. (For LogicNets, activation functions are needed.)
            bl_module = get_similar_node_actual_target(bl_graph, node)
            new_module = create_new_module(
                get_mase_op(node),
                ori_module,
                node_config,
                node.meta,
                bl_module,
                successor_module,
            )
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
        output_layers_names = node_config.get("additional_layers_outputs", [])
        output_layers = [
            get_node_target_by_name(graph, name) for name in output_layers_names
        ]
        input_layers_names = node_config.get("additional_layers_inputs", [])
        input_layers = [
            get_node_target_by_name(graph, name) for name in input_layers_names
        ]
        if node.op == "call_module":
            ori_module = get_node_actual_target(node)
            new_module = create_new_module(
                get_mase_op(node),
                ori_module,
                node_config,
                node.meta,
                input_layers=input_layers,
                output_layers=output_layers,
            )
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
            update_quant_meta_param(node, node_config, get_mase_op(node))
            logger.debug(f"Quantized module: {node.target} with config: {node_config}")
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
            logger.debug(
                f"Quantized function: {node.target} with config: {node_config}"
            )
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
        # if get_mase_type(node) == "module":
        if node.op == "call_module":
            ori_module = graph.modules[node.target]
            new_module = create_new_module(
                get_mase_op(node), ori_module, node_config, node.meta
            )
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

def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)



def redefine_linear_transform_pass(graph, pass_args=None):
    main_config = pass_args.pop('config')
    #import pdb; pdb.set_trace()
    default = {'config': {'name': None}}
    if default is None:
        raise ValueError(f"default value must be provided.")
    i = 0
    pre_in=1
    pre_out=1
    for node in graph.fx_graph.nodes:
        i += 1
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)
        import pdb; pdb.set_trace()
        #print(name)
        if name is not None:
            #if node.target == 'x':
            	#continue
            #import pdb; pdb.set_trace()
            ori_module = graph.modules[node.target]
            in_features = 16
            out_features = 16
            bias = ori_module.bias
            if name == "output_only":
                out_features = out_features * config["channel_multiplier"]
                pre_out=config["channel_multiplier"]
            elif name == "both":
                in_features = in_features * pre_out
                out_features = out_features * config["channel_multiplier"]
                pre_in=config["channel_multiplier"]
            elif name == "input_only":
                in_features = in_features * pre_in
                out_features = 5
                #import pdb; pdb.set_trace()
            new_module = instantiate_linear(in_features, out_features, bias)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
    return graph, {}


def quantize_transform_pass(graph, pass_args=None):
    """
    Apply quantization transformation to the given graph.

    :param graph: The input graph to be transformed.
    :type graph: MaseGraph

    :param pass_args: Additional arguments for the transformation.
    :type pass_args: dict, optional

    :return: The transformed graph.
    :rtype: tuple
    :raises ValueError: If the quantize "by" argument is unsupported.


    - pass_args
        - by -> str : different quantization schemes choose from ["type", "name", "regx_name"]
    """

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
    return graph, {}
