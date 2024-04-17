from chop.ir.onnx import MaseOnnxGraph
from chop.ir.graph import MaseGraph

import torch
import torch.nn as nn
import torch.fx as fx
import onnx

from .utils import clean_name, init_submodule, deserialize_constant, add_kwarg

from chop.ir.onnx import ONNX_OP_MAPPING

# * Steps
# * ------------------------------


def _parse_attributes(
    onnx_graph: MaseOnnxGraph,
    graph_module: fx.GraphModule,
    fx_graph: fx.Graph,
    fx_nodes: dict,
):
    """Parse ONNX graph attributes and initialize torch module parameters.

    Args:
        onnx_graph (MaseOnnxGraph): _description_
        graph_module (fx.GraphModule): _description_
        fx_graph (fx.Graph): _description_
        fx_nodes (dict): _description_
    """
    for attr_path in onnx_graph.initializer_attributes.keys():
        module_str = ".".join(attr_path.split(".")[:-1])
        attr_name = attr_path.replace("::", "_").split(".")[-1]

        # Set torch module parameter
        init_submodule(graph_module, module_str)
        setattr(
            graph_module.get_submodule(module_str),
            attr_name,
            torch.nn.Parameter(
                torch.from_numpy(
                    onnx.numpy_helper.to_array(
                        onnx_graph.initializer_attributes[attr_path]
                    )
                )
            ),
        )

        # Create get_attr node
        node_name = clean_name(f"{module_str}_{attr_name}")
        node_name = node_name[1:] if node_name[0] == "_" else node_name
        new_node = fx_graph.create_node(
            op="get_attr",
            name=node_name,
            target=f"{module_str}.{attr_name}" if module_str != "" else attr_name,
        )
        fx_nodes[node_name] = new_node

    return fx_nodes


def _parse_constants(
    onnx_graph: MaseOnnxGraph,
    graph_module: fx.GraphModule,
    fx_graph: fx.Graph,
    fx_nodes: dict,
):
    """Parse ONNX graph constants and initialize torch module parameters.

    Args:
        onnx_graph (MaseOnnxGraph): _description_
        graph_module (fx.GraphModule): _description_
        fx_graph (fx.Graph): _description_
        fx_nodes (dict): _description_
    """
    for constant_name, constant in onnx_graph.constants.items():
        module_str = ".".join(constant_name.split("/")[1:-1])
        node_name = clean_name(constant_name)
        node_name = node_name[1:] if node_name[0] == "_" else node_name

        # Set torch module parameter
        init_submodule(graph_module, module_str)
        setattr(
            graph_module.get_submodule(module_str),
            node_name,
            torch.nn.Parameter(
                data=deserialize_constant(constant), requires_grad=False
            ),
        )

        # Create get_attr node
        new_node = fx_graph.create_node(
            op="get_attr",
            name=node_name,
            target=f"{module_str}.{node_name}" if module_str != "" else node_name,
        )
        fx_nodes[node_name] = new_node

    return fx_nodes


def _initialize_nodes(
    onnx_graph: MaseOnnxGraph,
    graph_module: fx.GraphModule,
    fx_graph: fx.Graph,
    fx_nodes: dict,
):
    # Create fx placeholders from ONNX graph inputs
    for in_node in onnx_graph.graph.input:
        name = clean_name(in_node.name)
        new_node = fx_graph.create_node(op="placeholder", name=name, target=name)
        fx_nodes[name] = new_node

    # Initialize all nodes
    for onnx_node in onnx_graph.graph.node:

        # Don't register Constant nodes
        if onnx_node.op_type == "Constant":
            continue

        if "name_override" in ONNX_OP_MAPPING[onnx_node.op_type].keys():
            onnx_node.name = onnx_node.name.replace(
                onnx_node.op_type, ONNX_OP_MAPPING[onnx_node.op_type]["name_override"]
            )

        # Register submodule
        node_module = ".".join(onnx_node.name.split("/")[1:-1])
        init_submodule(graph_module, node_module)  # in case not yet initialized

        # Create node in FX graph
        name = clean_name(onnx_node.name)
        new_node = fx_graph.create_node(
            op=ONNX_OP_MAPPING[onnx_node.op_type]["fx_op"],
            name=name,
            target=ONNX_OP_MAPPING[onnx_node.op_type]["target"],
        )
        fx_nodes[name] = new_node

    # Create output nodes for ONNX graph outputs
    for out_node in onnx_graph.graph.output:
        name = clean_name(out_node.name)
        new_node = fx_graph.create_node(op="output", name=name, target=name)
        input_onnx_node = onnx_graph.edge_mapping[out_node.name]
        new_node.args = [fx_nodes[clean_name(input_onnx_node.name)]]
        fx_nodes[name] = new_node

    return fx_nodes


def _map_onnx_node_inputs(
    onnx_node: onnx.NodeProto,
    fx_node_name: str,
    onnx_graph: MaseOnnxGraph,
    graph_module: fx.GraphModule,
    fx_graph: fx.Graph,
    fx_nodes: dict,
):
    for input_idx, input_name in enumerate(onnx_node.input):

        # Variadic inputs: input list of arbitrary length that maps to a single torch argument
        if len(onnx_node.input) > len(
            ONNX_OP_MAPPING[onnx_node.op_type]["input_mapping"]
        ):
            # ? It may be possible to have a variadic input + fixed length input
            torch_arg_name = ONNX_OP_MAPPING[onnx_node.op_type]["input_mapping"][0]
        else:
            torch_arg_name = ONNX_OP_MAPPING[onnx_node.op_type]["input_mapping"][
                input_idx
            ]

        # ONNX input is not mapped to a torch input, so skip
        if torch_arg_name == "":
            continue

        # * (A) Input from model parameters mapped from onnx graph attributes, i.e. get_attr fx node
        if input_name in onnx_graph.initializer_attributes.keys():
            add_kwarg(
                fx_nodes[fx_node_name],
                torch_arg_name,
                fx_nodes[clean_name(input_name.replace(f"::", "_"))],
            )

        # * (B) Input from onnx constant node, i.e. get_attr fx node
        elif onnx_graph.edge_mapping[input_name].name in onnx_graph.constants.keys():
            add_kwarg(
                fx_nodes[fx_node_name],
                torch_arg_name,
                fx_nodes[clean_name(onnx_graph.edge_mapping[input_name].name)],
            )

        # * (C) Input from another node
        elif input_name in onnx_graph.edge_mapping.keys():
            # Call method nodes in an FX graph expect the first argument to be the parent node to call the method on
            if (
                ONNX_OP_MAPPING[onnx_node.op_type]["fx_op"] == "call_method"
                and input_idx == 0
            ):
                fx_nodes[fx_node_name].args += (
                    fx_nodes[clean_name(onnx_graph.edge_mapping[input_name].name)],
                )

            else:
                add_kwarg(
                    fx_nodes[fx_node_name],
                    torch_arg_name,
                    fx_nodes[clean_name(onnx_graph.edge_mapping[input_name].name)],
                )

        else:
            raise RuntimeError(
                f"Unrecognized input {input_name} for ONNX node {onnx_node.name}."
            )

    return fx_nodes


def _map_onnx_node_attributes(
    onnx_node: onnx.NodeProto,
    fx_node_name: str,
    onnx_graph: MaseOnnxGraph,
    graph_module: fx.GraphModule,
    fx_graph: fx.Graph,
    fx_nodes: dict,
):
    for attribute in onnx_node.attribute:
        torch_arg_name = ONNX_OP_MAPPING[onnx_node.op_type]["attribute_mapping"][
            attribute.name
        ]

        # ONNX attribute is not mapped to a torch input, so skip
        if torch_arg_name == "":
            continue

        # ! TO DO: consider other types of attributes
        attr = onnx.helper.get_attribute_value(attribute)
        if attribute.type == onnx.AttributeProto.TENSOR:
            attr = torch.from_numpy(onnx.numpy_helper.to_array(attr))

        if (
            "attribute_transform" in ONNX_OP_MAPPING[onnx_node.op_type].keys()
            and ONNX_OP_MAPPING[onnx_node.op_type]["attribute_transform"][
                attribute.name
            ]
            is not None
        ):
            attr = ONNX_OP_MAPPING[onnx_node.op_type]["attribute_transform"][
                attribute.name
            ](attr)

        add_kwarg(fx_nodes[fx_node_name], torch_arg_name, attr)

    # Handle case when not all attributes are included in ONNX node
    if len(onnx_node.attribute) < len(
        ONNX_OP_MAPPING[onnx_node.op_type]["attribute_mapping"].keys()
    ):
        node_attributes = [attr.name for attr in onnx_node.attribute]
        for onnx_attr, torch_arg_name in ONNX_OP_MAPPING[onnx_node.op_type][
            "attribute_mapping"
        ].items():
            # Check that attribute hasn't already been mapped
            if (
                torch_arg_name not in fx_nodes[fx_node_name].kwargs.keys()
                and torch_arg_name != ""
            ):
                attr = ONNX_OP_MAPPING[onnx_node.op_type]["attribute_default"][
                    onnx_attr
                ]
                add_kwarg(fx_nodes[fx_node_name], torch_arg_name, attr)

    return fx_nodes


def _map_onnx_node_arguments(
    onnx_graph: MaseOnnxGraph,
    graph_module: fx.GraphModule,
    fx_graph: fx.Graph,
    fx_nodes: dict,
):
    # * ONNX node arguments can be represented as "inputs" or "attributes".
    for onnx_node in onnx_graph.graph.node:

        # Don't register Constant nodes
        if onnx_node.op_type == "Constant":
            continue

        if onnx_node.op_type not in ONNX_OP_MAPPING.keys():
            raise ValueError(
                f"Unrecognized ONNX operator: {onnx_node.op_type}. To add support, include this in chop.ir.onnx.ONNX_OP_MAPPING."
            )

        fx_node_name = clean_name(onnx_node.name)

        # * (1) First we map "inputs", which can come from initializer attributes (i.e. module parameters), ONNX constant
        # *   or another node (through edge mapping)
        fx_nodes = _map_onnx_node_inputs(
            onnx_node, fx_node_name, onnx_graph, graph_module, fx_graph, fx_nodes
        )

        # * (2) Now map node attributes to kwargs
        fx_nodes = _map_onnx_node_attributes(
            onnx_node, fx_node_name, onnx_graph, graph_module, fx_graph, fx_nodes
        )

    return fx_nodes


# * ONNX to FX Translation Pass
# * ------------------------------


def export_fx_graph_analysis_pass(onnx_model_dict, pass_args=None):
    """Receives a MaseOnnxGraph and pass_args, and returns a MaseGraph.

    Args:
        onnx_graph (_type_): _description_
        pass_args (_type_): _description_
    """

    # assert isinstance(
    #     onnx_graph, MaseOnnxGraph
    # ), f"Expected MaseOnnxGraph, got {type(onnx_graph)}"

    mase_graphs = {}
    for model_name, onnx_model in onnx_model_dict.items():
        onnx_graph = MaseOnnxGraph(onnx_model, model_name=model_name)

        fx_graph = fx.Graph()
        gm = fx.GraphModule(nn.Module(), fx_graph)

        # fx.graph.nodes is not subscriptable, so we maintain this dict as new nodes are added
        fx_nodes = {}

        fx_nodes = _parse_attributes(onnx_graph, gm, fx_graph, fx_nodes)
        fx_nodes = _parse_constants(onnx_graph, gm, fx_graph, fx_nodes)
        fx_nodes = _initialize_nodes(onnx_graph, gm, fx_graph, fx_nodes)
        fx_nodes = _map_onnx_node_arguments(onnx_graph, gm, fx_graph, fx_nodes)

        mg = MaseGraph(model=gm)

        # Recompile the code after graph transformations
        mg.model.recompile()

        mase_graphs[model_name] = mg

    return mase_graphs, {}
