import torch
import torch.fx as fx
import torch.nn as nn


def raise_granularity_transform_pass(mg, pass_args={}):
    """Pass to raise the granularity of an FX graph from matmul to module level. FX graphs generated through the ONNX backend contain call_function nodes like reshape, gather, view, etc, which we cannot (currently) emit verilog for directly. This pass raises the granularity so the user can emit Verilog code at module level, avoiding the need to handle implicit nodes in hardware. The submodule structure is taken from the existing model in the MaseGraph, and nodes are expected to have the following naming standard:

    node_name: <submodule name>_<mase_op>

    Args:
        mg (MaseGraph): input MaseGraph
        pass_args (dict, optional): pass arguments. Defaults to {}.

    Raises:
        RuntimeError: When the FX graph generation fails for a given submodule.

    Returns:
        MaseGraph: output MaseGraph at module granularity.
    """

    # * Generate dictionary of leaf submodules
    leaf_modules = {
        i[0].replace(".", "_"): i[1]
        for i in mg.model.named_modules()
        if not any(i[1].named_children())
    }

    for submodule_name, submodule in leaf_modules.items():
        # * Initialize new GraphModule
        sm_fx_graph = fx.Graph()
        sm_fx_graph_nodes = {}
        sm_gm = fx.GraphModule(nn.Module(), sm_fx_graph)

        submodule_users = {}

        # Copy parameters into the new submodule
        for name, value in submodule.state_dict().items():
            setattr(sm_gm, name, torch.nn.Parameter(value))

        # * List of nodes belonging to this submodule, filtered by name
        node_group = [node for node in mg.fx_graph.nodes if submodule_name in node.name]

        # * Mapping function to create new placeholder nodes when the kwarg node is outside the new graph
        def map_kwarg_to_placeholder(arg_node):
            if arg_node in node_group:
                return sm_fx_graph_nodes[arg_node.name]
            else:
                return sm_fx_graph.create_node(
                    op="placeholder",
                    name=f"{arg_node.name}_input",
                    target=f"{arg_node.name}_input",
                )

        # * Copy nodes into new graph submodule
        for node in node_group:
            new_node = sm_fx_graph.node_copy(
                node, arg_transform=map_kwarg_to_placeholder
            )
            sm_fx_graph_nodes[new_node.name] = new_node

            # ? If new node is get_attr, update target from absolute module path to local path
            # ? This means leaf submodules cannot have hierarchical parameters. Is this valid?
            if new_node.op == "get_attr":
                new_node.target = new_node.target.split(".")[-1]

            # * If there is at least one external node consuming this node,
            # * add this as a submodule output
            if len([user for user in node.users.keys() if user not in node_group]) > 0:
                new_output = sm_fx_graph.create_node(
                    op="output",
                    name=f"{node.name}_output",
                    target=f"{node.name}_output",
                )
                new_output.args = [new_node]
                sm_fx_graph_nodes[new_output.name] = new_output

        # if submodule_name == "bert_encoder_layer_11_attention_self_query":
        #     breakpoint()

        try:
            sm_fx_graph.lint()
        except:
            raise RuntimeError(
                f"Error generating GraphModule for submodule {submodule_name}"
            )

        # * Replace submodule in mg.model
        setattr(mg.model, submodule_name, sm_gm)

        # * Add new call_module node to original graph and delete old nodes
        mg.fx_graph.create_node(
            op="call_module", name=submodule_name, target=submodule_name
        )

        # * Remove old nodes associated with this submodule

    # Final consistency checks
    try:
        mg.fx_graph.lint()
    except:
        raise RuntimeError(f"Error updating MaseGraph")

    return mg, {}
