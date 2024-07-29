from copy import copy

import torch
import torch.fx as fx
from torch.distributed._tensor.placement_types import Replicate, Shard

from chop.tools import get_logger
from chop.nn.functional.dtensor import redistribute_dtensor
from chop.ir.graph import MaseMetadata

logger = get_logger(__name__)
logger.setLevel("INFO")


def _insert_resharding_nodes(mg, pass_args={}):
    """Insert resharding nodes"""
    logger.info(
        f"Running resharding_transform_pass to insert resharding nodes along necessary edges."
    )
    for node in mg.fx_graph.nodes:

        if node.op == "call_function" and node.target == redistribute_dtensor:
            continue

        flattened_args = node.args + tuple(node.kwargs.values())
        kwarg_keys = list(node.kwargs.keys())

        # Number of arguments should match metadata
        if node.op != "output" and len(flattened_args) != len(
            node.meta["mase"]["common"]["args"]
        ):
            logger.warning(
                f"Skipping node: {node.name} because number of arguments do not match metadata."
            )
            continue

        for arg_idx, arg_name in enumerate(node.meta["mase"]["common"]["args"].keys()):

            # Check if argument is an FX node, otherwise it's a constant
            arg_obj = flattened_args[arg_idx]
            if not isinstance(arg_obj, fx.Node):
                logger.debug(
                    f"Skipping node: {node.name}, argument: {arg_name} because it is a constant."
                )
                continue

            # Check if the parent node output spec is different from the arg input spec
            arg_info = node.meta["mase"]["common"]["args"][arg_name]
            arg_specs = arg_info.get("dtensor_spec", None)
            parent_out_specs = arg_obj.meta["mase"]["common"]["results"][
                "data_out_0"
            ].get("dtensor_spec", None)

            if arg_specs is None or parent_out_specs is None:
                logger.warning(
                    f"Skipping edge {arg_obj} -> {node}.{arg_name} because dtensor_spec was not found"
                )
                continue

            if arg_specs.placements != parent_out_specs.placements:
                logger.info(
                    f"Inserting resharding node along edge {arg_obj} -> {node.name} because arg {arg_name} requires placement {arg_specs.placements} but parent node {arg_obj.name} has placement {parent_out_specs.placements}."
                )

                # Create resharding node
                with mg.fx_graph.inserting_before(node):
                    resharding_node = mg.fx_graph.call_function(
                        redistribute_dtensor,
                        args=(arg_obj, arg_specs.placements),
                        kwargs={
                            "async_op": False,
                        },
                    )

                resharding_node.meta["mase"] = MaseMetadata(
                    node=resharding_node,
                    model=mg.model,
                )

                # Update the current node's argument
                # Node arg can be referenced in either node.args or node.kwargs so we
                # infer which container to update based on the arg_idx value, which
                # indexes the combined list of args and kwargs
                if arg_idx < len(node.args):
                    updated_args = list(node.args)
                    updated_args[arg_idx] = resharding_node
                    node.args = tuple(updated_args)
                else:
                    kwarg_idx = arg_idx - len(node.args)
                    arg_key = kwarg_keys[kwarg_idx]
                    kwarg_dict = {}

                    # Reconstruct they node.kwargs dict since this is immutable
                    for key, value in node.kwargs.items():
                        if key == arg_key:
                            kwarg_dict[key] = resharding_node
                        else:
                            kwarg_dict[key] = value
                    node.kwargs = kwarg_dict

    # Insert DTensor import at the top of code
    def insert_imports(body):
        return [
            "from torch.distributed._tensor.placement_types import Replicate, Shard, Partial; sum = 'sum' \n",
            *body,
        ]

    mg.fx_graph.on_generate_code(lambda _: insert_imports)

    # Check the model is valid
    mg.fx_graph.lint()
    mg.model.recompile()

    return mg, {}


def resharding_transform_pass(mg, pass_args={}):
    return _insert_resharding_nodes(mg, pass_args)
