from chop.tools import get_logger

from .alpa_intra_operator import alpa_intra_op_sharding_pass

logger = get_logger(__name__)
logger.setLevel("DEBUG")


def deepgetattr(obj, attr, default=None):
    """Recurses through an attribute chain to get the ultimate value."""
    try:
        return functools.reduce(getattr, attr.split("."), obj)
    except AttributeError:
        return default


def get_node_target(node):
    if isinstance(node.target, str):
        return deepgetattr(node.meta["mase"].model, node.target, None)
    else:
        return node.target


def mark_choices(mg):
    """
    Once the metadata has already been filled for each op with the possible shardings and costs,
    and the ILP has been solved, this function marks the chosen sharding for each op.
    """
    for node in mg.fx_graph.nodes:
        chosen_idx = (
            0
            if isinstance(
                node.meta["mase"]["software"]["autosharding"]["opt_var"], np.ndarray
            )
            else np.where(
                node.meta["mase"]["software"]["autosharding"]["opt_var"].value == 1
            )[0][0]
        )
        node.meta["mase"]["software"]["autosharding"]["input_sharding"] = node.meta[
            "mase"
        ]["software"]["autosharding"]["valid_input_shardings"][chosen_idx]
        node.meta["mase"]["software"]["autosharding"]["output_sharding"] = node.meta[
            "mase"
        ]["software"]["autosharding"]["valid_output_shardings"][chosen_idx]
        chosen_sharding = {
            key: node.meta["mase"]["software"]["autosharding"]["input_sharding"][key]
            for key in node.meta["mase"]["software"]["autosharding"][
                "input_sharding"
            ].keys()
        }

        # Write into module map (used by distributed launcher)
        target = get_node_target(node)
        if node.op == "call_module" and target is not None:
            module_map[target] = {"node": node.name, "sharding": chosen_sharding}
            module_map[target]["sharding"]["output"] = node.meta["mase"]["software"][
                "autosharding"
            ]["output_sharding"]

    return mg


def alpa_autosharding_pass(mg, mesh, pass_args={}):
    mg, module_map = alpa_intra_op_sharding_pass(mg, mesh, pass_args=pass_args)
    return mg, module_map
