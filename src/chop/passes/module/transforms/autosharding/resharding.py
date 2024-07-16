from chop.tools import get_logger

logger = get_logger(__name__)
logger.setLevel("INFO")


def resharding_transform_pass(mg, pass_args={}):
    """
    This pass inserts a wrapper around each module in the graph to handle resharding
    activation tensors when the output of the previous module has a different sharding
    profile to the one assigned to the current module.
    """

    module_map = pass_args.get("module_map", None)
    device_mesh = pass_args.get("device_mesh", None)
    if module_map is None or device_mesh is None:
        raise ValueError(
            "module_map and device_mesh are required for resharding_transform_pass"
        )

    for node in mg.fx_graph.nodes:
        pass

    mg.model.recompile()

    return mg, {}
