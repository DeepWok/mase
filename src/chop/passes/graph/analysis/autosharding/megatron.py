from chop.ir import MaseGraph
from .mesh_model import MeshModel


def megatron_autosharding_pass(
    mg: MaseGraph,
    mesh: MeshModel,
    pass_args: dict,
):
    for node in mg.fx_graph.nodes:
        meta = node.meta["mase"]["common"]

        for arg, arg_spec in meta["args"].items():
            if not isinstance(arg_spec, dict):
                continue
            arg_spec["dtensor_spec"] = None

        for result, result_spec in meta["results"].items():
            if not isinstance(result_spec, dict):
                continue
            result_spec["dtensor_spec"] = None

    return mg, {"solution": {}}
