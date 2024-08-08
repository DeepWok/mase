from torch.distributed._tensor._op_schema import DTensorSpec
from torch.distributed._tensor.placement_types import Replicate

from chop.ir import MaseGraph
from .mesh_model import MeshModel


def fully_replicated_autosharding_pass(
    mg: MaseGraph,
    mesh: MeshModel,
    pass_args: dict,
):
    spec = DTensorSpec(
        None,
        (Replicate(), Replicate()),
        None,
    )

    for node in mg.nodes:
        meta = node.meta["mase"]

        for arg, arg_info in meta["common"]["args"].items():
            arg_info["dtensor_spec"] = spec

        for result, result_info in meta["common"]["results"].items():
            result_info["dtensor_spec"] = spec

    return mg, {"solution": {}}
