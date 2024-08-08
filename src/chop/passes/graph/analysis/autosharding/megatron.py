from chop.ir import MaseGraph
from .mesh_model import MeshModel


def megatron_autosharding_pass(
    mg: MaseGraph,
    mesh: MeshModel,
    pass_args: dict,
):
    raise NotImplementedError("Megatron autosharding pass is not implemented yet.")
