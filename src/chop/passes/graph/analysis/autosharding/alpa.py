from chop.tools import get_logger

from .alpa_intra_operator import alpa_intra_op_sharding_pass

logger = get_logger(__name__)
logger.setLevel("DEBUG")


def alpa_autosharding_pass(mg, mesh, pass_args={}):
    """A lightweight implementation of the core algorithm from the Alpa paper: https://arxiv.org/abs/2201.12023

    Args:
        mg (MaseGraph): Input MaseGraph.
        mesh (MeshModel): Input MeshModel.
        pass_args (dict, optional): pass arguments. Defaults to {}.

    Returns:
        MaseGraph: MaseGraph with sharding strategy annotated for each operator.
    """
    mg, pass_outs = alpa_intra_op_sharding_pass(mg, mesh, pass_args=pass_args)
    return mg, pass_outs
