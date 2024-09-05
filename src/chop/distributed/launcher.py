import torch.multiprocessing as mp

from chop.distributed.utils import _get_mesh_from_world_size
from ..tools import get_logger

logger = get_logger(__name__)
logger.setLevel("DEBUG")


class MaseLauncher:
    """
    MaseLauncher launches an optimized model on multiple GPUs using torch.distributed.
    """

    def __init__(
        self,
        mg=None,
        world_size=None,
        device_mesh=None,
        device_fn=None,
    ):
        """Initialize the MaseLauncher.

        Args:
            mase_graph (MaseGraph): The MaseGraph object containing the model.
            world_size (int, optional): Number of GPUs to use. Defaults to None.
            device_mesh (list, optional): List of GPUs to use. Defaults to None.
        """
        self.mg = mg
        self.world_size = world_size
        self.device_fn = device_fn

        if device_mesh is None:
            self.device_mesh, _ = _get_mesh_from_world_size(world_size)

    def run(
        self,
        model_class=None,
        model_config=None,
        cli_args=None,
    ):
        logger.info(f"Launching model with world size {self.world_size}.")

        mp.spawn(
            self.device_fn,
            args=(
                self.world_size,
                self.device_mesh,
                model_class,
                model_config,
                cli_args,
            ),
            nprocs=self.world_size,
            join=True,
        )
