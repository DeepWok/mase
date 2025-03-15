from dataclasses import dataclass, field
from typing import ClassVar, Type, Optional, Union, Any
from .simulator_tile import TorchSimulatorTile

class RPUConfig:
    ## TODO
    pass

@dataclass
class TorchInferenceRPUConfig(RPUConfig):
    """TorchInference configuration.

    This configuration defaults to a tile module implementation that
    supported a subset of functions of the ``InferenceRPUConfig`` but
    uses native torch instead of the RPUCuda library for simulating
    the analog MVM.

    The advantage is that autograd is more fully supported and
    hardware aware training is more flexible to be modified. However,
    some nonidealities are not supported.

    Note:

        For features that are not supported a ``NotImplementedError`` or a
        ``TorchTileConfigError`` is raised.
    """

    tile_class: Type = TorchSimulatorTile
    """Tile class that corresponds to this RPUConfig."""

    tile_array_class: Type = TileModuleArray
    """Tile class used for mapped logical tile arrays."""