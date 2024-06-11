
from torch.distributed._tensor import (
    Replicate,
    Shard,
)

from chop.passes.graph.analysis.autosharding.common import SpmdShard

def placement_from_sharding_config(sharding_config):
    """
    Sharding config is given as a tuple such as (R, S_0) where a symbol S_x at index i indicates
    that tensor dimension i is sharded along the x-th dimension of the device mesh. However,
    the distribute_tensor API expects a tuple of Shard() and Replicate() objects where a Shard(x)
    at index i indicates that tensor dimension x is sharded along device mesh dimension i.
    """
    placement = [Replicate(), Replicate()]
    for shard_type in [SpmdShard.S_0, SpmdShard.S_1]:
        if shard_type in sharding_config:
            idx = sharding_config.index(shard_type)
            placement[shard_type.value] = Shard(idx)
    return placement
        