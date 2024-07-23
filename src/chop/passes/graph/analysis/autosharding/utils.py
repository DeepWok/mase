from typing import cast, Iterable, List, Sequence, Tuple, Union
from torch.distributed._tensor.placement_types import DTensorSpec, Shard


def is_tensor_shardable(shape: Sequence[int], spec: DTensorSpec) -> bool:
    """Check if the shape is shardable according to the spec."""
    # number of shards in each tensor dimension
    shards_map = [1] * len(shape)
    for i, placement in enumerate(spec.placements):
        if placement.is_shard():
            shard_dim = cast(Shard, placement).dim
            shards_map[shard_dim] *= spec.mesh[i]

    for i, dim_size in enumerate(shape):
        # TODO: maybe we should determine is_shardable based on
        #       whether it's evenly sharded or not
        if shards_map[i] > 1 and dim_size < shards_map[i]:
            return False

    return True
