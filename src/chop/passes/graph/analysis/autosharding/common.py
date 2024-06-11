from enum import Enum

class Shard(Enum):
    S_0 = 0
    S_1 = 1
    R = 3

    def __repr__(self):
        return self.name

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented


VALID_2D_TENSOR_SHARDINGS = [
    (Shard.R, Shard.R),
    (Shard.R, Shard.S_0),
    (Shard.R, Shard.S_1),
    (Shard.S_0, Shard.R),
    (Shard.S_0, Shard.S_1),
    (Shard.S_1, Shard.R),
    (Shard.S_1, Shard.S_0),
]