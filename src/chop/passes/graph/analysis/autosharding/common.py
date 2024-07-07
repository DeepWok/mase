from enum import Enum


class SpmdShard(Enum):
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
    (SpmdShard.R, SpmdShard.R),
    (SpmdShard.R, SpmdShard.S_0),
    (SpmdShard.R, SpmdShard.S_1),
    (SpmdShard.S_0, SpmdShard.R),
    (SpmdShard.S_0, SpmdShard.S_1),
    (SpmdShard.S_1, SpmdShard.R),
    (SpmdShard.S_1, SpmdShard.S_0),
]
