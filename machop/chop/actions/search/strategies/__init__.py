from .random import StrategyRandom
from .rl import StrategyRL


strategy_map = {
    "random": StrategyRandom,
    "rl": StrategyRL,
}
