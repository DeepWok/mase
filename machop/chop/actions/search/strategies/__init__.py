from .rl import StrategyRL
from .optuna import StrategyOptunaAlgorithm


strategy_map = {
    "rl": StrategyRL,
    "optuna": StrategyOptunaAlgorithm,
}
