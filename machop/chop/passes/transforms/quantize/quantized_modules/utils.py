from typing import Dict, Tuple
from torch import Tensor


def extract_required_config(self, config: Dict):
    r_config = {}

    for key in self._required_config_keys:
        try:
            r_config[key] = config[key]
        except KeyError:
            raise KeyError(f"key '{key}' is required for {type(self)}'s config")

    return r_config


def get_stats(config: dict, stat_name: str) -> float | None:
    if not config.get(stat_name) in [
        None,
        "NA",
    ]:  # if entry does not exists, None is returned, "NA" if no stats available in config file
        return config[stat_name]
    else:
        if "weight" in stat_name:
            stat = config["node_meta_stat"]["weight"]["stat"]
        elif "bias" in stat_name:
            stat = config["node_meta_stat"]["bias"]["stat"]
        elif "data_in" in stat_name:
            stat = config["node_meta_stat"]["data_in_0"]["stat"]
            # TODO FIX MULTI ARG

        if "mean" in stat_name:
            return stat["abs_mean"]["abs_mean"] if "abs_mean" in stat else None
        elif "median" in stat_name:
            return stat["range_quantile"]["max"] if "range_quantile" in stat else None
        elif "max" in stat_name:
            return stat["range_min_max"]["max"] if "range_min_max" in stat else None


def quantiser_passthrough(x: Tensor):
    return x
