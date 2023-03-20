from typing import Dict, Tuple


def extract_required_config(self, config: Dict):
    r_config = {}

    for key in self._required_config_keys:
        try:
            r_config[key] = config[key]
        except KeyError:
            raise KeyError(f"key '{key}' is required for {type(self)}'s config")

    return r_config
