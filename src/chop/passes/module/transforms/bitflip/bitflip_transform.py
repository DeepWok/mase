try:
    import mase_triton
    import mase_triton.random_bitflip

    MASE_TRITON_AVAILABLE = True
except ImportError:
    MASE_TRITON_AVAILABLE = False

import torch
from ...state_dict_map import match_a_pattern
from ...module_modify_helper import replace_by_name


def get_config_by_name(config: dict, name: str):
    if name in config:
        return config[name]
    else:
        if "default" in config:
            return config["default"]
        else:
            return None


def get_config_by_regex_name(config: dict, name: str):
    matched_pattern = match_a_pattern(name, config.keys())
    if matched_pattern is None:
        if "default" in config:
            return config["default"]
        else:
            return None
    else:
        return config[matched_pattern]


def get_layer_config(
    layer_name_to_config: dict[str, dict], use_regex: bool, layer_name: str
) -> dict | None:
    if use_regex:
        config = get_config_by_regex_name(layer_name_to_config, layer_name)
    else:
        config = get_config_by_name(layer_name_to_config, layer_name)
    return config


if MASE_TRITON_AVAILABLE:
    BITFLIP_CLS_MAP = {
        torch.nn.Linear: mase_triton.random_bitflip.layers.RandomBitFlipLinear,
    }

    def bitflip_module_transform_pass(
        network: torch.nn.Module, pass_args: dict
    ) -> torch.nn.Module:
        """
        Apply bitflip module transform pass to the network.

        :param network: The network to be transformed.
        :type network: torch.nn.Module
        :param pass_args: The arguments for the transformation.
        :type pass_args: dict
        :return: The transformed network.
        :rtype: torch.nn.Module
        :raises AssertionError: If the `by` argument is not in ["name", "regex"].
        """
        target_classes = tuple(BITFLIP_CLS_MAP.keys())
        by = pass_args.pop("by", "regex_name")
        assert by in [
            "name",
            "regex_name",
        ], f"by should be in ['name', 'regex_name'], but got {by}"

        for m_name, m in network.named_modules():
            if not isinstance(m, target_classes):
                continue
            m_config = get_layer_config(
                pass_args, use_regex=by == "regex_name", layer_name=m_name
            )
            if m_config is None:
                continue
            new_m_cls = BITFLIP_CLS_MAP[type(m)]
            new_m = new_m_cls.from_linear(m, **m_config)
            replace_by_name(network, name=m_name, module=new_m)

        return network

else:

    def bitflip_module_transform_pass(
        network: torch.nn.Module, pass_args: dict
    ) -> torch.nn.Module:
        raise RuntimeError("mase_triton is not available, please install it first.")
