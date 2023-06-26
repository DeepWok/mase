from typing import Dict, Union

import toml

from ...modify.quantizers.quantizers import integer_fraction


def profile_to_int_frac_config(
    profile: Union[Dict, str],
    stat_name="reduced_soft_range",
    width: int = 8,
    frac_choices=(-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    save_path: str = None,
):
    if isinstance(profile, str):
        profile = toml.load(profile)

    q_config = {
        "default": {
            "data_in_width": 8,
            "data_in_frac_width": 8,
            "weight_width": 8,
            "weight_frac_width": 8,
            "bias_width": 8,
            "bias_frac_width": 8,
        },
        "module_nodes_to_modify": {},
        "function_nodes_to_modify": {},
        "method_nodes_to_modify": {},
    }
    for name, stats in profile.items():
        node_name = name.split("::")[0]
        item_name = name.split("::")[1]
        if item_name == "data_in_0":
            item_name = "data_in"
        if item_name == "data_in_1":
            item_name = "weight"
        if not item_name in ["data_in", "weight", "bias"]:
            raise RecursionError(f"Unknown item name: {item_name}")

        op = stats["op"]

        min_value = stats[stat_name]["min"]
        max_value = stats[stat_name]["max"]
        frac_width = integer_fraction(
            width=width,
            min_value=min_value,
            max_value=max_value,
            frac_choices=frac_choices,
        )
        assert op in ["call_module", "call_function", "call_method"]
        if op == "call_module":
            tgt_entry = "module_nodes_to_modify"
        elif op == "call_function":
            tgt_entry = "function_nodes_to_modify"
        else:
            tgt_entry = "method_nodes_to_modify"
        if node_name not in q_config[tgt_entry]:
            q_config[tgt_entry][node_name] = {"name": "integer"}
        q_config[tgt_entry][node_name] |= {
            f"{item_name}_width": width,
            f"{item_name}_frac_width": frac_width,
        }

    if save_path is not None:
        with open(save_path, "w") as f:
            toml.dump(q_config, f)
    return q_config
