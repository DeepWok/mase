from chop.passes.graph.utils import vf
from chop.passes.graph.analysis.add_metadata.hardware_metadata_layers import (
    INTERNAL_COMP,
)


def add_node_name_to_keys_with_precision(node_name: str, key: str) -> str:
    """
    Parses parameter keys with the format of: "[<n>:<n>] <param> [<n>:<n>]",
    where <n> can be any value. The returned key is in the format of:
    "[<n>:<n>] <node_name>_<param> [<n>:<n>]".
    """
    prec_sta = key.find("[")
    prec_end = key.find("]")
    prec = key[prec_sta : prec_end + 1]  # precision notation

    akey_sta = prec_end + 2
    akey_end = key.find("[", prec_end) - 2
    akey = key[akey_sta : akey_end + 1]  # actual key

    arry_sta = akey_end + 2
    arry = key[arry_sta:]  # array notation

    return f"{prec} {node_name}_{akey} {arry}"


def get_top_param_name_with_precision(key: str) -> str:
    """
    Parses parameter keys of the format "[\<n\>:\<n\>] \<param\> [\<n\>:\<n\>]", where
    \<n\> can be any value. The returned key is this case would be just \<param\>.
    """
    prec_end = key.find("]")
    akey_sta = prec_end + 2
    akey_end = key.find("[", prec_end) - 2
    return key[akey_sta : akey_end + 1]


def get_node_param_name_with_precision(node_name: str, key: str) -> str:
    """
    Parses parameter keys of the format "[<n>:<n>] <node>_<param> [<n>:<n>]",
    where <n> can be any value. The returned key is this case would be just <param>.
    """
    tmp = get_top_param_name_with_precision(key)
    tmp = tmp.replace(f"{node_name}_", "")
    return tmp


def get_verilog_parameters(graph):
    parameter_map = {}

    for node in graph.fx_graph.nodes:
        if node.meta["mase"].parameters["hardware"]["is_implicit"]:
            continue
        node_name = vf(node.name)

        for key, value in (
            node.meta["mase"].parameters["hardware"]["verilog_param"].items()
        ):
            if value is None:
                continue
            if isinstance(value, list):
                new_value = "'{"
                for num in value:
                    new_value += str(num) + ","
                new_value = new_value[0:-1]  # pop last comma for verilog
                new_value += "}"
                value = new_value
            elif not isinstance(value, (int, float, complex, bool)):
                value = '"' + value + '"'
            assert (
                f"{node_name}_{key}" not in parameter_map.keys()
            ), f"{node_name}_{key} already exists in the parameter map"

            if "[" in key:
                parameter_map[add_node_name_to_keys_with_precision(node_name, key)] = (
                    value
                )
            else:
                parameter_map[f"{node_name}_{key}"] = value

    # * Return graph level parameters
    for node in graph.nodes_in + graph.nodes_out:
        for key, value in (
            node.meta["mase"].parameters["hardware"]["verilog_param"].items()
        ):
            if "DATA_IN" in key or "DATA_OUT" in key:
                parameter_map[key] = value

    return parameter_map


def include_ip_to_project(node):
    """
    Copy internal files to the project
    """
    mase_op = node.meta["mase"].parameters["common"]["mase_op"]
    return node.meta["mase"].parameters["hardware"]["dependence_files"]
