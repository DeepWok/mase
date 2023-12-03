import importlib
import os

import regex as re
from chop.passes.graph.common import MASE_IMPLICIT_FUNCS

# from ..session.plt_wrapper.nlp.classification import NLPClassificationModelWrapper
# from ..session.plt_wrapper.nlp.lm import NLPLanguageModelingModelWrapper
# from ..session.plt_wrapper.nlp.translation import NLPTranslationModelWrapper
# from ..session.plt_wrapper.vision import VisionModelWrapper


def _import_config_from_py_file(model_name: str, file_path: str):
    """
    load a config dict from .py file where a ignore_modules included nn.Module classes to be ignored in profiling
    """
    # default config
    config = {
        "print_profile": True,
        "detailed": True,
        "module_depth": -1,
        "top_modules": 1,
        "warm_up": 10,
        "as_string": True,
        "output_file": "estimate_sw_reports/{}.txt".format(
            model_name.replace("/", "-")
        ),
        "ignore_modules": [],
    }
    # import the custom config from .py file
    if file_path is not None:
        assert os.path.isfile(file_path) and file_path.endswith(
            ".py"
        ), "The config file should be an existing .py file"
        spec = importlib.util.spec_from_file_location("config_py", file_path)
        config_py = spec.loader.load_module()
        imported_config = config_py.config
        config.update(imported_config)

    return config


def _get_next_call_node(node, nodes_in):
    for next_node, x in node.users.items():
        name = (
            next_node.name[0 : next_node.name.find("_")]
            if "_" in next_node.name
            else next_node.name
        )
        # No need to synthsize into hardware
        if (
            name in MASE_IMPLICIT_FUNCS
            or next_node.meta["mase"].parameters["common"]["mase_type"]
            == "implicit_func"
        ):
            nodes_in = _get_next_call_node(next_node, nodes_in)
            next_node.meta["mase"].parameters["hardware"]["is_implicit"] = True
        elif next_node not in nodes_in:
            nodes_in.append(next_node)
    return nodes_in


def _get_prev_call_node(node, nodes_out):
    for prev_node in node.all_input_nodes:
        # No need to synthsize into hardware
        name = (
            prev_node.name[0 : prev_node.name.find("_")]
            if "_" in prev_node.name
            else prev_node.name
        )
        implicit = (
            prev_node.meta["mase"].parameters["common"]["mase_type"] == "implicit_func"
            or prev_node.meta["mase"].parameters["common"]["mase_type"] == "placeholder"
        )
        if implicit:
            nodes_out = _get_prev_call_node(prev_node, nodes_out)
            prev_node.meta["mase"].parameters["hardware"]["is_implicit"] = True
        elif prev_node not in nodes_out:
            nodes_out.append(prev_node)
    return nodes_out


def get_input_nodes(fx_graph):
    nodes_in = []
    for node in fx_graph.nodes:
        if node.op == "placeholder":
            nodes_in = _get_next_call_node(node, nodes_in)
            node.meta["mase"].parameters["hardware"]["is_implicit"] = True
    return nodes_in


def get_output_nodes(fx_graph):
    nodes_out = []
    for node in fx_graph.nodes:
        if node.op == "output":
            nodes_out = _get_prev_call_node(node, nodes_out)
            node.meta["mase"].parameters["hardware"]["is_implicit"] = True
    return nodes_out


def pattern_name_match(pattern, name):
    return bool(re.fullmatch(pattern, name))


# names are likely to be func_name_[0-9]+
def match_and_filter(name, funcs):
    for pattern in funcs:
        if (pattern == name) or (pattern + "_" in name):
            return True, pattern
    return False, None


def is_tensor_constant(s):
    # Define the regular expression pattern to match "_tensor_constant" followed by one or more digits
    pattern = r"_tensor_constant\d+"

    # Use re.match to check if the string matches the pattern at the beginning
    match = re.match(pattern, s)

    # If there's a match, return True; otherwise, return False
    return bool(match)


def is_seq_blocks_parameter(s):
    # Define the regular expression pattern to match "seq_blocks_" followed by a digit, an underscore, and a parameter name
    # TODO: need to make this more general
    pattern = r"(block|seq_blocks|linear|conv2d)(_\d+)?(_\d+)?_(weight|bias|gamma|means|pruning_masks)"
    # Use re.match to check if the string matches the pattern
    match = re.match(pattern, s)

    # If there's a match, return True; otherwise, return False
    return bool(match)
