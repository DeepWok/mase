import importlib
import os

import regex as re
from chop.passes.common import MASE_IMPLICIT_FUNCS
from chop.plt_wrapper import (
    NLPClassificationModelWrapper,
    NLPLanguageModelingModelWrapper,
    NLPTranslationModelWrapper,
    VisionModelWrapper,
    get_model_wrapper,
)

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


def get_input_args(model_name, model, task, data_loader, info):
    wrapper_cls = get_model_wrapper(model_name, task)
    plt_model = wrapper_cls(model, info=info, learning_rate=1e-4)

    input_args = []
    if isinstance(plt_model, VisionModelWrapper):
        data_loader.prepare_data()
        data_loader.setup()
        batch_x, _ = next(iter(data_loader.train_dataloader()))
        input_args = [batch_x[[0], ...]]
    elif isinstance(
        plt_model,
        (
            NLPClassificationModelWrapper,
            NLPLanguageModelingModelWrapper,
        ),
    ):
        batch = next(iter(data_loader.train_dataloader()))
        input_args = [
            batch["input_ids"][[0], ...],
            batch["attention_mask"][[0], ...],
            None,
        ]
    elif isinstance(plt_model, NLPTranslationModelWrapper):
        batch = next(iter(data_loader.train_dataloader()))
        input_args = [
            batch["input_ids"][[0], ...],
            batch["attention_mask"][[0], ...],
            batch["decoder_input_ids"][[0], ...],
            batch["decoder_attention_mask"][[0], ...],
        ]
    else:
        raise RuntimeError("Unsupported model class")
    return plt_model, input_args


class InputArgsGenerator:
    def __init__(
        self, model_name, task, data_module, which_dataloader="train_dataloader"
    ) -> None:
        self.wrapper_cls = get_model_wrapper(model_name, task)
        data_module.prepare_data()
        data_module.setup()
        assert which_dataloader in [
            "train_dataloader",
            "val_dataloader",
            "test_dataloader",
        ]
        self.dataloader_iter = iter(getattr(data_module, which_dataloader)())

    def __iter__(self):
        return self

    def __next__(self):
        input_args = []
        if self.wrapper_cls is VisionModelWrapper:
            batch_x, _ = next(self.dataloader_iter)
            input_args = [batch_x]
        elif self.wrapper_cls is NLPClassificationModelWrapper:
            batch = next(self.dataloader_iter)
            input_args = [
                batch["input_ids"],
                batch["attention_mask"],
            ]
        elif self.wrapper_cls is NLPLanguageModelingModelWrapper:
            batch = next(self.dataloader_iter)
            input_args = [
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
            ]
        elif self.wrapper_cls is NLPTranslationModelWrapper:
            batch = next(self.dataloader_iter)
            input_args = [
                batch["input_ids"],
                batch["attention_mask"],
                batch["decoder_input_ids"],
                batch["decoder_attention_mask"],
            ]
        else:
            raise RuntimeError("Unsupported model class")
        return input_args


def _get_next_call_node(node, nodes_in):
    for next_node, x in node.users.items():
        # No need to synthsize into hardware
        if next_node.target in MASE_IMPLICIT_FUNCS:
            nodes_in = _get_next_call_node(next_node, nodes_in)
            next_node.meta["mase"].parameters["hardware"]["is_implicit"] = True
        elif next_node not in nodes_in:
            nodes_in.append(next_node)
    return nodes_in


def _get_prev_call_node(node, nodes_out):
    for prev_node in node.all_input_nodes:
        # No need to synthsize into hardware
        if prev_node.target in MASE_IMPLICIT_FUNCS:
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
