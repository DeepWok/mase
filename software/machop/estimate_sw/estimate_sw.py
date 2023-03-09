import importlib
import logging
import os

import torch
from deepspeed.profiling.flops_profiler import get_model_profile

from ..session.plt_wrapper import get_model_wrapper
from ..session.plt_wrapper.nlp.classification import NLPClassificationModelWrapper
from ..session.plt_wrapper.nlp.lm import NLPLanguageModelingModelWrapper
from ..session.plt_wrapper.nlp.translation import NLPTranslationModelWrapper
from ..session.plt_wrapper.vision import VisionModelWrapper


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
        "output_file": f"estimate-sw_report_{model_name}.txt",
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


def estimate_sw_single_gpu(
    model_name: int,
    info: dict,
    model: torch.nn.Module,
    task: str,
    data_loader,
    config_path: str = None,
):
    """
    estimate the FLOPs and latency on torch.cuda.device(0)

    !: currently `ignore_list` only support nn.Module. nn.functional or Tensor function like torch.matmul will still be profiled.
    !: currently only single gpu profiling is supported. DeepSpeed's flops profiler docs on multiple gpus remains to be updated.

    This function uses DeepSpeed flops profiler

    ---
    config_path (str, None): path to .py file where a config dict for estimate-sw is defined.

    ---
    DeepSpeed flops profiler: https://deepspeed.readthedocs.io/en/latest/flops-profiler.html#deepspeed.profiling.flops_profiler.profiler.get_model_profile

    """

    config = _import_config_from_py_file(model_name, config_path)

    assert isinstance(
        config["ignore_modules"], (list, tuple, set)
    ), "ignore_modules should be a list/tuple/set of string nn.Module names"
    logging.debug(f"estimate-sw config: {config}")

    with torch.cuda.device(0):
        wrapper_cls = get_model_wrapper(model_name, task)
        plt_model = wrapper_cls(model, info=info, learning_rate=1e-4)

        input_args = []
        if isinstance(plt_model, VisionModelWrapper):
            batch_x, _ = next(iter(data_loader.train_dataloader))
            input_args = [batch_x[[0], ...]]
        elif isinstance(
            plt_model,
            (
                NLPClassificationModelWrapper,
                NLPLanguageModelingModelWrapper,
            ),
        ):
            batch = next(iter(data_loader.train_dataloader))
            # breakpoint()
            input_args = [
                batch["input_ids"][[0], ...],
                batch["attention_mask"][[0], ...],
                None,
            ]
        elif isinstance(plt_model, NLPTranslationModelWrapper):
            batch = next(iter(data_loader.train_dataloader))
            input_args = [
                batch["input_ids"][[0], ...],
                batch["attention_mask"][[0], ...],
                batch["decoder_input_ids"][[0], ...],
                batch["decoder_attention_mask"][[0], ...],
            ]
        else:
            raise RuntimeError("Unsupported model class")
        flops, macs, params = get_model_profile(plt_model, args=input_args, **config)

        profiled_result = dict(flops=flops, macs=macs, params=params)
        logging.info(profiled_result)
        return profiled_result
