import importlib
import logging
import os

import torch
from deepspeed.profiling.flops_profiler import get_model_profile

from ..utils import get_input_args

# from ..session.plt_wrapper import get_model_wrapper
# from ..session.plt_wrapper.nlp.classification import NLPClassificationModelWrapper
# from ..session.plt_wrapper.nlp.lm import NLPLanguageModelingModelWrapper
# from ..session.plt_wrapper.nlp.translation import NLPTranslationModelWrapper
# from ..session.plt_wrapper.vision import VisionModelWrapper

logger = logging.getLogger(__name__)


def estimate_sw_deepspeed(
    model_name: int,
    task: str,
    info: dict,
    model: torch.nn.Module,
    data_module,
    config: dict,
    save_dir: str,
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

    # config = _import_config_from_py_file(model_name, config_path)

    assert isinstance(
        config["ignore_modules"], (list, tuple, set)
    ), "ignore_modules should be a list/tuple/set of string nn.Module names"
    logger.debug(f"Estimate-sw config: {config}")

    config["output_file"] = os.path.join(
        save_dir, config.get("output_file", "estimate_deepspeed.toml")
    )

    with torch.cuda.device(0):
        plt_model, input_args = get_input_args(
            model_name, model, task, data_module, info
        )
        flops, macs, params = get_model_profile(plt_model, args=input_args, **config)

        profiled_result = dict(flops=flops, macs=macs, params=params)
        logger.info("Estimate-sw result")
        print(profiled_result)
        logger.info("Estimate-sw report saved to {}".format(config["output_file"]))
        return profiled_result
