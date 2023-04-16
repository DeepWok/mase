import os
from logging import getLogger

import toml
import torch
from torch.fx import GraphModule

from .mase_interpreter import MaseInterpreter
from .mase_tracer import mase_symbolic_trace
from .passes.optimize import fuse_conv_bn, remove_assert, remove_dropout
from .passes.set_metadata_common import (
    set_and_check_metadata_common_without_forward,
    set_metadata_common_after_call_function,
    set_metadata_common_after_call_method,
    set_metadata_common_after_call_module,
    set_metadata_common_before_call_function,
    set_metadata_common_before_call_method,
    set_metadata_common_before_call_module,
)
from .passes.utils import get_input_args

logger = getLogger(__name__)


def optimize_sw_model_for_synthesis(
    model,
    dummy_inputs,
):
    model.eval()
    graph_module = mase_symbolic_trace(model, concrete_args=dummy_inputs)

    with torch.no_grad():
        mase_interpreter = MaseInterpreter(
            graph_module=graph_module,
            hooks_before_forward=(remove_dropout, fuse_conv_bn, remove_assert),
        )
        graph_module = mase_interpreter.interpret_before_forward()
    return graph_module


def create_and_save_common_metadata(
    modified_graph_model: GraphModule,
    model_name: str,
    task: str,
    data_module,
    save_dir: str,
):
    gm = modified_graph_model
    gm.eval()
    with torch.no_grad():
        mase_interpreter = MaseInterpreter(
            gm,
            # hooks_before_forward=(remove_assert, remove_dropout, fuse_conv_bn),
            # hooks_before_forward=(remove_assert,),
            hook_before_call_function=set_metadata_common_before_call_function,
            hook_after_call_function=set_metadata_common_after_call_function,
            hook_before_call_method=set_metadata_common_before_call_method,
            hook_after_call_method=set_metadata_common_after_call_method,
            hook_before_call_module=set_metadata_common_before_call_module,
            hook_after_call_module=set_metadata_common_after_call_module,
            hooks_after_forward=(set_and_check_metadata_common_without_forward,),
        )

        input_args = get_input_args(model_name, task, data_module)
        mase_interpreter.interpret_with_forward(*input_args)
    # breakpoint()
    # for n in gm.graph.nodes:
    #     print(n.name, n.meta)
    common_dict_to_save = {}
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "common_meta.toml")
    # breakpoint()
    for n in gm.graph.nodes:
        if n.op in ("call_function", "call_module", "call_method"):
            common_dict_to_save[n.name] = n.meta["common"]
    # breakpoint()
    with open(save_path, "w+") as f:
        toml.dump(common_dict_to_save, f)
    logger.info(f"common metadata toml is saved at {save_path}")
    return gm
