import os
from logging import getLogger

import toml
import torch
from torch.fx import GraphModule

from ..graph.mase_interpreter import MaseInterpreter
from ..graph.passes.set_metadata_common import (
    set_and_check_metadata_common_without_forward,
    set_metadata_common_after_call_function,
    set_metadata_common_after_call_method,
    set_metadata_common_after_call_module,
    set_metadata_common_before_call_function,
    set_metadata_common_before_call_method,
    set_metadata_common_before_call_module,
)
from ..graph.passes.utils import get_input_args

logger = getLogger(__name__)


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
            hook_before_call_function=set_metadata_common_before_call_function,
            hook_after_call_function=set_metadata_common_after_call_function,
            hook_before_call_method=set_metadata_common_before_call_method,
            hook_after_call_method=set_metadata_common_after_call_method,
            hook_before_call_module=set_metadata_common_before_call_module,
            hook_after_call_module=set_metadata_common_after_call_module,
            hook_after_forward=set_and_check_metadata_common_without_forward,
        )

        input_args = get_input_args(model_name, task, data_module)
        mase_interpreter.forward_to_interpret(*input_args)
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
