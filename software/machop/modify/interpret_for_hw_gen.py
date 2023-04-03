import os
from logging import getLogger

import toml

from ..graph.mase_interpreter import MaseInterpreter
from ..graph.passes.set_metadata_common import (
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
    modified_graph_model, model_name: str, task: str, data_module, save_dir: str
):
    gm = modified_graph_model

    mase_interpreter = MaseInterpreter(
        gm,
        hook_before_call_function=set_metadata_common_before_call_function,
        hook_after_call_function=set_metadata_common_after_call_function,
        hook_before_call_method=set_metadata_common_before_call_method,
        hook_after_call_method=set_metadata_common_after_call_method,
        hook_before_call_module=set_metadata_common_before_call_module,
        hook_after_call_module=set_metadata_common_after_call_module,
    )

    input_args = get_input_args(model_name, task, data_module)
    mase_interpreter.forward_and_interpret(*input_args)

    common_dict_to_save = {}
    save_path = os.path.join(save_dir, "common_meta.toml")
    for n in gm.graph.nodes:
        common_dict_to_save[n.name] = n.meta["common"]
    # breakpoint()
    with open(save_path, "w+") as f:
        toml.dump(common_dict_to_save, f)
    logger.info(f"hw-gen.toml is saved at {save_path}")
    return gm
