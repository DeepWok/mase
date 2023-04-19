import os
from copy import deepcopy
from logging import getLogger

import toml
import torch
from torch.fx import GraphModule

from ..mase_interpreter import MaseInterpreter
from ..mase_tracer import mase_symbolic_trace
from .set_metadata_common import (
    set_and_check_metadata_common_without_forward,
    set_metadata_common_after_call_function,
    set_metadata_common_after_call_method,
    set_metadata_common_after_call_module,
    set_metadata_common_before_call_function,
    set_metadata_common_before_call_method,
    set_metadata_common_before_call_module,
)
from .utils import get_input_args

logger = getLogger(__name__)

# TODO: There is a serious issue that the node metadata structure of modifier
# is not different from the masegraph node. So here temporarily use a model-level
# pass to emit toml, which will be loaded by a masegraph

# def update_common_metadata_pass(
#     mase_graph,
#     input_args,
#     save_path,
# ):
#     # TODO: make a dummy gm for interpret - it cannot directly take a model
#     with torch.no_grad():
#         mase_interpreter = MaseInterpreter(
#             mase_graph.quantized_model,
#             hook_before_call_function=set_metadata_common_before_call_function,
#             hook_after_call_function=set_metadata_common_after_call_function,
#             hook_before_call_method=set_metadata_common_before_call_method,
#             hook_after_call_method=set_metadata_common_after_call_method,
#             hook_before_call_module=set_metadata_common_before_call_module,
#             hook_after_call_module=set_metadata_common_after_call_module,
#             hooks_after_forward=(set_and_check_metadata_common_without_forward,),
#         )
#
#         mase_interpreter.interpret_with_forward(*input_args)
#     # for n in gm.graph.nodes:
#     #     print(n.name, n.meta)
#     common_dict_to_save = {}
#     for n in mase_graph.fx_graph.nodes:
#         if n.op in ("call_function", "call_module", "call_method"):
#             common_dict_to_save[n.name] = deepcopy(n.meta.parameters["common"])
#             for arg_i, type_precision_i in n.meta.parameters["common"]["args"].items():
#                 if type_precision_i["type"] == "NA":
#                     common_dict_to_save[n.name]["args"].pop(arg_i)
#     with open(save_path, "w+") as f:
#         toml.dump(common_dict_to_save, f)
#     logger.info(f"common metadata toml is saved at {save_path}")
#     return mase_graph


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
            hooks_after_forward=(set_and_check_metadata_common_without_forward,),
        )

        input_args = get_input_args(model_name, task, data_module)
        mase_interpreter.interpret_with_forward(*input_args)
    # for n in gm.graph.nodes:
    #     print(n.name, n.meta)
    common_dict_to_save = {}
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "common_meta.toml")

    for n in gm.graph.nodes:
        if n.op in ("call_function", "call_module", "call_method"):
            # if not "assert" in str(n.target):
            #     common_dict_to_save[n.name] = deepcopy(n.meta["common"])
            common_dict_to_save[n.name] = deepcopy(n.meta["common"])
            for arg_i, type_precision_i in n.meta["common"]["args"].items():
                if "bias" in arg_i and type_precision_i["type"] == "NA":
                    common_dict_to_save[n.name]["args"].pop(arg_i)
    with open(save_path, "w+") as f:
        toml.dump(common_dict_to_save, f)
    logger.info(f"common metadata toml is saved at {save_path}")
    return gm
