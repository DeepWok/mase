import sys, os

import torch
import torch.nn as nn

import pytest

from transformers.activations import GELUActivation

import chop.passes as passes
import chop.actions as actions
from chop.ir import MaseGraph
from chop.models.llama import LlamaConfig, LlamaModel
from chop.models.llama.modeling_llama import LlamaAttention
from chop.passes.graph.utils import deepsetattr

# from chop.nn.quantized import LlamaAttentionInteger
from chop.tools import get_logger, set_excepthook

from mase_components import get_module_dependencies
from mase_components.helper.generate_memory import generate_sv_lut

import operator
from functools import partial

logger = get_logger(__name__)
logger.setLevel("DEBUG")
set_excepthook()

# * Define custom ops (leaf submodules during tracing)
# * This is useful so we can write a single optimised verilog file for self attention,
# * instead of relying on emit_verilog to instantiate each submodule
LLAMA_CUSTOM_OPS = {
    "modules": {},
    "functions": {},
}


def llama_module_level_quantize(model, model_config, q_config):
    return model


def llama_update_metadata(mg, q_config):
    """
    The following processing is a temporary hot fix to get emit verilog working on the llama model. We
    update the type and precision for the add, getitem and split (fork) nodes which are currently
    inserted in the patched model code. In the (near) future, inserting forking nodes and setting their
    precision correctly will be handled automatedly as a preprocessing step for the emit verilog pass,
    so this function will be unnecessary.
    """
    return mg, {}


def emit_verilog_llama(
    config,
    q_config,
    config_sequence_length,
    wait_count=15,
    wait_unit="ms",
    max_parallelism=4,
):
    # * Get model and quantize self attention, linear and layer norm layers
    model = LlamaModel(config)
    model = llama_module_level_quantize(model, config, q_config)
    logger.info(f"Quantized Llama model: {model}")

    # * Trace the model
    mg = MaseGraph(model, custom_ops=LLAMA_CUSTOM_OPS)
    mg, _ = passes.init_metadata_analysis_pass(mg)

    mg, _ = passes.report_graph_analysis_pass(mg, pass_args={"file_name": "llama.txt"})

    # * Add metadata analysis passes
    mg, _ = passes.add_common_metadata_analysis_pass(
        mg,
        pass_args={
            "dummy_in": {
                "input_ids": torch.randn(
                    (1, config_sequence_length, config.hidden_size)
                )
            },
            "add_value": False,
        },
    )

    mg, _ = llama_update_metadata(mg, q_config)

    mg, _ = passes.add_hardware_metadata_analysis_pass(
        mg,
        pass_args={
            "max_parallelism": [max_parallelism] * 4,
        },
    )

    # * Save the metadata to a file for debugging
    mg, _ = passes.report_node_meta_param_analysis_pass(
        mg,
        pass_args={
            "which": ["common", "hardware"],
            "save_path": "llama_graph_meta_params.txt",
        },
    )

    mg, _ = passes.emit_verilog_top_transform_pass(mg)
    mg, _ = passes.emit_bram_transform_pass(mg)
    mg, _ = passes.emit_internal_rtl_transform_pass(mg)
    mg, _ = passes.emit_cocotb_transform_pass(
        mg,
        pass_args={
            "wait_time": wait_count,
            "wait_unit": wait_unit,
        },
    )
    mg, _ = passes.emit_vivado_project_transform_pass(mg)

    # Temporary: fix data coherency checks
    os.environ["COCOTB_RESOLVE_X"] = "ZEROS"

    actions.simulate(
        skip_build=False, skip_test=False, gui=False, waves=False, simulator="questa"
    )


def get_default_qconfig():
    return {
        "data_in_width": 8,
        "data_in_frac_width": 3,
        "weight_width": 8,
        "weight_frac_width": 3,
        "bias_width": 8,
        "bias_frac_width": 3,
        "data_out_width": 8,
        "data_out_frac_width": 3,
    }


@pytest.mark.skip(reason="Fixing needed")
def test_emit_verilog_llama_smoke():
    config = LlamaConfig()
    # config.num_hidden_layers = 3
    # config.hidden_size = 96
    # config.intermediate_size = 384

    # Make config match 7b model
    config.max_position_embeddings = 4096
    config.rms_norm_eps = 1e-5
    config_sequence_length = 4

    q_config = get_default_qconfig()
    emit_verilog_llama(
        config, q_config, config_sequence_length, wait_count=10, max_parallelism=2
    )


# if __name__ == "__main__":
#     generate_sv_lut("silu", 8, 3, data_width=8, f_width=3, path_with_dtype=False)
#     test_emit_verilog_llama_smoke()
