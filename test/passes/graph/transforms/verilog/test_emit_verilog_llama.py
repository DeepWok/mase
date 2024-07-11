import os, operator
import pytest

import torch
import torch.nn as nn

from chop import AutoPipelineForEmitVerilog
import chop.passes as passes
import chop.actions as actions
from chop.ir import MaseGraph
from chop.passes.graph.utils import deepsetattr

from chop.models.patched.llama import LlamaConfig, LlamaModel
from chop.models.patched.llama.modeling_llama import LlamaSdpaAttention, LlamaRMSNorm

from chop.nn.quantized import (
    LlamaSdpaAttentionInteger,
    LinearInteger,
    RMSNormInteger,
    SiLUInteger,
)

from chop.tools import get_logger, set_excepthook

from mase_components import get_module_dependencies
from mase_components.helper.generate_memory import generate_sv_lut

logger = get_logger(__name__)
logger.setLevel("DEBUG")
set_excepthook()

# Temporary: fix data coherency checks
os.environ["COCOTB_RESOLVE_X"] = "ZEROS"

SMOKE_TEST_SCALE_FACTOR = 8

# * Define custom ops (leaf submodules during tracing)
# * This is useful so we can write a single optimised verilog file for self attention,
# * instead of relying on emit_verilog to instantiate each submodule
LLAMA_CUSTOM_OPS = {
    "modules": {
        LlamaSdpaAttention: {
            "args": {
                "hidden_states": "data_in",
                "attention_mask": None,
                "position_ids": None,
                "past_key_value": None,
                "output_attentions": None,
                "use_cache": None,
                "cache_position": None,
            },
            "toolchain": "INTERNAL_RTL",
            "module": "fixed_self_attention_single_precision_wrapper",
            "dependence_files": get_module_dependencies(
                "attention/fixed_self_attention_single_precision_wrapper"
            ),
        },
        RMSNormInteger: {
            "args": {
                "hidden_states": "data_in",
            },
            "toolchain": "INTERNAL_RTL",
            "module": "norm",
            "dependence_files": get_module_dependencies("norm/norm"),
        },
        SiLUInteger: {
            "args": {
                "input": "data_in",
            },
            "toolchain": "INTERNAL_RTL",
            "module": "silu",
            "dependence_files": get_module_dependencies("silu/silu"),
        },
    },
    "functions": {},
}


def llama_module_level_quantize(model, model_config, q_config):
    for name, module in model.named_modules():
        if isinstance(module, LlamaSdpaAttention):
            new_module = LlamaSdpaAttentionInteger(
                config=model_config,
                q_config=q_config,
                output_tensor_only=True,
            )
        elif isinstance(module, nn.Linear):
            new_module = LinearInteger(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                config=q_config,
            )
        elif isinstance(module, LlamaRMSNorm):
            new_module = RMSNormInteger(
                normalized_shape=None,
                eps=module.variance_epsilon,
                config=q_config,
            )
        elif isinstance(module, nn.SiLU):
            new_module = SiLUInteger(
                inplace=module.inplace,
                config=q_config,
            )
        else:
            continue
        logger.info(f"Replacing module: {name}")
        deepsetattr(model, name, new_module)
    return model


def emit_verilog_llama(
    config,
    q_config,
    config_sequence_length,
    config_batch_size,
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

    pipeline = AutoPipelineForEmitVerilog()
    mg = pipeline(
        mg,
        pass_args={
            "report_graph_analysis_pass": {"file_name": "llama.txt"},
            "add_common_metadata_analysis_pass": {
                "dummy_in": {
                    "input_ids": torch.randn(
                        (config_batch_size, config_sequence_length, config.hidden_size)
                    )
                },
                "add_value": False,
            },
            "patch_metadata_transform_pass": {
                "q_config": q_config,
            },
            "add_hardware_metadata_analysis_pass": {
                "max_parallelism": [max_parallelism] * 4,
            },
            "report_node_meta_param_analysis_pass": {
                "which": ["common", "hardware"],
                "save_path": "llama_graph_meta_params.txt",
            },
            "emit_cocotb_transform_pass": {
                "wait_time": wait_count,
                "wait_unit": wait_unit,
            },
        },
    )

    actions.simulate(
        skip_build=False, skip_test=False, gui=True, waves=False, simulator="questa"
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
    config.num_hidden_layers = 1
    config.hidden_size //= SMOKE_TEST_SCALE_FACTOR
    config.intermediate_size //= SMOKE_TEST_SCALE_FACTOR
    config.max_position_embeddings = 4096
    config.rms_norm_eps = 1e-5

    config_batch_size = 5
    config_sequence_length = 4

    q_config = get_default_qconfig()

    emit_verilog_llama(
        config,
        q_config,
        config_sequence_length,
        config_batch_size,
        wait_count=10,
        max_parallelism=2,
    )


if __name__ == "__main__":
    generate_sv_lut(
        "silu",
        8,
        3,
        data_width=8,
        f_width=3,
        path="./src/mase_components/activation_layers/rtl",
        path_with_dtype=False,
    )
    test_emit_verilog_llama_smoke()
