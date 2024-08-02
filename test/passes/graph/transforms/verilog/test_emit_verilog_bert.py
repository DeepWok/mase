import sys, os, pytest

import torch
import torch.nn as nn

from transformers.activations import GELUActivation

import chop.passes as passes
import chop.actions as actions
from chop.ir import MaseGraph
from chop.models.patched.bert import BertConfig, BertModel
from chop.models.patched.bert.modeling_bert import BertSelfAttention
from chop.passes.graph.utils import deepsetattr
from chop.nn.quantized import (
    BertSelfAttentionInteger,
    LinearInteger,
    LayerNormInteger,
    GELUInteger,
)
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
BERT_CUSTOM_OPS = {
    "modules": {
        BertSelfAttentionInteger: {
            "args": {
                "hidden_states": "data_in",
                "attention_mask": None,
                "head_mask": None,
                "encoder_hidden_states": None,
                "encoder_attention_mask": None,
                "past_key_value": None,
                "output_attentions": "config",
            },
            "toolchain": "INTERNAL_RTL",
            "module": "fixed_self_attention_single_precision_wrapper",
            "dependence_files": get_module_dependencies(
                "attention/fixed_self_attention_single_precision_wrapper"
            ),
        },
    },
    "functions": {},
}


def bert_module_level_quantize(model, model_config, q_config):
    for module in model.named_modules():
        if isinstance(module[1], BertSelfAttention):
            new_module = BertSelfAttentionInteger(
                model_config, q_config, output_tensor_only=True
            )
        elif isinstance(module[1], nn.Linear):
            new_module = LinearInteger(
                in_features=module[1].in_features,
                out_features=module[1].out_features,
                bias=module[1].bias is not None,
                config=q_config,
            )
        elif isinstance(module[1], nn.LayerNorm):
            new_module = LayerNormInteger(
                normalized_shape=module[1].normalized_shape,
                eps=module[1].eps,
                config=q_config,
            )
        elif isinstance(module[1], GELUActivation):
            new_module = GELUInteger(config=q_config)
        else:
            continue
        logger.info(f"Replacing module: {module[0]}")
        deepsetattr(model, module[0], new_module)
    return model


def bert_update_metadata(mg, q_config):
    """
    The following processing is a temporary hot fix to get emit verilog working on the bert model. We
    update the type and precision for the add, getitem and split (fork) nodes which are currently
    inserted in the patched model code. In the (near) future, inserting forking nodes and setting their
    precision correctly will be handled automatedly as a preprocessing step for the emit verilog pass,
    so this function will be unnecessary.
    """
    for node in mg.fx_graph.nodes:

        # Update args
        if (
            node.target == operator.add
            or node.target == operator.getitem
            or node.meta["mase"]["common"]["mase_op"] == "df_split"
        ):
            node.meta["mase"]["common"]["args"]["data_in_0"]["type"] = "fixed"
            node.meta["mase"]["common"]["args"]["data_in_0"]["precision"] = [
                q_config["data_in_width"],
                q_config["data_in_frac_width"],
            ]
            if "data_in_1" in node.meta["mase"]["common"]["args"]:
                node.meta["mase"]["common"]["args"]["data_in_1"]["type"] = "fixed"
                node.meta["mase"]["common"]["args"]["data_in_1"]["precision"] = [
                    q_config["data_in_width"],
                    q_config["data_in_frac_width"],
                ]

        # Update results
        if (
            node.target == operator.add
            or node.target == operator.getitem
            or node.meta["mase"]["common"]["mase_op"] == "df_split"
            or node.op == "placeholder"
            or node.op == "output"
        ):
            node.meta["mase"]["common"]["results"]["data_out_0"]["type"] = "fixed"
            node.meta["mase"]["common"]["results"]["data_out_0"]["precision"] = [
                q_config["data_out_width"],
                q_config["data_out_frac_width"],
            ]
            if "data_out_1" in node.meta["mase"]["common"]["results"]:
                node.meta["mase"]["common"]["results"]["data_out_1"]["type"] = "fixed"
                node.meta["mase"]["common"]["results"]["data_out_1"]["precision"] = [
                    q_config["data_out_width"],
                    q_config["data_out_frac_width"],
                ]

        # Set one of the args to none according to the select value
        if node.target == operator.getitem:
            select = 0 if node.args[1] == 1 else 1
            node.meta["mase"]["common"]["args"][f"data_in_{select}"] = None

    return mg, {}


def emit_verilog_bert(
    config,
    q_config,
    config_sequence_length,
    wait_count=15,
    wait_unit="ms",
    max_parallelism=4,
):
    # * Get model and quantize self attention, linear and layer norm layers
    model = BertModel(config)
    model = bert_module_level_quantize(model, config, q_config)
    logger.info(f"Quantized BERT model: {model}")

    # * Trace the model
    mg = MaseGraph(model, custom_ops=BERT_CUSTOM_OPS)
    mg, _ = passes.init_metadata_analysis_pass(mg)

    mg, _ = passes.report_graph_analysis_pass(mg, pass_args={"file_name": "bert.txt"})

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

    mg, _ = bert_update_metadata(mg, q_config)

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
            "save_path": "graph_meta_params.txt",
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

    # check questa
    if os.system("questa") != 0:
        logger.info(
            "Questa is required for this test, but the system does not have it, so later part this test is skipped."
        )
        return

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


@pytest.mark.large
def test_emit_verilog_bert_smoke():
    config = BertConfig()
    config.num_hidden_layers = 3
    config.hidden_size = 96
    config.intermediate_size = 384
    config_sequence_length = 4
    q_config = get_default_qconfig()
    emit_verilog_bert(
        config, q_config, config_sequence_length, wait_count=10, max_parallelism=2
    )


@pytest.mark.large
def test_emit_verilog_bert_regression():
    config = BertConfig()
    config.num_hidden_layers = 3
    config.hidden_size = 384
    config.intermediate_size = 1536
    config_sequence_length = 128
    q_config = get_default_qconfig()
    emit_verilog_bert(
        config, q_config, config_sequence_length, wait_count=15, max_parallelism=16
    )


if __name__ == "__main__":
    generate_sv_lut(
        "gelu",
        8,
        3,
        data_width=8,
        f_width=3,
        path="./src/mase_components/activation_layers/rtl",
        path_with_dtype=False,
    )
    test_emit_verilog_bert_smoke()
    test_emit_verilog_bert_regression()
