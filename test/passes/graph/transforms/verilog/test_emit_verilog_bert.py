import sys

import torch
import torch.nn as nn
from torch import Tensor
from torch.fx import GraphModule

import chop.passes as passes
import chop.actions as actions
from chop.ir import MaseGraph
from chop.models.patched.bert import BertConfig, BertModel
from chop.models.patched.bert.modeling_bert import BertSelfAttention, BertEmbeddings
from chop.passes.graph.utils import deepsetattr
from chop.nn.quantized import BertSelfAttentionInteger, LinearInteger, LayerNormInteger
from chop.tools import get_logger, set_excepthook

from mase_components import get_module_dependencies

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
        LinearInteger: {
            "args": {
                "x": "data_in",
            },
            "toolchain": "INTERNAL_RTL",
            "module": "fixed_linear",
            "dependence_files": get_module_dependencies("linear/fixed_linear"),
        },
        LayerNormInteger: {
            "args": {
                "x": "data_in",
            },
            "toolchain": "INTERNAL_RTL",
            "module": "norm",
            "dependence_files": get_module_dependencies("norm/norm"),
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
        else:
            continue
        deepsetattr(model, module[0], new_module)
    return model


def test_emit_verilog_bert():

    # * Define custom configuration
    config = BertConfig()
    config.num_hidden_layers = 1
    config.hidden_size = 384
    config.intermediate_size = 1536

    q_config = {
        "data_in_width": 16,
        "data_in_frac_width": 3,
        "weight_width": 16,
        "weight_frac_width": 3,
        "bias_width": 16,
        "bias_frac_width": 3,
        "data_out_width": 16,
        "data_out_frac_width": 3,
    }

    # * Get model and quantize self attention, linear and layer norm layers
    model = BertModel(config)
    model = bert_module_level_quantize(model, config, q_config)
    logger.info(f"Quantized BERT model: {model}")

    # * Trace the model
    mg = MaseGraph(model, custom_ops=BERT_CUSTOM_OPS)
    mg, _ = passes.init_metadata_analysis_pass(mg)

    # * Save the print tabular to a file
    with open("bert.txt", "w") as f:
        sys.stdout = f
        mg.fx_graph.print_tabular()
        sys.stdout = sys.__stdout__

    # * Add metadata analysis passes
    mg, _ = passes.add_common_metadata_analysis_pass(
        mg,
        pass_args={
            "dummy_in": {"input_ids": torch.randn((1, 128, 384))},
            "add_value": False,
        },
    )

    mg, _ = passes.add_hardware_metadata_analysis_pass(mg)

    # ! TO DO: remove (debug)
    # i = 0
    # for node in mg.fx_graph.nodes:
    #     if i > 5:
    #         break
    #     print(f"\n\nNode: {node.name}")
    #     print(f"Mase type: {node.meta['mase']['common']['mase_type']}")
    #     print(f"Mase op: {node.meta['mase']['common']['mase_op']}")
    #     for k, v in node.meta["mase"]["common"]["args"].items():
    #         print(f"Args/{k}: {v}")

    #     print(f"HW meta: {node.meta['mase']['hardware']}")
    #     breakpoint()

    #     i += 1

    mg, _ = passes.emit_verilog_top_transform_pass(mg)
    # mg, _ = passes.emit_bram_transform_pass(mg)
    mg, _ = passes.emit_internal_rtl_transform_pass(mg)
    mg, _ = passes.emit_cocotb_transform_pass(mg)
    mg, _ = passes.emit_vivado_project_transform_pass(mg)

    actions.simulate(skip_build=False, skip_test=False)


if __name__ == "__main__":
    test_emit_verilog_bert()
