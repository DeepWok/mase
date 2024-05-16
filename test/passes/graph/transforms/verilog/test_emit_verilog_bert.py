from transformers import AutoConfig, AutoModel
from transformers.utils.fx import symbolic_trace

from torch import Tensor
from torch.fx import GraphModule
from chop.ir import MaseGraph

from chop.models.patched.bert import BertConfig, BertModel
from chop.models.patched.bert.modeling_bert import BertSelfAttention, BertEmbeddings

from chop.passes import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
    emit_verilog_top_transform_pass,
    emit_internal_rtl_transform_pass,
    emit_bram_transform_pass,
)

import sys, pdb, traceback
import torch


def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)


# sys.excepthook = excepthook


def test_emit_verilog_bert():

    # * Get model with custom configuration
    config = BertConfig()
    config.num_hidden_layers = 3
    config.hidden_size = 384
    config.intermediate_size = 1536

    model = BertModel(config)

    # * Define custom ops (leaf submodules during tracing)
    custom_ops = {
        "modules": {
            BertEmbeddings: {
                "args": {
                    "input_ids": "data_in",
                    "token_type_ids": "data_in",
                    "position_ids": "data_in",
                    "inputs_embeds": "data_in",
                    "past_key_values_length": "data_in",
                },
            },
            BertSelfAttention: {
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
                "module": "attention/fixed_self_attention.sv",
            },
        },
        "functions": {},
    }

    # * Trace the model
    mg = MaseGraph(model, custom_ops=custom_ops)
    mg, _ = init_metadata_analysis_pass(mg)

    # * Save the print tabular to a file
    with open("bert.txt", "w") as f:
        sys.stdout = f
        mg.fx_graph.print_tabular()
        sys.stdout = sys.__stdout__

    # * Add metadata analysis passes
    mg, _ = add_common_metadata_analysis_pass(
        mg,
        pass_args={
            "dummy_in": {"input_ids": torch.randn((1, 128, 384))},
            "add_value": False,
        },
    )

    mg, _ = add_hardware_metadata_analysis_pass(mg)

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

    mg, _ = emit_verilog_top_transform_pass(mg)
    # mg, _ = emit_internal_rtl_transform_pass(mg)
    # mg, _ = emit_bram_transform_pass(mg)


if __name__ == "__main__":
    test_emit_verilog_bert()
