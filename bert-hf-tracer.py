from transformers import AutoConfig, AutoModel
from transformers.utils.fx import symbolic_trace
from torch.fx import GraphModule
from chop.ir import MaseGraph

from chop.models.patched.bert import BertConfig, BertModel
from chop.models.patched.bert.modeling_bert import BertSelfAttention, BertEmbeddings

from chop.passes import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
)

import sys, pdb, traceback
import torch


def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)


# Set the custom exception hook
sys.excepthook = excepthook

config = BertConfig()
config.num_hidden_layers = 3

model = BertModel(config)

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
                "hidden_states": "embeddings",
                "attention_mask": "data_in",
                "head_mask": "data_in",
                "encoder_hidden_states": "data_in",
                "encoder_attention_mask": "data_in",
                "past_key_value": "data_in",
                "output_attentions": "config",
            },
            "mase_component": "attention/self_attention.sv",
        },
    },
    "functions": {},
}

mg = MaseGraph(model, custom_ops=custom_ops)
mg, _ = init_metadata_analysis_pass(mg)

# Save the print tabular to a file
with open("bert.txt", "w") as f:
    sys.stdout = f
    mg.fx_graph.print_tabular()
    sys.stdout = sys.__stdout__

mg, _ = add_common_metadata_analysis_pass(
    mg,
    pass_args={
        "dummy_in": {"input_ids": torch.randint(0, config.vocab_size, (1, 128))}
    },
)

mg, _ = add_hardware_metadata_analysis_pass(mg)

# i = 0
# for node in mg.fx_graph.nodes:
#     if i > 5:
#         break
#     print(f"\n\nNode: {node.name}")
#     for k, v in node.meta["mase"]["hardware"].items():
#         print(f"{k}: {v}")
#     i += 1
