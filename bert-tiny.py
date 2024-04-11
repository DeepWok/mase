from transformers.models.bert import BertConfig, BertTokenizer

from chop import MaseGraph
from chop.passes import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
    report_node_type_analysis_pass,
    report_node_meta_param_analysis_pass,
    report_graph_analysis_pass
)

import sys, traceback, pdb

def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)

# Set the custom exception hook
sys.excepthook = excepthook

# Initialize the config, model and tokenizer.

cf = BertConfig(
    num_hidden_layers=3,
    num_attention_heads=32,
    hidden_size=384,
    intermediate_size=1536,
)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
example_input = tokenizer("Hello, world!", return_tensors="pt")
print(example_input["input_ids"])


# Generate a MaseGraph, which encapsulates the intermediate representation (IR) that MASE uses, then initialize this by adding common metadata. See [here](https://deepwok.github.io/mase/modules/api/analysis/add_metadata.html#chop.passes.graph.analysis.add_metadata.add_common_metadata.add_common_metadata_analysis_pass) for a reference of how metadata is initialized.

from optimum.exporters.onnx.model_configs import BertOnnxConfig

config = BertOnnxConfig(config=cf)
mg = MaseGraph("bert-base-uncased", onnx_config={"encoder_model": config})

mg, _ = init_metadata_analysis_pass(mg)

print(f"=============================================")
# print(mg.model)

mg.model.additional_inputs = []
mg.model.patched_op_names = []
mg, _ = add_common_metadata_analysis_pass(mg, pass_args={"dummy_in": example_input})

