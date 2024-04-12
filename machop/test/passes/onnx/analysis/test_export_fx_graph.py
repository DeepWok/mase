import sys, traceback, pdb
import logging

import torch
from transformers.models.bert import BertConfig, BertTokenizer
from optimum.exporters.onnx.model_configs import BertOnnxConfig

from chop import MaseGraph, MaseOnnxGraph
from chop.passes import (
    export_fx_graph_analysis_pass,
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)


sys.excepthook = excepthook


def test_export_fx_graph():

    # Initialize the config, model and tokenizer.
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer("Hello, world!", return_tensors="pt")

    onnx_graph = MaseOnnxGraph.from_pretrained("bert-base-uncased")
    mg, _ = export_fx_graph_analysis_pass(onnx_graph)

    mg, _ = init_metadata_analysis_pass(mg)
    mg, _ = add_common_metadata_analysis_pass(mg, pass_args={"dummy_in": tokens})


if __name__ == "__main__":
    test_export_fx_graph()
