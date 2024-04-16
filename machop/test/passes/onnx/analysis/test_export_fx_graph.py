import sys, traceback, pdb
import logging

import torch
from transformers import AutoConfig, AutoTokenizer

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


def test_export_fx_graph_model(pretrained):
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    tokens = tokenizer(["Hello, world!", "How are you?"], return_tensors="pt")

    onnx_graph = MaseOnnxGraph.from_pretrained(pretrained)
    mg, _ = export_fx_graph_analysis_pass(onnx_graph)

    mg, _ = init_metadata_analysis_pass(mg)
    mg, _ = add_common_metadata_analysis_pass(mg, pass_args={"dummy_in": tokens})


def test_export_fx_graph_alberta():
    test_export_fx_graph_model(
        "albert-base-v2",
    )


def test_export_fx_graph_bart():
    test_export_fx_graph_model(
        "facebook/bart-base",
    )


def test_export_fx_graph_bert():
    test_export_fx_graph_model(
        "bert-base-uncased",
    )


def test_export_fx_graph_bloom():
    test_export_fx_graph_model(
        "bigscience/bloom-1b7",
    )


def test_export_fx_graph_distilbert():
    test_export_fx_graph_model(
        "distilbert-base-uncased",
    )


def test_export_fx_graph_gpt2():
    test_export_fx_graph_model(
        "gpt2",
    )


def test_export_fx_graph_opt():
    test_export_fx_graph_model(
        "facebook/opt-125m",
    )


def test_export_fx_graph_roberta():
    test_export_fx_graph_model(
        "roberta-base",
    )


def test_export_fx_graph_t5():
    test_export_fx_graph_model(
        "t5-base",
    )


def test_export_fx_graph_stable_diffusion():
    test_export_fx_graph_model(
        "runwayml/stable-diffusion-v1-5",
    )


if __name__ == "__main__":
    # test_export_fx_graph_alberta()
    # test_export_fx_graph_bart()
    test_export_fx_graph_bert()
    # test_export_fx_graph_bloom()
    # test_export_fx_graph_distilbert()
    # test_export_fx_graph_gpt2()
    # test_export_fx_graph_opt()
    # test_export_fx_graph_roberta()
    # test_export_fx_graph_t5()
    # test_export_fx_graph_stable_diffusion()
