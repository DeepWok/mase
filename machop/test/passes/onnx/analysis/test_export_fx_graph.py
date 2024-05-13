import sys, traceback, pdb
import logging

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel

from chop import MaseGraph, MaseOnnxGraph
from chop.passes import (
    export_fx_graph_analysis_pass,
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
    raise_granularity_transform_pass,
)

import onnxruntime as rt
import os

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)


# sys.excepthook = excepthook


def export_fx_graph_model(
    pretrained: str, monolith=True, task="auto", skip_export=False
):
    # * Get model
    hf_model = AutoModel.from_pretrained(pretrained)

    # * Get dummy input
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    # ! TO DO: bert only works when batch size > 1
    model_inputs = tokenizer(
        [
            "Studies have been shown that owning a dog is good for you",
            "Studies have been shown that owning a dog makes you 10x happier",
        ],
        padding=True,
        return_tensors="pt",
    )

    # * Run HuggingFace model (for debug)
    _ = hf_model(**model_inputs)
    # * Run Onnx runtime
    session = rt.InferenceSession(
        f"{os.environ['HOME']}/.mase/onnx/{pretrained}/model.onnx"
    )
    onnx_model_inputs = {k: v.numpy() for k, v in model_inputs.items()}
    onnx_out = session.run([], onnx_model_inputs)

    # * Export MaseGraph from ONNX graph
    onnx_graph = MaseOnnxGraph.from_pretrained(
        pretrained, monolith=monolith, task=task, skip_export=skip_export
    )
    mg, _ = export_fx_graph_analysis_pass(onnx_graph)
    mg = mg["model"]
    mg, _ = init_metadata_analysis_pass(mg)
    mg, _ = add_common_metadata_analysis_pass(mg, pass_args={"dummy_in": model_inputs})
    # * Run FX GraphModule
    mg_out = mg.model(**model_inputs)

    # assert(torch.equal(onnx_out, mg_out))

    return mg


def test_export_fx_graph_bert():
    mg = export_fx_graph_model("bert-base-uncased", skip_export=True)
    mg, _ = raise_granularity_transform_pass(mg)


def test_export_fx_graph_bloom():
    export_fx_graph_model(
        "bigscience/bloom-1b7",
        skip_export=True,
    )


def test_export_fx_graph_gpt2():
    export_fx_graph_model(
        "openai-community/gpt2",
        skip_export=True,
    )


def test_export_fx_graph_llama():
    export_fx_graph_model(
        "huggyllama/llama-7b",
    )


def test_export_fx_graph_mistral():
    export_fx_graph_model(
        "mistral-community/Mistral-7B-v0.2",
    )


def test_export_fx_graph_opt():
    export_fx_graph_model("facebook/opt-125m", task="text-generation", skip_export=True)


def test_export_fx_graph_swin():
    export_fx_graph_model(
        "microsoft/swin-tiny-patch4-window7-224",
        skip_export=True,
    )


def test_export_fx_graph_t5():
    export_fx_graph_model(
        "google/t5-efficient-tiny", task="text2text-generation", skip_export=True
    )


def test_export_fx_graph_vit():
    export_fx_graph_model(
        "google/vit-base-patch16-224",
        skip_export=True,
    )


def test_export_fx_graph_whisper():
    export_fx_graph_model(
        "openai/whisper-tiny",
        skip_export=True,
    )


if __name__ == "__main__":
    test_export_fx_graph_bert()  # works
    # test_export_fx_graph_opt()  # failing on gather
    # test_export_fx_graph_bloom()  # need to check node by node
    # test_export_fx_graph_t5()

    # test_export_fx_graph_swin()  # need to figure out how to preprocess data
    # test_export_fx_graph_vit()  # need to figure out how to preprocess data
    # test_export_fx_graph_whisper()  # need to figure out how to preprocess data

    # test_export_fx_graph_gpt2()

    # test_export_fx_graph_llama()  # 10GB download
    # test_export_fx_graph_mistral()  # 5GB download
