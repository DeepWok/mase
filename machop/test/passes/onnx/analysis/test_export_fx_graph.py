import sys, traceback, pdb
import logging

import torch
from transformers import AutoConfig, AutoTokenizer

from chop import MaseGraph, MaseOnnxGraph
from chop.passes import (
    export_fx_graph_analysis_pass,
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
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


sys.excepthook = excepthook


def test_export_fx_graph_model(
    pretrained: str, monolith=True, task="auto", skip_export=False
):
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model_inputs = tokenizer(["Hello, world!", "How are you?"], return_tensors="pt")
    # model_inputs["decoder_input_ids"] = model_inputs["input_ids"]

    onnx_model_inputs = {k: v.numpy() for k, v in model_inputs.items()}

    onnx_graph = MaseOnnxGraph.from_pretrained(
        pretrained, monolith=monolith, task=task, skip_export=skip_export
    )
    mg, _ = export_fx_graph_analysis_pass(onnx_graph)
    mg = mg["model"]
    mg, _ = init_metadata_analysis_pass(mg)
    mg, _ = add_common_metadata_analysis_pass(mg, pass_args={"dummy_in": model_inputs})

    session = rt.InferenceSession(
        f"{os.environ['HOME']}/.mase/onnx/{pretrained}/model.onnx"
    )

    onnx_out = session.run([], onnx_model_inputs)
    onnx_out = torch.squeeze(torch.Tensor(onnx_out))
    mg_out = mg.model(**model_inputs)

    # mg, _ = add_hardware_metadata_analysis_pass(mg)

    # assert(torch.equal(onnx_out, mg_out))


def test_export_fx_graph_bert():
    test_export_fx_graph_model("bert-base-uncased", skip_export=True)


def test_export_fx_graph_bloom():
    test_export_fx_graph_model(
        "bigscience/bloom-1b7",
    )


def test_export_fx_graph_gpt2():
    test_export_fx_graph_model(
        "openai-community/gpt2-medium",
    )


def test_export_fx_graph_graphormer():
    test_export_fx_graph_model(
        "clefourrier/graphormer-base-pcqm4mv2",
    )


def test_export_fx_graph_llama():
    test_export_fx_graph_model(
        "meta-llama/Llama-2-7b",
    )


def test_export_fx_graph_longformer():
    test_export_fx_graph_model(
        "allenai/longformer-base-4096",
    )


def test_export_fx_graph_mixtral():
    test_export_fx_graph_model(
        "mistralai/Mixtral-8x7B-v0.1",
    )


def test_export_fx_graph_opt():
    test_export_fx_graph_model(
        "facebook/opt-125m", task="text-generation", skip_export=True
    )


def test_export_fx_graph_swin():
    test_export_fx_graph_model(
        "microsoft/swin-tiny-patch4-window7-224",
    )


def test_export_fx_graph_t5():
    test_export_fx_graph_model("t5-base", task="text2text-generation", skip_export=True)


def test_export_fx_graph_vit():
    test_export_fx_graph_model(
        "google/vit-base-patch16-224",
    )


def test_export_fx_graph_whisper():
    test_export_fx_graph_model(
        "openai/whisper-tiny",
    )


if __name__ == "__main__":
    # test_export_fx_graph_bert()
    test_export_fx_graph_bloom()
    # test_export_fx_graph_gpt2()
    # test_export_fx_graph_graphormer()
    # test_export_fx_graph_llama()
    # test_export_fx_graph_longformer()
    # test_export_fx_graph_mixtral()
    # test_export_fx_graph_opt()
    # test_export_fx_graph_swin()
    # test_export_fx_graph_t5()
    # test_export_fx_graph_vit()
    # test_export_fx_graph_whisper()
