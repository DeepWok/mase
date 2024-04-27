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


def test_export_fx_graph_model(
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

    if "t5" in pretrained:
        # encoder decoder model
        decoder_input_ids = tokenizer(
            "Studies show that", return_tensors="pt"
        ).input_ids
        decoder_input_ids = hf_model._shift_right(decoder_input_ids)
        model_inputs["decoder_input_ids"] = decoder_input_ids
        print(f"model_inputs: {model_inputs.keys()}")

    # * Run HuggingFace model (for debug)
    _ = hf_model(**model_inputs)

    # * Run Onnx runtime
    session = rt.InferenceSession(
        f"{os.environ['HOME']}/.mase/onnx/{pretrained}/model.onnx"
    )
    onnx_model_inputs = {k: v.numpy() for k, v in model_inputs.items()}
    onnx_out = session.run([], onnx_model_inputs)
    # onnx_out = torch.squeeze(torch.Tensor(onnx_out))

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

    # mg, _ = add_hardware_metadata_analysis_pass(mg)

    # assert(torch.equal(onnx_out, mg_out))

    return mg


def test_export_fx_graph_bert():
    mg = test_export_fx_graph_model("bert-base-uncased", skip_export=True)
    mg, _ = raise_granularity_transform_pass(mg)


# def test_export_fx_graph_bloom():
#     test_export_fx_graph_model(
#         "bigscience/bloom-1b7",
#         skip_export=True,
#     )


# def test_export_fx_graph_gpt2():
#     test_export_fx_graph_model(
#         "openai-community/gpt2-medium",
#     )


# def test_export_fx_graph_graphormer():
#     test_export_fx_graph_model(
#         "clefourrier/graphormer-base-pcqm4mv2",
#     )


# def test_export_fx_graph_llama():
#     test_export_fx_graph_model(
#         "huggyllama/llama-7b",
#     )


# def test_export_fx_graph_mistral():
#     test_export_fx_graph_model(
#         "mistral-community/Mistral-7B-v0.2",
#     )


# def test_export_fx_graph_opt():
#     test_export_fx_graph_model(
#         "facebook/opt-125m", task="text-generation", skip_export=True
#     )


# def test_export_fx_graph_swin():
#     test_export_fx_graph_model(
#         "microsoft/swin-tiny-patch4-window7-224",
#     )


# def test_export_fx_graph_t5():
#     test_export_fx_graph_model("t5-base", task="text2text-generation", skip_export=True)


# def test_export_fx_graph_vit():
#     test_export_fx_graph_model(
#         "google/vit-base-patch16-224",
#     )


# def test_export_fx_graph_whisper():
#     test_export_fx_graph_model(
#         "openai/whisper-tiny",
#     )


if __name__ == "__main__":
    test_export_fx_graph_bert()  # works
    # test_export_fx_graph_opt() # failing on gather
    # test_export_fx_graph_bloom() # need to check node by node
    # test_export_fx_graph_t5()

    # test_export_fx_graph_graphormer() # need to figure out how to preprocess data
    # test_export_fx_graph_swin() # need to figure out how to preprocess data
    # test_export_fx_graph_vit() # need to figure out how to preprocess data
    # test_export_fx_graph_whisper() # need to figure out how to preprocess data

    # test_export_fx_graph_gpt2() # too big to download on 4G
    # test_export_fx_graph_llama() # too big to download on 4G
    # test_export_fx_graph_mistral() # too big to download on 4G
