import sys, os
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[3].as_posix())

from chop.passes.graph import MaseGraph
from chop.dataset import get_dataset_info
from chop.models import get_model, get_model_info, get_tokenizer
from chop.tools.get_input import get_cf_args
from chop.tools import load_config

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.graph_drawer import FxGraphDrawer

import onnx

ROOT = Path(__file__).resolve().parents[4].as_posix()

# TO DO:
# update args ordering
# replace attribute strs with get_attr nodes
# rename onnx namespace attributes
# ask about slice
# onnx nodes know their specific torch module. can change granularity?
# test code gen equivalency


def get_opt_mg():
    config = load_config(ROOT + f"/machop/configs/examples/opt_uniform.toml")
    wikitext_info = get_dataset_info("wikitext2")
    model_info = get_model_info("facebook/opt-125m:patched")
    opt = get_model(
        "facebook/opt-125m:patched",
        task="lm",
        dataset_info=wikitext_info,
        pretrained=True,
    )
    opt_tokenizer = get_tokenizer("facebook/opt-125m:patched")

    if "cf_args" not in config:
        cf_args = get_cf_args(model_info=model_info, task="lm", model=opt)
    else:
        cf_args = config["cf_args"]

    return MaseGraph(model=opt, cf_args=cf_args)


def check(mg):
    mg.model.graph.print_tabular()
    mg.model.recompile()


model_id = ""

# TO DO: enable for all models by checking for encoder/decoder structure or just model.onnx
model_ids = [
    # "albert-base-v2",
    # "facebook/bart-base",
    # "bert-base-uncased",
    # "bigscience/bloom-1b7",
    # "distilbert-base-uncased",
    # "gpt2",
    "facebook/opt-125m",
    # "roberta-base",
    # "t5-base",
    # "runwayml/stable-diffusion-v1-5",
]

for model_id in model_ids:
    if model_id == "runwayml/stable-diffusion-v1-5":
        from optimum.onnxruntime import ORTStableDiffusionPipeline

        save_directory = ROOT + f"mase_output/onnx/{model_id}/"

        if not os.path.exists(save_directory):
            pipeline = ORTStableDiffusionPipeline.from_pretrained(model_id, export=True)
            pipeline.save_pretrained(save_directory)

        text_encoder = onnx.load(save_directory + "text_encoder/model.onnx")
        mg = MaseGraph(text_encoder)
        check(mg)

        unet = onnx.load(save_directory + "/unet/model.onnx")
        mg = MaseGraph(unet)
        check(mg)

        vae_encoder = onnx.load(save_directory + "vae_encoder/model.onnx")
        mg = MaseGraph(vae_encoder)
        check(mg)

        vae_decoder = onnx.load(save_directory + "vae_decoder/model.onnx")
        mg = MaseGraph(vae_decoder)
        check(mg)

    else:
        mg = MaseGraph(model_id)
        check(mg)
