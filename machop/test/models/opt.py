#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# sys.path.append(
#     os.path.join(
#         os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "machop"
#     )
# )

sys.path.append(Path(__file__).resolve().parents[3].joinpath("machop").as_posix())

from chop.dataset import get_dataset_info, MaseDataModule
from chop.models import model_map
from chop.passes import PASSES
from chop.passes.graph.mase_graph import MaseGraph
from chop.tools import load_config
from chop.tools.get_input import get_cf_args, get_dummy_input


# --------------------------------------------------
#   Emit Verilog using Mase
# --------------------------------------------------
def main():
    machop_dir = Path(__file__).resolve().parents[3] / "machop"
    config_toml = machop_dir / "configs" / "examples" / "opt_uniform.toml"
    assert config_toml.exists(), f"config_toml {config_toml} does not exist"
    config = load_config(config_toml)

    load_pretrained = True

    # OPT
    wikitext_info = get_dataset_info("wikitext2")
    opt_dict = model_map["facebook/opt-125m@patched"](
        name="facebook/opt-125m@patched",
        task="lm",
        info=wikitext_info,
        pretrained=load_pretrained,
    )
    opt = opt_dict["model"]
    opt_tokenizer = opt_dict["tokenizer"]

    if "cf_args" not in config:
        cf_args = get_cf_args(
            model_name="facebook/opt-125m@patched", task="lm", model=opt
        )
    else:
        cf_args = config["cf_args"]
    print(f"cf_args: {cf_args}")

    data_module = MaseDataModule(
        name="wikitext2",
        batch_size=1,
        num_workers=os.cpu_count(),
        max_token_len=128,
        tokenizer=opt_tokenizer,
        load_from_cache_file=True,
        model_name="facebook/opt-125m@patched",
    )
    data_module.prepare_data()
    data_module.setup()

    graph = MaseGraph(model=opt, cf_args=cf_args)
    graph = PASSES["init_metadata"](graph, pass_args=None)

    dummy_in = get_dummy_input(datamodule=data_module, task="lm", is_nlp_model=True)
    if len(graph.model.additional_inputs) > 0:
        dummy_in = dummy_in | graph.model.additional_inputs
    graph = PASSES["add_common_metadata"](graph, pass_args=dummy_in)
    # graph = PASSES["add_hardware_metadata"](graph, pass_args=None)
    graph = PASSES["add_software_metadata"](graph, pass_args=None)

    graph = PASSES["quantize"](graph, pass_args=config["passes"]["quantize"])


# --------------------------------------------------
#   Execution
# --------------------------------------------------
main()
