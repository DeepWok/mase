#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import random
import os, sys, logging, traceback, pdb
from chop.passes.graph.analysis.report.report_node import report_node_type_analysis_pass
import pytest
import toml

import torch
import torch.nn as nn

import chop as chop
import chop.passes as passes

from pathlib import Path

from chop.actions import simulate
from chop.tools.logger import set_logging_verbosity
from chop.tools import get_logger

set_logging_verbosity("debug")

logger = get_logger(__name__)


# --------------------------------------------------
#   Model specifications
#   prefer small models for fast test
# --------------------------------------------------
class MLP(torch.nn.Module):
    def __init__(self, features: list[int]) -> None:
        super().__init__()

        layers = []
        for in_f, out_f in zip(features[:-1], features[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)
        print(self.model)

    def forward(self, x):
        return self.model(x)


def test_emit_verilog_mxint_mlp(seed: int = 10):
    torch.manual_seed(seed)
    random.seed(seed)

    block_size = random.randint(2, 10)
    batch_parallelism = random.randint(2, 10)
    mlp_depth = random.randint(1, 10)
    mlp_features = [block_size * random.randint(1, 10) for _ in range(mlp_depth + 1)]

    params = {
        "seed": seed,
        "block_size": block_size,
        "batch_parallelism": batch_parallelism,
        "m_width": (m_width := random.randint(4, 10)),
        "e_width": random.randint(3, min(m_width - 1, 10)),
        "batches": batch_parallelism * random.randint(1, 20),
        "num_batches": random.randint(1, 20),
    }

    mlp = MLP(mlp_features)
    input_shape = (mlp_features[0],)
    logger.info(
        f"{block_size=}, {batch_parallelism=}, {params['e_width']=}, {params['m_width']=}, {params['batches']=}"
    )

    shared_emit_verilog_mxint(mlp, input_shape, params)


def test_emit_verilog_mxint_linear(seed: int = 10):
    torch.manual_seed(seed)
    random.seed(seed)

    block_size = random.randint(2, 10)
    batch_parallelism = random.randint(2, 10)
    IN_FEATURES = block_size * random.randint(1, 10)
    OUT_FEATURES = block_size * random.randint(1, 10)

    params = {
        "seed": seed,
        "block_size": block_size,
        "batch_parallelism": batch_parallelism,
        "m_width": (m_width := random.randint(5, 10)),
        "e_width": random.randint(4, min(m_width - 1, 10)),
        "batches": batch_parallelism * random.randint(1, 20),
        "num_batches": random.randint(1, 20),
    }

    class LinearModel(torch.nn.Module):
        def __init__(self, IN_FEATURES, OUT_FEATURES) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(IN_FEATURES, OUT_FEATURES)

        def forward(self, x):
            return self.fc1(x)

    linear = LinearModel(IN_FEATURES, OUT_FEATURES)
    input_shape = (IN_FEATURES,)
    logger.info(
        f"{block_size=}, {batch_parallelism=}, {params['e_width']=}, {params['m_width']=}, {params['batches']=}"
    )

    shared_emit_verilog_mxint(linear, input_shape, params)


def shared_emit_verilog_mxint(model, input_shape, params: dict):
    # Set seeds
    torch.manual_seed(params["seed"])
    random.seed(params["seed"])

    block_size = params["block_size"]
    batch_parallelism = params["batch_parallelism"]
    m_width = params["m_width"]
    e_width = params["e_width"]
    batches = params["batches"]
    num_batches = params["num_batches"]

    mg = chop.MaseGraph(model=model)
    x = torch.randn((batches, *input_shape))
    dummy_in = {"x": x}

    mg, _ = passes.init_metadata_analysis_pass(mg, None)
    mg, _ = passes.add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})

    quan_args = {
        "by": "type",
        "default": {
            "config": {
                "name": "mxint",
                "data_in_width": m_width,
                "data_in_exponent_width": e_width,
                "data_in_block_size": [batch_parallelism, block_size],
                "weight_width": m_width,
                "weight_exponent_width": e_width,
                "weight_block_size": [block_size, block_size],
                "bias_width": m_width,
                "bias_exponent_width": e_width,
                "bias_block_size": [1, block_size],
            }
        },
    }

    mg, _ = passes.quantize_transform_pass(mg, quan_args)
    _ = report_node_type_analysis_pass(mg)
    mg, _ = passes.report_node_meta_param_analysis_pass(mg)

    # Parallelism adjustments
    for node in mg.fx_graph.nodes:
        node_meta = node.meta["mase"].parameters["common"]
        args = node_meta["args"]
        results = node_meta["results"]
        match node_meta["mase_op"]:
            case "linear":
                args["data_in_0"]["parallelism_0"] = block_size
                args["data_in_0"]["parallelism_1"] = batch_parallelism
                args["weight"]["parallelism_0"] = block_size
                args["weight"]["parallelism_1"] = block_size
                args["bias"]["parallelism_0"] = block_size
                args["bias"]["parallelism_1"] = 1

                results["data_out_0"]["parallelism_0"] = block_size
                results["data_out_0"]["parallelism_1"] = batch_parallelism
            case "relu":
                args["data_in_0"]["parallelism_0"] = block_size
                args["data_in_0"]["parallelism_1"] = batch_parallelism
                results["data_out_0"]["parallelism_0"] = block_size
                results["data_out_0"]["parallelism_1"] = batch_parallelism

    mg, _ = passes.add_hardware_metadata_analysis_pass(mg)
    mg, _ = passes.report_node_hardware_type_analysis_pass(mg)

    mg, _ = passes.emit_verilog_top_transform_pass(mg)
    mg, _ = passes.emit_bram_transform_pass(mg)
    mg, _ = passes.emit_internal_rtl_transform_pass(mg)
    mg, _ = passes.emit_cocotb_transform_pass(
        mg,
        pass_args={
            "wait_time": 10 * block_size * batch_parallelism * num_batches,
            "wait_unit": "us",
            "num_batches": num_batches,
        },
    )

    simulate(
        skip_build=False,
        skip_test=False,
        simulator="verilator",
        waves=True,
    )

    logger.info(
        f"{block_size=}, {batch_parallelism=}, {m_width=}, {e_width=}, {batches=}"
    )


if __name__ == "__main__":
    seed = os.getenv("COCOTB_SEED")
    if seed is None:
        seed = random.randrange(sys.maxsize)
        logger.info(f"Generated {seed=}")
    else:
        seed = int(seed)
        logger.info(f"Using provided {seed=}")
    test_emit_verilog_mxint_linear(seed)
    # test_emit_verilog_mxint_mlp(seed)
    logger.info(f"{seed=}")
