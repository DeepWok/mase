import math, time, os, logging, torch, glob, shutil

from chop.passes.graph.utils import vf, v2p, init_project
from chop.passes.graph.transforms.quantize.quantizers import (
    integer_quantizer_for_hw,
    integer_quantizer,
)

logger = logging.getLogger(__name__)

from pathlib import Path

import cocotb
from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.z_qlayers.tensor_cast import quantize_to_int

import dill
import inspect


@cocotb.test()
async def test(dut):
    from pathlib import Path
    import dill
    from cocotb.triggers import Timer

    tb_path = Path.home() / ".mase" / "top" / "hardware" / "test" / "mase_top_tb"
    with open(tb_path / "tb_obj.dill", "rb") as f:
        tb = dill.load(f)(dut)

    tb.initialize()

    in_tensors = tb.generate_inputs(batches=3)
    exp_out = tb.model(*list(in_tensors.values()))

    tb.load_drivers(in_tensors)
    tb.load_monitors(exp_out)

    await Timer(100, units="us")


def _emit_cocotb_test(graph):
    test_template = f"""
import cocotb

{inspect.getsource(test)}
"""

    tb_path = Path.home() / ".mase" / "top" / "hardware" / "test" / "mase_top_tb"
    tb_path.mkdir(parents=True, exist_ok=True)
    with open(tb_path / "test.py", "w") as f:
        f.write(test_template)

    verilator_build = f"""
#!/bin/bash
# This script is used to build the verilator simulation
verilator --binary --build {verilator_buff}
"""
    verilator_file = os.path.join(sim_dir, "build.sh")
    with open(verilator_file, "w", encoding="utf-8") as outf:
        outf.write(verilator_build)


def _emit_cocotb_tb(graph):
    class MaseGraphTB(Testbench):
        def __init__(self, dut):
            super().__init__(dut, dut.clk, dut.rst)

            # Instantiate as many drivers as required inputs to the model
            for arg in graph.meta["mase"]["common"]["args"].keys():
                self.input_drivers.append(
                    StreamDriver(
                        dut.clk,
                        getattr(dut, arg),
                        getattr(dut, f"{arg}_valid"),
                        getattr(dut, f"{arg}_ready"),
                    )
                )

            # Instantiate as many monitors as required outputs
            for result in graph.meta["mase"]["common"]["results"].keys():
                self.output_monitors.append(
                    StreamMonitor(
                        dut.clk,
                        getattr(dut, result),
                        getattr(dut, f"{result}_valid"),
                        getattr(dut, f"{result}_ready"),
                    )
                )

            self.model = graph.model

            # To do: precision per input argument
            self.input_precision = graph.meta["mase"]["common"]["args"]["data_in_0"][
                "precision"
            ]

        def generate_inputs(self, batches):
            """
            Generate inputs for the model by sampling a random tensor
            for each input argument, according to its shape

            :param batches: number of batches to generate for each argument
            :type batches: int
            :return: a dictionary of input arguments and their corresponding tensors
            :rtype: Dict
            """
            inputs = {}
            for arg, arg_info in graph.meta["mase"]["common"]["args"].items():
                # Batch dimension always set to 1 in metadata
                inputs[arg] = torch.rand(([batches] + arg_info["shape"][1:]))
            return inputs

        def load_drivers(self, in_tensors):
            for arg_idx, arg_batches in enumerate(in_tensors.values()):
                # Quantize input tensor according to precision
                if len(self.input_precision) > 1:
                    arg_batches = integer_quantizer(
                        arg_batches,
                        width=self.input_precision[0],
                        frac_width=self.input_precision[1],
                    )
                    # Convert to integer equivalent of fixed point representation
                    arg_batches = (
                        (arg_batches * (2 ** self.input_precision[1])).int().tolist()
                    )
                else:
                    # TO DO: convert to integer equivalent of floating point representation
                    pass

                # Append to input driver
                for batch in arg_batches:
                    self.input_drivers[arg_idx].append(batch)

        def load_monitors(self, expectation):
            self.output_monitors[-1].expect(expectation.tolist())

    # Serialize testbench object to be instantiated within test by cocotb runner
    cls_obj = MaseGraphTB
    tb_path = Path.home() / ".mase" / "top" / "hardware" / "test" / "mase_top_tb"
    tb_path.mkdir(parents=True, exist_ok=True)
    with open(tb_path / "tb_obj.dill", "wb") as file:
        dill.dump(cls_obj, file)
    with open(tb_path / "__init__.py", "w") as file:
        file.write("from .test import test")


def emit_cocotb_transform_pass(graph, pass_args={}):
    """
    Emit test bench and related files for simulation

    :param graph: a MaseGraph
    :type graph: MaseGraph
    :param pass_args: this pass requires additional arguments which is explained below, defaults to {}
    :type pass_args: _type_, optional
    :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
    :rtype: tuple(MaseGraph, Dict)

    - pass_args
        - project_dir -> str : the directory of the project
    """
    logger.info("Emitting testbench...")
    project_dir = (
        pass_args["project_dir"] if "project_dir" in pass_args.keys() else "top"
    )

    init_project(project_dir)

    _emit_cocotb_test(graph)
    _emit_cocotb_tb(graph)

    return graph, None
