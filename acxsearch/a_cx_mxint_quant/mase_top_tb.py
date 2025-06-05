from pathlib import Path

import cocotb
import logging, torch
from pathlib import Path

logger = logging.getLogger(__name__)

from pathlib import Path

import cocotb
from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
import sys
from os import getenv, PathLike

import torch
from pathlib import Path
import time
import warnings
from cocotb.runner import get_runner, get_results

from chop.tools import get_logger
import mase_components
from mase_components import get_modules

import glob, os

def simulate(
    model: torch.nn.Module = None,
    model_info=None,
    task: str = "",
    dataset_info=None,
    data_module=None,
    load_name: PathLike = None,
    load_type: str = None,
    run_emit: bool = False,
    skip_build: bool = False,
    skip_test: bool = False,
    trace_depth: int = 3,
    gui: bool = False,
    waves: bool = False,
    simulator: str = "verilator",
    pass_args = {},
):
    SIM = getenv("SIM", simulator)
    runner = get_runner(SIM)

    project_dir = (
        pass_args["project_dir"]
        if "project_dir" in pass_args.keys()
        else Path.home() / ".mase" / "top"
    )

    if not skip_build:
        # To do: extract from mz checkpoint
        if simulator == "questa":
            sources = glob.glob(os.path.join(project_dir / "hardware" / "rtl", "*.sv"))
            build_args = []

        elif simulator == "verilator":
            # sources = ["../../../top.sv"]
            sources = glob.glob(os.path.join(project_dir / "hardware" / "rtl", "*.sv"))
            build_args = [
                "-Wno-fatal",
                "-Wno-lint",
                "-Wno-style",
                "--trace-fst",
                "--trace-structs",
                "--trace-depth",
                str(trace_depth),
                "--unroll-count",
                "16384"
            ]
        else:
            raise ValueError(f"Unrecognized simulator: {simulator}")

        includes = [
            project_dir / "hardware" / "rtl",
        ] + [
            Path(mase_components.__file__).parent / module / "rtl"
            for module in get_modules()
        ]

        build_start = time.time()

        runner.build(
            verilog_sources=sources,
            includes=includes,
            hdl_toplevel="top",
            build_args=build_args,
            parameters=[],  # use default parameters,
        )

        build_end = time.time()
        logger.info(f"Build finished. Time taken: {build_end - build_start:.2f}s")

    if not skip_test:
        # Add tb file to python path

        # sys.path.append(str(pass_args["test_dir"]))

        test_start = time.time()
        runner.test(
            hdl_toplevel="top",
            test_module="mase_top_tb",
            hdl_toplevel_lang="verilog",
            gui=gui,
            waves=waves,
        )
        test_end = time.time()
        logger.info(f"Test finished. Time taken: {test_end - test_start:.2f}s")

class MaseGraphTB(Testbench):
    def __init__(self, dut, fail_on_checks=True):
        super().__init__(dut, dut.clk, dut.rst, fail_on_checks=fail_on_checks)

        # Instantiate as many drivers as required inputs to the model
        self.input_drivers = {}
        self.output_monitors = {}

        arg = "data_in_0"
        result = "data_out_0"
        self.input_drivers[arg] = StreamDriver(
            dut.clk,
            getattr(dut, arg),
            getattr(dut, f"{arg}_valid"),
            getattr(dut, f"{arg}_ready"),
        )
        self.input_drivers[arg].log.setLevel(logging.DEBUG)

        # Instantiate as many monitors as required outputs
        self.output_monitors[result] = StreamMonitor(
            dut.clk,
            getattr(dut, result),
            getattr(dut, f"{result}_valid"),
            getattr(dut, f"{result}_ready"),
            check=False,
        )
        self.output_monitors[result].log.setLevel(logging.DEBUG)

    def generate_inputs(self, batches):
        """
        Generate inputs for the model by sampling a random tensor
        for each input argument, according to its shape

        :param batches: number of batches to generate for each argument
        :type batches: int
        :return: a dictionary of input arguments and their corresponding tensors
        :rtype: Dict
        """
        # ! TO DO: iterate through graph.args instead to generalize
        inputs = torch.randn(batches, self.get_parameter(f"DATA_IN_0_TENSOR_SIZE_DIM_1"), self.get_parameter(f"DATA_IN_0_TENSOR_SIZE_DIM_0"))
        outputs = torch.randn(batches, self.get_parameter(f"DATA_OUT_0_TENSOR_SIZE_DIM_1"), self.get_parameter(f"DATA_OUT_0_TENSOR_SIZE_DIM_0"))
        return inputs, outputs

    def load_drivers(self, in_tensors):
        from mase_cocotb.utils import fixed_preprocess_tensor

        in_data_blocks = fixed_preprocess_tensor(
            tensor=in_tensors,
            q_config={
                "width": self.get_parameter(f"DATA_IN_0_PRECISION_0"),
                "frac_width": self.get_parameter(
                    f"DATA_IN_0_PRECISION_1"
                ),
            },
            parallelism=[
                self.get_parameter(f"DATA_IN_0_PARALLELISM_DIM_1"),
                self.get_parameter(f"DATA_IN_0_PARALLELISM_DIM_0"),
            ],
            floor=True,
        )

        # Append all input blocks to input driver
        # ! TO DO: generalize
        block_size = self.get_parameter(
            "DATA_IN_0_PARALLELISM_DIM_0"
        ) * self.get_parameter("DATA_IN_0_PARALLELISM_DIM_1")
        for block in in_data_blocks:
            if len(block) < block_size:
                block = block + [0] * (block_size - len(block))
            self.input_drivers["data_in_0"].append(block)

    def load_monitors(self, expectation):
        from mase_cocotb.utils import fixed_preprocess_tensor

        # Process the expectation tensor
        output_blocks = fixed_preprocess_tensor(
            tensor=expectation,
            q_config={
                "width": self.get_parameter(f"DATA_OUT_0_PRECISION_0"),
                "frac_width": self.get_parameter(f"DATA_OUT_0_PRECISION_1"),
            },
            parallelism=[
                self.get_parameter(f"DATA_OUT_0_PARALLELISM_DIM_1"),
                self.get_parameter(f"DATA_OUT_0_PARALLELISM_DIM_0"),
            ],
            floor=True,
        )

        # Set expectation for each monitor
        for block in output_blocks:
            # ! TO DO: generalize to multi-output models
            if len(block) < self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_0"):
                block = block + [0] * (
                    self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_0") - len(block)
                )
            self.output_monitors["data_out_0"].expect(block)

        # Drive the in-flight flag for each monitor
        self.output_monitors["data_out_0"].in_flight = True

@cocotb.test()
async def test(dut):

    tb = MaseGraphTB(dut, fail_on_checks=True)
    await tb.initialize()

    in_tensors, out_tensors = tb.generate_inputs(batches=10)

    tb.load_drivers(in_tensors)
    tb.load_monitors(out_tensors)

    await tb.wait_end(timeout=0.1, timeout_unit="s")


if __name__ == "__main__":
    pass_args = {
        "project_dir": Path("./int_linear"),
    }
    simulate(skip_build=False, skip_test=False, simulator="verilator", waves=True, gui=False, pass_args=pass_args)