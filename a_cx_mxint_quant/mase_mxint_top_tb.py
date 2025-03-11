from pathlib import Path

import cocotb
import logging, torch
from pathlib import Path

logger = logging.getLogger(__name__)

from pathlib import Path

import cocotb
from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import MultiSignalStreamDriver, MultiSignalErrorThresholdStreamMonitor, MultiSignalStreamMonitor
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
from cocotb.utils import get_sim_time
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
            test_module="mase_mxint_top_tb",
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
        self.input_drivers[arg] = MultiSignalStreamDriver(
            dut.clk, (dut.mdata_in_0, dut.edata_in_0), 
            dut.data_in_0_valid, dut.data_in_0_ready
        )
        # self.input_drivers[arg].log.setLevel(logging.DEBUG)

        # Instantiate as many monitors as required outputs
        self.output_monitors[result] = MultiSignalStreamMonitor(
            dut.clk,
            (dut.mdata_out_0, dut.edata_out_0),
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            check=False,
        )
        # self.output_monitors[result].log.setLevel(logging.DEBUG)

    def generate_inputs(self, batches, model=None):
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
        if model is not None:
            outputs = model(inputs)
        else:
            outputs = torch.randn(batches, self.get_parameter(f"DATA_OUT_0_TENSOR_SIZE_DIM_1"), self.get_parameter(f"DATA_OUT_0_TENSOR_SIZE_DIM_0"))
        return inputs, outputs

    def preprocess_tensor_for_mxint(self, tensor, config, parallelism):
        from mase_components.linear_layers.mxint_operators.test.utils import mxint_hardware
        from mase_components.linear_layers.mxint_operators.test.utils import pack_tensor_to_mx_listed_chunk

        (qtensor, mtensor, etensor) = mxint_hardware(tensor, config, parallelism)
        tensor_inputs = pack_tensor_to_mx_listed_chunk(mtensor, etensor, parallelism)
        return tensor_inputs

    def load_drivers(self, in_tensors):
        for i in range(in_tensors.shape[0]):
            data_0_inputs = self.preprocess_tensor_for_mxint(
                tensor=in_tensors[i],
                config={
                    "width": self.get_parameter("DATA_IN_0_PRECISION_0"),
                    "exponent_width": self.get_parameter("DATA_IN_0_PRECISION_1"),
                    "round_bits": 4
                },
                parallelism=[self.get_parameter("DATA_IN_0_PARALLELISM_DIM_1"), self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0")] 
            )
            self.input_drivers["data_in_0"].load_driver(data_0_inputs)

    def load_monitors(self, expectation):
        for i in range(expectation.shape[0]):
            exp_out = self.preprocess_tensor_for_mxint(
                tensor=expectation[i],
                config={
                    "width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
                    "exponent_width": self.get_parameter("DATA_OUT_0_PRECISION_1"),
                    "round_bits": 4
                },
                parallelism=[self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_1"), self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_0")] 
            )
            self.output_monitors["data_out_0"].load_monitor(exp_out)

import torch.nn as nn
@cocotb.test()
async def test(dut):
    # cocotb.start_soon(check_signal(dut))
    tb = MaseGraphTB(dut, fail_on_checks=True)
    await tb.initialize()
    in_tensors, out_tensors = tb.generate_inputs(batches=1)

    tb.load_drivers(in_tensors)
    tb.load_monitors(out_tensors)

    await tb.wait_end(timeout=100, timeout_unit="ms")

from cocotb.triggers import *
async def check_signal(dut):
    await Timer(40, units="ns")
    # Initialize counters for each data handshake interface
    data_in_0_count = 0
    data_out_0_count = 0
    linear1_data_count = 0
    act_data_count = 0
    linear2_data_count = 0
    norm1_data_count = 0
    attention_data_count = 0
    norm2_data_count = 0
    add_data_count = 0
    add1_data_count = 0
    out_depth = 192/4
    # Initialize timestamps for measuring handshake intervals
    data_in_time = get_sim_time(units='ns')
    data_out_time = get_sim_time(units='ns')
    linear1_time = get_sim_time(units='ns')
    act_time = get_sim_time(units='ns')
    linear2_time = get_sim_time(units='ns')
    norm1_time = get_sim_time(units='ns')
    attention_time = get_sim_time(units='ns')
    norm2_time = get_sim_time(units='ns')
    add_time = get_sim_time(units='ns')
    add1_time = get_sim_time(units='ns')

    while True:
        await RisingEdge(dut.clk)
        await ReadOnly()

        # Count handshakes for main input/output
        if dut.data_in_0_valid.value and dut.data_in_0_ready.value:
            data_in_0_count += 1
            if data_in_0_count == out_depth:
                data_in_0_count = 0
                new_data_in_time = get_sim_time(units='ns')
                diff_data_in = new_data_in_time - data_in_time
                data_in_time = get_sim_time(units='ns')
                print(f"data_in_0 handshake time: {diff_data_in}")

        if dut.data_out_0_valid.value and dut.data_out_0_ready.value:
            data_out_0_count += 1
            if data_out_0_count == out_depth:
                data_out_0_count = 0
                new_data_out_time = get_sim_time(units='ns')
                diff_data_out = new_data_out_time - data_out_time
                data_out_time = get_sim_time(units='ns')
                print(f"data_out_0 handshake time: {diff_data_out}")

        if dut.stream_blocks_0_linear1_data_out_0_valid.value and dut.stream_blocks_0_linear1_data_out_0_ready.value:
            linear1_data_count += 1
            if linear1_data_count == out_depth:
                linear1_data_count = 0
                new_linear1_time = get_sim_time(units='ns')
                diff_linear1 = new_linear1_time - linear1_time
                linear1_time = get_sim_time(units='ns')
                print(f"linear1 handshake time: {diff_linear1}")

        if dut.stream_blocks_0_act_data_out_0_valid.value and dut.stream_blocks_0_act_data_out_0_ready.value:
            act_data_count += 1
            if act_data_count == out_depth:
                act_data_count = 0
                new_act_time = get_sim_time(units='ns')
                diff_act = new_act_time - act_time
                act_time = get_sim_time(units='ns')
                print(f"act handshake time: {diff_act}")

        if dut.stream_blocks_0_linear2_data_out_0_valid.value and dut.stream_blocks_0_linear2_data_out_0_ready.value:
            linear2_data_count += 1
            if linear2_data_count == out_depth:
                linear2_data_count = 0
                new_linear2_time = get_sim_time(units='ns')
                diff_linear2 = new_linear2_time - linear2_time
                linear2_time = get_sim_time(units='ns')
                print(f"linear2 handshake time: {diff_linear2}")

        if dut.stream_blocks_0_norm1_data_out_0_valid.value and dut.stream_blocks_0_norm1_data_out_0_ready.value:
            norm1_data_count += 1
            if norm1_data_count == out_depth:
                norm1_data_count = 0
                new_norm1_time = get_sim_time(units='ns')
                diff_norm1 = new_norm1_time - norm1_time
                norm1_time = get_sim_time(units='ns')
                print(f"norm1 handshake time: {diff_norm1}")

        if dut.stream_blocks_0_attention_data_out_0_valid.value and dut.stream_blocks_0_attention_data_out_0_ready.value:
            attention_data_count += 1
            if attention_data_count == out_depth:
                attention_data_count = 0
                new_attention_time = get_sim_time(units='ns')
                diff_attention = new_attention_time - attention_time
                attention_time = get_sim_time(units='ns')
                print(f"attention handshake time: {diff_attention}")

        if dut.stream_blocks_0_norm2_data_out_0_valid.value and dut.stream_blocks_0_norm2_data_out_0_ready.value:
            norm2_data_count += 1
            if norm2_data_count == out_depth:
                norm2_data_count = 0
                new_norm2_time = get_sim_time(units='ns')
                diff_norm2 = new_norm2_time - norm2_time
                norm2_time = get_sim_time(units='ns')
                print(f"norm2 handshake time: {diff_norm2}")

        if dut.stream_blocks_0_add_data_out_0_valid.value and dut.stream_blocks_0_add_data_out_0_ready.value:
            add_data_count += 1
            if add_data_count == out_depth:
                add_data_count = 0
                new_add_time = get_sim_time(units='ns')
                diff_add = new_add_time - add_time
                add_time = get_sim_time(units='ns')
                print(f"add handshake time: {diff_add}")

        if dut.stream_blocks_0_add_1_data_out_0_valid.value and dut.stream_blocks_0_add_1_data_out_0_ready.value:
            add1_data_count += 1
            if add1_data_count == out_depth:
                add1_data_count = 0
                new_add1_time = get_sim_time(units='ns')
                diff_add1 = new_add1_time - add1_time
                add1_time = get_sim_time(units='ns')
                print(f"add1 handshake time: {diff_add1}")
        



if __name__ == "__main__":
    pass_args = {
        "project_dir": Path("./mxint_vit_block"),
    }
    simulate(skip_build=False, skip_test=False, simulator="verilator", waves=True, gui=False, trace_depth=5, pass_args=pass_args)