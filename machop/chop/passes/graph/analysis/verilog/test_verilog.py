import logging, toml, os, glob, math
from pathlib import Path
import torch

import cocotb
from cocotb.runner import get_runner
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock

from chop.passes.graph.utils import vf
from mase_cocotb.random_test import RandomSource, RandomSink, check_results

logger = logging.getLogger(__name__)

# =============================================================================
# DUT test specifications
# =============================================================================


def hardware_reshape(input_data, input_shape, tiling):
    """
    Apply 2D tiling. TODO: For higher dimensions, just faltten it in time.
    """

    assert len(input_shape) == 2, "Default hardware test bench only support 2D inputs"

    row_size = int(math.ceil(input_shape[0] / tiling[0]))
    col_size = int(math.ceil(input_shape[1] / tiling[1]))
    output_data = [
        [0 for _ in range(tiling[1] * tiling[0])] for _ in range(row_size * col_size)
    ]
    for i in range(row_size):
        for j in range(col_size):
            for ii in range(0, tiling[0]):
                for jj in range(0, tiling[1]):
                    rowi = i * tiling[0] + ii
                    coli = j * tiling[1] + jj
                    if rowi < input_shape[0] and coli < input_shape[1]:
                        output_data[i * row_size + j][ii * tiling[1] + jj] = int(
                            input_data[rowi][coli]
                        )

    return output_data


class VerificationCase:
    # TODO: sample > 1 needs to be added
    def __init__(self, samples=1):
        self.samples = samples

    def generate_tv(self, mg):
        """
        Generate test vector and emit to ~/.mase.{pid}.toml
        """

        # Generate random inputs
        test_inputs = {}
        # TODO: here we just enumerate the inputs of the input nodes - which may be
        # order insensitive and require manual connection when adding the graph to
        # a system.
        name_idx = 0
        for node in mg.nodes_in:
            for arg, arg_info in node.meta["mase"].parameters["common"]["args"].items():
                if "data_in" in arg:
                    test_inputs[f"data_in_{name_idx}"] = torch.randint(
                        32, arg_info["shape"]
                    )
                    name_idx += 1
        logger.debug(test_inputs)

        # Get software results
        y = mg.model(*list(test_inputs.values()))

        output_toml = {}
        output_toml["samples"] = 1

        # Reshape values for hardware testing
        # TODO: assume 2D inputs
        reshaped_inputs = {}
        name_idx = 0
        for node in mg.nodes_in:
            for arg, arg_info in node.meta["mase"].parameters["common"]["args"].items():
                if "data_in" in arg:

                    # By default: the data is passed column by column
                    reshaped_inputs[f"data_in_{name_idx}"] = hardware_reshape(
                        test_inputs[f"data_in_{name_idx}"],
                        arg_info["shape"],
                        node.meta["mase"].parameters["hardware"]["parallelism"][arg],
                    )
                    name_idx += 1

        output_toml["inputs"] = reshaped_inputs

        assert len(mg.nodes_out) == 1, "Expect the model only has one output!"
        reshaped_y = reshaped_inputs[f"data_out_0"] = hardware_reshape(
            y,
            mg.nodes_out[0]
            .meta["mase"]
            .parameters["common"]["results"]["data_out_0"]["shape"],
            mg.nodes_out[0]
            .meta["mase"]
            .parameters["hardware"]["parallelism"]["data_out_0"],
        )

        output_toml["outputs"] = {"data_out_0": reshaped_y}

        home = Path.home()
        Path(os.path.join(home, f".mase")).mkdir(parents=True, exist_ok=True)
        fname = os.path.join(home, f".mase", f"tv.toml")
        assert not os.path.isfile(
            fname
        ), f"Cannot create a temporary toml for testing data - {fname} already exists"
        with open(fname, "w") as toml_file:
            toml.dump(output_toml, toml_file)

        logger.debug(f"Test data saved to {fname}")

    def load_tv(self, fname=""):
        home = Path.home()
        fname = os.path.join(home, ".mase", f"tv.toml")
        assert os.path.isfile(
            fname
        ), f"Cannot find the temporary toml for testing data - {fname}"
        with open(fname, "r") as f:
            input_toml = toml.load(f)

        self.samples = input_toml["samples"]

        for val, values in input_toml["inputs"].items():
            setattr(
                self,
                val,
                RandomSource(
                    name=val,
                    samples=len(values),
                    num=len(values[0]),
                    max_stalls=0,
                ),
            )
            source = getattr(self, val)
            source.data = values

        for val, values in input_toml["outputs"].items():
            setattr(
                self,
                val,
                RandomSink(
                    name=val,
                    samples=len(values),
                    num=len(values[0]),
                    max_stalls=0,
                ),
            )
            self.ref = values

        os.remove(fname)
        logger.debug(f"Test data loaded from {fname}")


class TestBehavior:
    async def test_bench_behavior(dut):
        """Test top-level model hardware design (default behavior)"""
        test_case = VerificationCase()
        test_case.load_tv()

        # Reset cycle
        await Timer(20, units="ns")
        dut.rst.value = 1
        await Timer(100, units="ns")
        dut.rst.value = 0

        # Create a 10ns-period clock on port clk
        clock = Clock(dut.clk, 10, units="ns")
        # Start the clock
        cocotb.start_soon(clock.start())
        await Timer(500, units="ns")

        # Synchronize with the clock
        dut.data_in_0_valid.value = 0
        dut.data_out_0_ready.value = 1
        await FallingEdge(dut.clk)
        await FallingEdge(dut.clk)

        done = False
        # Set a timeout to avoid deadlock
        for i in range(test_case.samples * 100):
            await FallingEdge(dut.clk)

            dut.data_in_0_valid.value = test_case.data_in_0.pre_compute()
            await Timer(1, units="ns")

            dut.data_out_0_ready.value = test_case.data_out_0.pre_compute(
                dut.data_out_0_valid.value
            )
            await Timer(1, units="ns")

            dut.data_in_0_valid.value, dut.data_in_0.value = (
                test_case.data_in_0.compute(dut.data_in_0_ready.value)
            )
            await Timer(1, units="ns")

            dut.data_out_0_ready.value = test_case.data_out_0.compute(
                dut.data_out_0_valid.value, dut.data_out_0.value
            )

            if test_case.data_in_0.is_empty() and test_case.data_out_0.is_full():
                done = True
                break
        assert (
            done
        ), "Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)"

        check_results(test_case.data_out_0.data, test_case.ref)


# =============================================================================
# Cocotb interface setup
# =============================================================================


@cocotb.test()
async def test_top(dut):
    await TestBehavior.test_bench_behavior(dut)


def get_dut_parameters(graph):
    parameter_map = {}

    for node in graph.fx_graph.nodes:
        if node.meta["mase"].parameters["hardware"]["is_implicit"]:
            continue
        node_name = vf(node.name)

        for key, value in (
            node.meta["mase"].parameters["hardware"]["verilog_param"].items()
        ):
            if not isinstance(value, (int, float, complex, bool)):
                value = '"' + value + '"'
            assert (
                f"{node_name}_{key}" not in parameter_map.keys()
            ), f"{node_name}_{key} already exists in the parameter map"
            parameter_map[f"{node_name}_{key}"] = value
    return parameter_map


def runner(mg, project_dir, top_name):
    sim = os.getenv("SIM", "verilator")

    # TODO: Grab internal verilog source only. Need to include HLS hardware as well.
    sv_srcs = []
    for v in glob.glob(os.path.join(project_dir, "hardware", "rtl", "*.sv")):
        sv_srcs.append(os.path.relpath(v, os.getcwd()))

    p = get_dut_parameters(mg)
    # logger.debug(p)

    # set parameters
    extra_args = []
    for k, v in p.items():
        extra_args.append(f"-G{k}={v}")
    logger.debug(extra_args)
    runner = get_runner(sim)
    runner.build(
        verilog_sources=sv_srcs,
        hdl_toplevel=top_name,
        build_args=extra_args,
    )

    runner.test(
        hdl_toplevel=top_name,
        test_module=f"chop.passes.graph.analysis.verilog.test_verilog",
    )


def test_verilog_analysis_pass(mg, pass_args={}):
    """Test the top-level hardware design using Cocotb

    :param graph: a MaseGraph
    :type graph: MaseGraph
    :param pass_args: this pass requires additional arguments which is explained below, defaults to {}
    :type pass_args: _type_, optional
    :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
    :rtype: tuple(MaseGraph, Dict)

    - pass_args
        - project_dir -> str : the directory of the project for cosimulation
        - top_name -> str : top-level name
        - samples -> str : the number of test inputs, samples = 1 by default
        - test_bench -> str : the test bench behavior specified by the user, which runs end-to-end simulation by default
        - preprocess -> str : preprocess of IO for testing, which generates random inputs by default
    """

    logger.info(f"Running hardware simulation using Cocotb")
    logger.debug(f"test verilog pass pass_args = {pass_args}")

    project_dir = (
        pass_args["project_dir"]
        if "project_dir" in pass_args.keys()
        else Path.home() / ".mase" / "top"
    )
    top_name = pass_args["top_name"] if "top_name" in pass_args.keys() else "top"
    samples = pass_args["samples"] if "samples" in pass_args.keys() else 1

    # TODO: Create a global variable traced by pass ID. This is bad...
    test_case = VerificationCase(samples)
    globals()["test_verilog_analysis_pass_tc"] = test_case
    print(globals())

    if "preprocess" in pass_args.keys():
        test_case.preprocess = pass_args["preprocess"]
    if "test_bench" in pass_args.keys():
        test_case.test_bench_behavior = pass_args["test_bench"]

    test_case.generate_tv(mg)

    runner(mg, project_dir, top_name)
    return mg, {}
