import logging, torch
from pathlib import Path
from textwrap import indent

from chop.passes.graph.utils import vf, v2p, init_project
from chop.nn.quantizers import (
    integer_quantizer_for_hw,
    integer_quantizer,
)

logger = logging.getLogger(__name__)

from pathlib import Path

torch.manual_seed(0)

import cocotb
from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor


import dill
import inspect


def _cap(name):
    """
    capitalize a string
    """
    return str(name).upper()


def _emit_cocotb_test(graph, pass_args={}):

    wait_time = pass_args.get("wait_time", 2)
    wait_unit = pass_args.get("wait_units", "ms")
    batch_size = pass_args.get("batch_size", 1)

    test_template = f"""
import cocotb

@cocotb.test()
async def test(dut):
    from pathlib import Path
    import dill
    from cocotb.triggers import Timer

    tb_path = Path.home() / ".mase" / "top" / "hardware" / "test" / "mase_top_tb"
    with open(tb_path / "tb_obj.dill", "rb") as f:
        tb = dill.load(f)(dut, fail_on_checks=True)

    await tb.initialize()

    in_tensors = tb.generate_inputs(batches={batch_size})
    exp_out = tb.model(*list(in_tensors.values()))

    tb.load_drivers(in_tensors)
    tb.load_monitors(exp_out)

    await tb.wait_end(timeout={wait_time}, timeout_unit="{wait_unit}")
"""

    tb_path = Path.home() / ".mase" / "top" / "hardware" / "test" / "mase_top_tb"
    tb_path.mkdir(parents=True, exist_ok=True)
    with open(tb_path / "test.py", "w") as f:
        f.write(test_template)


def _emit_cocotb_tb(graph):
    class MaseGraphTB(Testbench):
        def _iter_dram_parameter_specs(self):
            """Yield (node, node_name, arg, arg_info) for DRAM-backed tensor params."""
            for node in graph.fx_graph.nodes:
                if not hasattr(node, "meta") or "mase" not in node.meta:
                    continue

                node_params = node.meta["mase"].parameters
                hardware = node_params.get("hardware", {})
                if hardware.get("is_implicit", False):
                    continue

                args = node_params.get("common", {}).get("args", {})
                interfaces = hardware.get("interface", {})
                for arg, arg_info in args.items():
                    if "data_in" in arg or not isinstance(arg_info, dict):
                        continue
                    if interfaces.get(arg, {}).get("storage") != "DRAM":
                        continue
                    yield node, vf(node.name), arg, arg_info

        def __init__(self, dut, fail_on_checks=True):
            super().__init__(dut, dut.clk, dut.rst, fail_on_checks=fail_on_checks)

            # Instantiate as many drivers as required inputs to the model
            self.input_drivers = {}
            self.output_monitors = {}

            for node in graph.nodes_in:
                for arg in node.meta["mase"]["common"]["args"].keys():
                    if "data_in" not in arg:
                        continue
                    self.input_drivers[arg] = StreamDriver(
                        dut.clk,
                        getattr(dut, arg),
                        getattr(dut, f"{arg}_valid"),
                        getattr(dut, f"{arg}_ready"),
                    )
                    self.input_drivers[arg].log.setLevel(logging.DEBUG)

            # Instantiate as many monitors as required outputs
            for node in graph.nodes_out:
                for result in node.meta["mase"]["common"]["results"].keys():
                    if "data_out" not in result:
                        continue
                    self.output_monitors[result] = StreamMonitor(
                        dut.clk,
                        getattr(dut, result),
                        getattr(dut, f"{result}_valid"),
                        getattr(dut, f"{result}_ready"),
                        check=False,
                    )
                    self.output_monitors[result].log.setLevel(logging.DEBUG)

            self.model = graph.model

            # To do: precision per input argument
            self.input_precision = graph.meta["mase"]["common"]["args"]["data_in_0"][
                "precision"
            ]

            # Create StreamDriver instances for DRAM-backed weight/bias parameters.
            self.dram_drivers = {}
            for _node, node_name, arg, _arg_info in self._iter_dram_parameter_specs():
                port_name = f"{node_name}_{arg}"
                try:
                    self.dram_drivers[port_name] = StreamDriver(
                        dut.clk,
                        getattr(dut, port_name),
                        getattr(dut, f"{port_name}_valid"),
                        getattr(dut, f"{port_name}_ready"),
                    )
                    self.dram_drivers[port_name].log.setLevel(logging.DEBUG)
                except AttributeError:
                    logger.warning(
                        "Skipping DRAM port '%s': signal/handshake not found on DUT",
                        port_name,
                    )

            logger.debug("Discovered %d DRAM parameter ports", len(self.dram_drivers))
            if self.dram_drivers:
                logger.info(
                    "DRAM drivers enabled for ports: %s",
                    ", ".join(sorted(self.dram_drivers.keys())),
                )
            else:
                logger.info("No DRAM parameter ports discovered for this testbench")

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
            inputs = {}
            for node in graph.nodes_in:
                for arg, arg_info in node.meta["mase"]["common"]["args"].items():
                    # Batch dimension always set to 1 in metadata
                    if "data_in" not in arg:
                        continue
                    # print(f"Generating data for node {node}, arg {arg}: {arg_info}")
                    inputs[f"{arg}"] = torch.rand(([batches] + arg_info["shape"][1:]))
            return inputs

        def load_drivers(self, in_tensors):
            logger.info("Loading input drivers for %d tensor inputs", len(in_tensors))
            for arg, arg_batches in in_tensors.items():
                # DiffLogic: do not need precision, fully unrolled so 1 batch only
                if "difflogic" in graph.nodes_in[0].meta["mase"]["hardware"]["module"]:
                    block = arg_batches[0].round().int().tolist()
                    if isinstance(block[0], list):
                        out = []
                        for row in block:
                            num = ""
                            for i in range(len(row) - 1, -1, -1):
                                num += str(row[i])
                            num = int(num, 2)
                            out.append(num)
                    else:
                        num = ""
                        for i in range(len(block) - 1, -1, -1):
                            num += str(block[i])
                        num = int(num, 2)
                        out = [num]
                    self.input_drivers[arg].append(out)
                    continue

                # Quantize input tensor according to precision
                if len(self.input_precision) > 1:
                    from mase_cocotb.utils import fixed_preprocess_tensor

                    in_data_blocks = fixed_preprocess_tensor(
                        tensor=arg_batches,
                        q_config={
                            "width": self.get_parameter(f"{_cap(arg)}_PRECISION_0"),
                            "frac_width": self.get_parameter(
                                f"{_cap(arg)}_PRECISION_1"
                            ),
                        },
                        parallelism=[
                            self.get_parameter(f"{_cap(arg)}_PARALLELISM_DIM_1"),
                            self.get_parameter(f"{_cap(arg)}_PARALLELISM_DIM_0"),
                        ],
                    )

                else:
                    # TO DO: convert to integer equivalent of floating point representation
                    pass

                # Append all input blocks to input driver
                # ! TO DO: generalize
                block_size = self.get_parameter(
                    "DATA_IN_0_PARALLELISM_DIM_0"
                ) * self.get_parameter("DATA_IN_0_PARALLELISM_DIM_1")
                for block in in_data_blocks:
                    if len(block) < block_size:
                        block = block + [0] * (block_size - len(block))
                    self.input_drivers[arg].append(block)

            # Preload DRAM parameter drivers with quantized parameter blocks.
            if self.dram_drivers:
                from mase_cocotb.utils import fixed_preprocess_tensor

                logger.info("Preloading DRAM parameter streams for %d ports", len(self.dram_drivers))
                total_blocks_queued = 0

                for node, node_name, arg, _arg_info in self._iter_dram_parameter_specs():
                    port_name = f"{node_name}_{arg}"
                    if port_name not in self.dram_drivers:
                        continue

                    module = getattr(node.meta["mase"], "module", None)
                    if module is None:
                        logger.warning(
                            "Skipping DRAM parameter '%s': node module is unavailable",
                            port_name,
                        )
                        continue

                    try:
                        param_tensor = module.get_parameter(arg)
                    except Exception as exc:
                        logger.warning(
                            "Skipping DRAM parameter '%s': cannot fetch parameter '%s' (%s)",
                            port_name,
                            arg,
                            exc,
                        )
                        continue

                    arg_cap = _cap(arg)
                    try:
                        parallelism_0 = self.get_parameter(
                            f"{node_name}_{arg_cap}_PARALLELISM_DIM_0"
                        )
                        parallelism_1 = self.get_parameter(
                            f"{node_name}_{arg_cap}_PARALLELISM_DIM_1"
                        )
                        width = self.get_parameter(f"{node_name}_{arg_cap}_PRECISION_0")
                        frac_width = self.get_parameter(
                            f"{node_name}_{arg_cap}_PRECISION_1"
                        )
                    except Exception as exc:
                        logger.warning(
                            "Skipping DRAM parameter '%s': missing DUT parameter metadata (%s)",
                            port_name,
                            exc,
                        )
                        continue

                    param_blocks = fixed_preprocess_tensor(
                        tensor=param_tensor,
                        q_config={
                            "width": width,
                            "frac_width": frac_width,
                        },
                        parallelism=[parallelism_1, parallelism_0],
                    )
                    block_size = parallelism_0 * parallelism_1
                    port_blocks_queued = 0
                    for block in param_blocks:
                        if len(block) < block_size:
                            block = block + [0] * (block_size - len(block))
                        self.dram_drivers[port_name].append(block)
                        port_blocks_queued += 1

                    total_blocks_queued += port_blocks_queued
                    logger.info(
                        "Queued %d DRAM blocks for port '%s' (block_size=%d)",
                        port_blocks_queued,
                        port_name,
                        block_size,
                    )

                logger.info("Queued %d DRAM parameter blocks in total", total_blocks_queued)
            else:
                logger.info("No DRAM parameter streams to preload")

        def load_monitors(self, expectation):
            # DiffLogic: do not need precision, fully unrolled so 1 batch only
            if "difflogic" in graph.nodes_out[0].meta["mase"]["hardware"]["module"]:
                self.output_monitors["data_out_0"].expect(expectation)
                self.output_monitors["data_out_0"].in_flight = True
                return

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
        - trace -> bool : trace waves in the simulation
    """
    logger.info("Emitting testbench...")
    project_dir = (
        pass_args["project_dir"]
        if "project_dir" in pass_args.keys()
        else Path.home() / ".mase" / "top"
    )

    init_project(project_dir)

    _emit_cocotb_test(graph, pass_args=pass_args)
    _emit_cocotb_tb(graph)

    return graph, None
