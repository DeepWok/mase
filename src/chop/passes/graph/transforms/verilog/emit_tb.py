from typing import Dict, Literal, Tuple
from mase_components.linear_layers.mxint_operators.test.utils import (
    block_mxint_quant,
    pack_tensor_to_mx_listed_chunk,
)
import numpy as np
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
from mase_cocotb.interfaces.streaming import (
    MultiSignalStreamDriver,
    MultiSignalStreamMonitor,
    StreamDriver,
    StreamMonitor,
)
from cocotb.result import TestFailure


import dill
import inspect


class FixedDriver(StreamDriver):
    def __init__(
        self, clk, data, valid, ready, precision, parallelism, record_num_beats=False
    ) -> None:
        super().__init__(clk, data, valid, ready, record_num_beats)
        self.precision = precision
        self.parallelism = parallelism

    def quantize_and_load(self, tensor_batches):
        from mase_cocotb.utils import fixed_preprocess_tensor

        in_data_blocks = fixed_preprocess_tensor(
            tensor=tensor_batches,
            q_config={
                "width": self.precision[0],
                "frac_width": self.precision[1],
            },
            parallelism=self.parallelism,
        )

        block_size = self.parallelism[0] * self.parallelism[1]
        for block in in_data_blocks:
            if len(block) < block_size:
                block = block + [0] * (block_size - len(block))
            self.append(block)


class MxIntDriver(MultiSignalStreamDriver):
    def __init__(self, clk, data, valid, ready, config, parallelism) -> None:
        super().__init__(clk, data, valid, ready)
        self.config = config
        self.parallelism = parallelism

    def quantize_and_load(self, tensor_batches):
        (_qtensor, mtensor, etensor) = block_mxint_quant(
            tensor_batches, self.config, self.parallelism
        )
        driver_input = pack_tensor_to_mx_listed_chunk(
            mtensor, etensor, self.parallelism
        )
        self.load_driver(driver_input)


class FixedMonitor(StreamMonitor):
    def __init__(
        self,
        clk,
        data,
        valid,
        ready,
        precision,
        parallelism,
        check=True,
        name=None,
        unsigned=False,
    ):
        super().__init__(clk, data, valid, ready, check, name, unsigned)
        self.precision = precision
        self.parallelism = parallelism

    def quantize_and_expect(self, tensor_expectation):
        from mase_cocotb.utils import fixed_preprocess_tensor

        output_blocks = fixed_preprocess_tensor(
            tensor=tensor_expectation,
            q_config={
                "width": self.precision[0],
                "frac_width": self.precision[1],
            },
            parallelism=self.parallelism,
        )

        block_size = self.parallelism[0] * self.parallelism[1]
        for block in output_blocks:
            if len(block) < block_size:
                block = block + [0] * (block_size - len(block))
            self.expect(block)
        self.in_flight = True


class MxIntMonitor(MultiSignalStreamMonitor):
    def __init__(
        self, clk, e_data, m_data, valid, ready, config, parallelism, off_by_value=0
    ):
        self.off_by = off_by_value
        self.config = config
        self.parallelism = parallelism
        super().__init__(
            clk,
            (m_data, e_data),
            valid,
            ready,
            check=True,
            signed=True,
            off_by_one=False,
        )

    def quantize_and_expect(self, tensor_expectation):
        (qtensor, mtensor, etensor) = block_mxint_quant(
            tensor_expectation, self.config, self.parallelism
        )
        tensor_output = pack_tensor_to_mx_listed_chunk(
            mtensor, etensor, self.parallelism
        )

        exp_max_val = 2 ** self.config["exponent_width"]
        for i, (tensor, exp) in enumerate(tensor_output):
            exp_signed = (2 * exp) % exp_max_val - (exp % exp_max_val)
            tensor_output[i] = (tensor, exp_signed)

        self.load_monitor(tensor_output)
        self.in_flight = True

    def _check(self, got, exp):
        got_m, got_e = got
        exp_m, exp_e = exp

        def check_equality(got, exp):
            if not np.equal(got, exp).all():
                diff = np.subtract(got, exp)
                if np.isclose(got, exp, atol=self.off_by).all():
                    self.log.warning(
                        f"Off-by-{max(abs(diff))} error: {diff=}\nGot {got}\nExp {exp}"
                    )
                else:
                    raise TestFailure(
                        "\nGot \n%s, \nExp \n%s,\nDiff \n%s" % (got, exp, diff)
                    )

        if exp_e == got_e:
            check_equality(got_m, exp_m)
        elif abs(diff := (exp_e - got_e)) == 1:
            adj_m = np.array(got_m) * 2 ** (-diff)
            self.log.warning(f"Normalisation Error {exp_e=} {got_e=}")
            check_equality(adj_m, exp_m)


def _cap(name):
    """
    capitalize a string
    """
    return str(name).upper()


def _emit_cocotb_test(graph, pass_args={}):
    wait_time = pass_args.get("wait_time", 100)
    wait_unit = pass_args.get("wait_units", "us")
    num_batches = pass_args.get("num_batches", 1)
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

    in_tensors = tb.generate_inputs(batches={num_batches})
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
        def __init__(self, dut, fail_on_checks=True):
            super().__init__(dut, dut.clk, dut.rst, fail_on_checks=fail_on_checks)

            self.input_drivers: Dict[str, FixedDriver | MxIntDriver] = {}
            self.output_monitors: Dict[str, FixedMonitor | MxIntMonitor] = {}
            for node in graph.nodes_in:
                for arg, arg_info in node.meta["mase"]["common"]["args"].items():
                    if "data_in" not in arg:
                        continue
                    match arg_info.get("type", None):
                        case "mxint":
                            config = {
                                "width": self.get_parameter(f"{_cap(arg)}_PRECISION_0"),
                                "exponent_width": self.get_parameter(
                                    f"{_cap(arg)}_PRECISION_1"
                                ),
                            }
                            parallelism = [
                                self.get_parameter(f"{_cap(arg)}_PARALLELISM_DIM_1"),
                                self.get_parameter(f"{_cap(arg)}_PARALLELISM_DIM_0"),
                            ]
                            self.input_drivers[arg] = MxIntDriver(
                                dut.clk,
                                (
                                    getattr(dut, f"m_{arg}"),
                                    getattr(dut, f"e_{arg}"),
                                ),
                                getattr(dut, f"{arg}_valid"),
                                getattr(dut, f"{arg}_ready"),
                                config,
                                parallelism,
                            )
                        case "fixed":
                            precision = [
                                self.get_parameter(f"{_cap(arg)}_PRECISION_0"),
                                self.get_parameter(f"{_cap(arg)}_PRECISION_1"),
                            ]
                            parallelism = [
                                self.get_parameter(f"{_cap(arg)}_PARALLELISM_DIM_1"),
                                self.get_parameter(f"{_cap(arg)}_PARALLELISM_DIM_0"),
                            ]
                            self.input_drivers[arg] = FixedDriver(
                                dut.clk,
                                getattr(dut, arg),
                                getattr(dut, f"{arg}_valid"),
                                getattr(dut, f"{arg}_ready"),
                                precision,
                                parallelism,
                            )
                        case t:
                            raise NotImplementedError(
                                f"Unsupported type format {t} for {node} {arg}"
                            )
                    self.input_drivers[arg].log.setLevel(logging.INFO)

            for node in graph.nodes_out:
                for result, result_info in node.meta["mase"]["common"][
                    "results"
                ].items():
                    if "data_out" not in result:
                        continue
                    match result_info.get("type", None):
                        case "mxint":
                            config = {
                                "width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
                                "exponent_width": self.get_parameter(
                                    "DATA_OUT_0_PRECISION_1"
                                ),
                            }
                            parallelism = [
                                self.get_parameter("DATA_IN_0_PARALLELISM_DIM_1"),
                                self.get_parameter("DATA_IN_0_PARALLELISM_DIM_0"),
                            ]
                            self.output_monitors[result] = MxIntMonitor(
                                dut.clk,
                                getattr(dut, f"e_{result}"),
                                getattr(dut, f"m_{result}"),
                                getattr(dut, f"{result}_valid"),
                                getattr(dut, f"{result}_ready"),
                                config,
                                parallelism,
                                off_by_value=1,
                            )
                        case "fixed":
                            precision = [
                                self.get_parameter("DATA_OUT_0_PRECISION_0"),
                                self.get_parameter("DATA_OUT_0_PRECISION_1"),
                            ]
                            parallelism = [
                                self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_1"),
                                self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_0"),
                            ]
                            self.output_monitors[result] = FixedMonitor(
                                dut.clk,
                                getattr(dut, result),
                                getattr(dut, f"{result}_valid"),
                                getattr(dut, f"{result}_ready"),
                                precision,
                                parallelism,
                                check=False,
                            )
                        case t:
                            raise NotImplementedError(
                                f"Unsupported type format {t} for {node} {result}"
                            )
                    self.output_monitors[result].log.setLevel(logging.INFO)

            self.model = graph.model
            self.input_precision = graph.meta["mase"]["common"]["args"]["data_in_0"][
                "precision"
            ]

        def generate_inputs(self, batches=1):
            inputs = {}
            for node in graph.nodes_in:
                for arg, arg_info in node.meta["mase"]["common"]["args"].items():
                    if "data_in" not in arg:
                        continue
                    print(
                        f"Generating data for node {node}, arg {arg}: {arg_info} {arg_info['shape']}"
                    )
                    inputs[f"{arg}"] = torch.randn(([batches] + arg_info["shape"]))
            return inputs

        def load_drivers(self, in_tensors):
            for arg, arg_batches in in_tensors.items():
                self.input_drivers[arg].quantize_and_load(arg_batches)

        def load_monitors(self, expectation):
            for result, monitor in self.output_monitors.items():
                monitor.quantize_and_expect(expectation)

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
