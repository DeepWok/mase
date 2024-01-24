import math, time, os, logging, torch, glob, shutil

from chop.passes.graph.utils import vf, v2p, init_project
from chop.passes.graph.transforms.quantize.quantizers import integer_quantizer_for_hw

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
    tb_path = Path.home() / ".mase" / "top" / "hardware" / "test" / "mase_top_tb"
    with open(tb_path / "tb_obj.dill", "rb") as f:
        tb = dill.load(f)(dut)

    await tb.reset()
    tb.output_monitor.ready.value = 1
    inputs = tb.generate_inputs()
    exp_out = tb.model(inputs)

    # To do: replace with tb.load_drivers(inputs)
    for i in inputs[0]:
        tb.data_in_0_driver.append(i)

    # To do: replace with tb.load_monitors(exp_out)
    for out in exp_out:
        tb.data_out_0_monitor.expect(out)

    # To do: replace with tb.run()
    await Timer(100, units="us")
    # To do: replace with tb.monitors_done() --> for monitor, call monitor_done()
    assert tb.data_out_0_monitor.exp_queue.empty()


def _emit_cocotb_test(graph):
    test_template = f"""
import cocotb
from pathlib import Path
import dill

{inspect.getsource(test)}
"""

    tb_path = Path.home() / ".mase" / "top" / "hardware" / "test" / "mase_top_tb"
    tb_path.mkdir(parents=True, exist_ok=True)
    with open(tb_path / "test.py", "w") as f:
        f.write(test_template)


def _emit_cocotb_tb(graph):
    class ModelTB(Testbench):
        def __init__(self, dut):
            super().__init__(dut, dut.clk, dut.rst)
            # Assign module parameters from parameter map
            # self.assign_self_params([])

            # Instantiate as many drivers as required inputs
            self.input_driver = [
                StreamDriver(
                    dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
                )
            ]

            # Instantiate as many monitors as required outputs
            self.output_monitor = [
                StreamMonitor(
                    dut.clk, dut.data_out_0, dut.data_out_0_valid, dut.data_out_0_ready
                )
            ]

        def get_random_tensor(self, shape):
            # Generate random tensors of the appropriate shape
            return torch.rand(shape)

        def generate_inputs(self):
            # Generate inputs for as many streaming interfaces as required

            # For every input to the model, generate random tensor according to its shape
            inputs = []
            inputs += self.torch.rand((1, 4))

            # Quantize each input tensor to required precision
            quantized_inputs = []
            quantized_inputs += quantize_to_int(data_in_0_inputs)

            return inputs, quantized_inputs

        def model(self, inputs):
            # Run the model with the provided inputs and return the outputs
            out = graph.model(inputs)
            return out

    # Serialize testbench object to be instantiated within test by cocotb runner
    cls_obj = ModelTB
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
