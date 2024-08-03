import logging
import random

import cocotb
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import bit_driver, batched
from mase_cocotb.matrix_tools import gen_random_matrix_input

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)

class MatrixToVectorTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        self.assign_self_params(
            [
                "DATA_IN_0_PARALLELISM_DIM_0",
                "DATA_IN_0_PARALLELISM_DIM_1",
                "DATA_IN_0_MAX_TENSOR_SIZE_DIM_0",
                "DATA_IN_0_MAX_TENSOR_SIZE_DIM_1",
                "DATA_IN_0_PRECISION_0",
                "DATA_IN_0_PRECISION_1",
            ]
        )

        dut.counter_max.value = 4
        dut.data_in_0_depth_dim0.value = 2
        # Driver/Monitor
        self.in_driver = StreamDriver(dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready)
        self.output_monitor = StreamMonitor(
            dut.clk, dut.data_out_0, dut.data_out_0_valid, dut.data_out_0_ready
        )


    async def run_test(self, us):
        await self.reset()
        inputs = self.generate_inputs()
        #exp_out = self.model(A_inputs, B_inputs)
        # Setup drivers and monitors
        self.in_driver.load_driver(inputs)
        #self.output_monitor.load_monitor(exp_out)
        await Timer(us, units="us")
        assert self.output_monitor.exp_queue.empty()

    def generate_inputs(self):
        inputs = gen_random_matrix_input(
            self.DATA_IN_0_MAX_TENSOR_SIZE_DIM_0,
            self.DATA_IN_0_MAX_TENSOR_SIZE_DIM_1,
            self.DATA_IN_0_PARALLELISM_DIM_0,
            self.DATA_IN_0_PARALLELISM_DIM_1,
            self.DATA_IN_0_PRECISION_0,
            self.DATA_IN_0_PRECISION_1,
        )
        return inputs

def test_matrix_to_vector():
    # Default is a square matrix mult
    # 4x4 4x4 matrix multiplication done using 2x2 window
    DEFAULT_CONFIG = {
        "DATA_IN_0_PARALLELISM_DIM_0": 2,
        "DATA_IN_0_PARALLELISM_DIM_1": 2,
        "DATA_IN_0_MAX_TENSOR_SIZE_DIM_0": 4,
        "DATA_IN_0_MAX_TENSOR_SIZE_DIM_1": 4,
        "DATA_IN_0_PRECISION_0": 8,
        "DATA_IN_0_PRECISION_1": 8,
    }
    mase_runner(
        module_param_list=[DEFAULT_CONFIG],
        trace=True,
        jobs=12,
        skip_build = False
    )

@cocotb.test()
async def single_mult(dut):
    tb = MatrixToVectorTB(dut)
    #tb.output_monitor.ready.value = 1
    await tb.run_test(us=100)

if __name__ == "__main__":
    test_matrix_to_vector()