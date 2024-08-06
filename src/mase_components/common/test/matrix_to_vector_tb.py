import logging
import random
from random import choice

import cocotb
from cocotb.triggers import *

from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import bit_driver, batched
from mase_cocotb.matrix_tools import (
    gen_random_matrix_input,
    rebuild_matrix,
    split_matrix,
)

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

        self.DIM0 = choice(range(self.DATA_IN_0_PARALLELISM_DIM_0, self.DATA_IN_0_MAX_TENSOR_SIZE_DIM_0, self.DATA_IN_0_PARALLELISM_DIM_0))
        self.DIM1 = choice(range(self.DATA_IN_0_PARALLELISM_DIM_1, self.DATA_IN_0_MAX_TENSOR_SIZE_DIM_1, self.DATA_IN_0_PARALLELISM_DIM_1))
        
        #self.DIM0 = 6
        #elf.DIM1 = 6

        dut.data_in_0_depth_dim0.value = self.DIM0 // self.DATA_IN_0_PARALLELISM_DIM_0
        print(self.DIM0 // self.DATA_IN_0_PARALLELISM_DIM_0)

        # should be set to depth0 * parallelism1
        dut.counter_max.value = self.DIM0 // self.DATA_IN_0_PARALLELISM_DIM_0 * self.DATA_IN_0_PARALLELISM_DIM_1


        # Driver/Monitor
        self.in_driver = StreamDriver(dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready)
        self.output_monitor = StreamMonitor(
            dut.clk, dut.data_out_0, dut.data_out_0_valid, dut.data_out_0_ready, unsigned = True
        )

        #self.output_monitor.log.setLevel(logging.DEBUG)


    async def run_test(self, batches, us):
        await self.reset()
        #self.output_monitor.ready.value = 1
        for _ in range(batches):
            inputs = self.generate_inputs()
            exp_out = self.model(inputs)
            # Setup drivers and monitors
            self.in_driver.load_driver(inputs)
            self.output_monitor.load_monitor(exp_out)
        await Timer(us, units="us")
        assert self.output_monitor.exp_queue.empty()

    def model(self, X):
        X_matrix = rebuild_matrix(
            X, self.DIM0, self.DIM1, self.DATA_IN_0_PARALLELISM_DIM_0, self.DATA_IN_0_PARALLELISM_DIM_1
        )
        return split_matrix(
            X_matrix,
            self.DIM0,
            self.DIM1,
            self.DATA_IN_0_PARALLELISM_DIM_0,
            1
        )


    def generate_inputs(self):
        inputs = gen_random_matrix_input(
            self.DIM0,
            self.DIM1,
            self.DATA_IN_0_PARALLELISM_DIM_0,
            self.DATA_IN_0_PARALLELISM_DIM_1,
            self.DATA_IN_0_PRECISION_0,
            self.DATA_IN_0_PRECISION_1,
        )
        return inputs

def test_matrix_to_vector():
    # Default is a square matrix mult
    # 4x4 4x4 matrix multiplication done using 2x2 window
    P2X2 = {
        "DATA_IN_0_PARALLELISM_DIM_0": 2,
        "DATA_IN_0_PARALLELISM_DIM_1": 2,
        "DATA_IN_0_MAX_TENSOR_SIZE_DIM_0": 8,
        "DATA_IN_0_MAX_TENSOR_SIZE_DIM_1": 8,
        "DATA_IN_0_PRECISION_0": 8,
        "DATA_IN_0_PRECISION_1": 8,
    }
    P2X4 = {
        "DATA_IN_0_PARALLELISM_DIM_0": 2,
        "DATA_IN_0_PARALLELISM_DIM_1": 4,
        "DATA_IN_0_MAX_TENSOR_SIZE_DIM_0": 8,
        "DATA_IN_0_MAX_TENSOR_SIZE_DIM_1": 8,
        "DATA_IN_0_PRECISION_0": 8,
        "DATA_IN_0_PRECISION_1": 8,
    }
    P4X2 = {
        "DATA_IN_0_PARALLELISM_DIM_0": 4,
        "DATA_IN_0_PARALLELISM_DIM_1": 2,
        "DATA_IN_0_MAX_TENSOR_SIZE_DIM_0": 8,
        "DATA_IN_0_MAX_TENSOR_SIZE_DIM_1": 8,
        "DATA_IN_0_PRECISION_0": 8,
        "DATA_IN_0_PRECISION_1": 8,
    }
    mase_runner(
        module_param_list=[P2X2, P2X4, P4X2],
        trace=True,
        jobs=12,
        skip_build = False
    )

@cocotb.test()
async def single_mult(dut):
    tb = MatrixToVectorTB(dut)
    tb.output_monitor.ready.value = 1
    await tb.run_test(batches=1, us=100)

@cocotb.test()
async def repeated_mult(dut):
    tb = MatrixToVectorTB(dut)
    tb.output_monitor.ready.value = 1
    await tb.run_test(batches=100, us=2000)

if __name__ == "__main__":
    test_matrix_to_vector()