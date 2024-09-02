import logging
import random
from random import choice
from pathlib import Path

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

from mase_cocotb.utils import (
    bit_driver,
    batched,
    sign_extend_t,
    verilator_str_param,
)

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


class swintopTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        #self.assign_self_params()

        # Driver/Monitor
        self.mha_pl0_0_data_in_0_driver = StreamDriver(dut.clk, dut.input_buffer_data_in_0, dut.input_buffer_data_in_0_valid, dut.input_buffer_data_in_0_ready)

        self.mha_pl0_0_weight_query_driver = StreamDriver(
            dut.clk, dut.mha_pl0_0_weight_query, dut.mha_pl0_0_weight_query_valid, dut.mha_pl0_0_weight_query_ready
        )
        self.mha_pl0_0_weight_key_driver = StreamDriver(
            dut.clk, dut.mha_pl0_0_weight_key, dut.mha_pl0_0_weight_key_valid, dut.mha_pl0_0_weight_key_ready
        )
        self.mha_pl0_0_weight_value_driver = StreamDriver(
            dut.clk, dut.mha_pl0_0_weight_value, dut.mha_pl0_0_weight_value_valid, dut.mha_pl0_0_weight_value_ready
        )
        self.mha_pl0_0_weight_out_driver = StreamDriver(
            dut.clk, dut.mha_pl0_0_weight_out, dut.mha_pl0_0_weight_out_valid, dut.mha_pl0_0_weight_out_ready
        )

        self.mha_pl0_0_bias_con_driver = StreamDriver(
            dut.clk, dut.mha_pl0_0_bias_con, dut.mha_pl0_0_bias_con_valid, dut.mha_pl0_0_bias_con_ready
        )

        self.mha_pl0_0_bias_pos_driver = StreamDriver(
            dut.clk, dut.mha_pl0_0_bias_pos, dut.mha_pl0_0_bias_pos_valid, dut.mha_pl0_0_bias_pos_ready
        )

        self.mha_pl0_0_bias_out_driver = StreamDriver(
            dut.clk, dut.mha_pl0_0_bias_out, dut.mha_pl0_0_bias_out_valid, dut.mha_pl0_0_bias_out_ready
        )

        self.mha_pl0_0_rel_k_driver = StreamDriver(
            dut.clk, dut.mha_pl0_0_pos_embed, dut.mha_pl0_0_pos_embed_valid, dut.mha_pl0_0_pos_embed_ready
        )

        self.linear_pl0_0_weight_driver = StreamDriver(
            dut.clk, dut.linear_pl0_0_weight, dut.linear_pl0_0_weight_valid, dut.linear_pl0_0_weight_ready
        )
 
        self.linear_pl0_0_bias_driver = StreamDriver(
            dut.clk, dut.linear_pl0_0_bias, dut.linear_pl0_0_bias_valid, dut.linear_pl0_0_bias_ready
        )

        self.linear_pl0_1_weight_driver = StreamDriver(
            dut.clk, dut.linear_pl0_1_weight, dut.linear_pl0_1_weight_valid, dut.linear_pl0_1_weight_ready
        )

        self.linear_pl0_1_bias_driver = StreamDriver(
            dut.clk, dut.linear_pl0_1_bias, dut.linear_pl0_1_bias_valid, dut.linear_pl0_1_bias_ready
        )


    async def run_test(self, batches, us):
        await self.reset()
        #self.output_monitor.ready.value = 1
        for _ in range(batches):
            inputs = self.generate_inputs()
            weights = self.generate_weights()
            pos_embed = self.generate_pos_embed()
            self.mha_pl0_0_data_in_0_driver.load_driver(inputs)
            self.mha_pl0_0_weight_query_driver.load_driver(weights)
            self.mha_pl0_0_weight_key_driver.load_driver(weights)
            self.mha_pl0_0_weight_value_driver.load_driver(weights)
            self.mha_pl0_0_weight_out_driver.load_driver(weights)
            self.mha_pl0_0_bias_pos_driver.load_driver(weights)
            self.mha_pl0_0_bias_con_driver.load_driver(weights)
            self.mha_pl0_0_bias_out_driver.load_driver(weights)
            self.mha_pl0_0_rel_k_driver.load_driver(pos_embed)
            self.linear_pl0_0_weight_driver.load_driver(weights)
            self.linear_pl0_0_bias_driver.load_driver(weights)
            self.linear_pl0_1_weight_driver.load_driver(weights)
            self.linear_pl0_1_bias_driver.load_driver(weights)

        await Timer(us, units="us")

    def model(self, X):
        return 
        


    def generate_inputs(self):
        inputs = gen_random_matrix_input(
            4,
            4,
            self.get_parameter("PARALLELISM_DIM0"),
            self.get_parameter("PARALLELISM_DIM1"),
            self.get_parameter("PRECISION_0"),
            self.get_parameter("PRECISION_1"),
        )
        return inputs

    def generate_weights(self):
        inputs = gen_random_matrix_input(
            16,
            16,
            self.get_parameter("PARALLELISM_DIM0"),
            self.get_parameter("PARALLELISM_DIM1"),
            self.get_parameter("PRECISION_0"),
            self.get_parameter("PRECISION_1"),
        )
        return inputs


    def generate_pos_embed(self):
        inputs = gen_random_matrix_input(
            256,
            256,
            self.get_parameter("PARALLELISM_DIM0"),
            self.get_parameter("PARALLELISM_DIM1") * self.get_parameter("PARALLELISM_DIM1"),
            self.get_parameter("PRECISION_0"),
            self.get_parameter("PRECISION_1"),
        )
        return inputs

def test_swin():
    mase_runner(
        # module_param_list={
        #     #"ISQRT_LUT_MEMFILE": verilator_str_param(str(Path(__file__).parent / "build" / "swintop_simple" / "mem"/ f"lutmem-default.mem"))
        #     #"ISQRT_LUT_MEMFILE": verilator_str_param("/workspace/src/mase_components/attention/test/build/swintop_simple/mem/lutmem-default.mem")
        # },
        trace=True,
        jobs=12,
        skip_build = False
    )

@cocotb.test()
async def single_test(dut):
    tb = swintopTB(dut)
    #tb.output_monitor.ready.value = 1
    await tb.run_test(batches=1, us=100)

    
if __name__ == "__main__":
    test_swin()
