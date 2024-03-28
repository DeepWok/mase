#!/usr/bin/env python3


# This script tests the fixed point linear

# Manually add mase_cocotb to system path
import sys, os
try:
    p = os.getenv("MASE_RTL")
    assert p != None
except:
    p = os.getenv("mase_rtl")
    assert p != None
p = os.path.join(p, '../')
sys.path.append(p)
###############################################
import os, math, logging

from mase_cocotb.random_test import *
from mase_cocotb.runner import mase_runner

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock

debug = True

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)






#!/usr/bin/env python3

# This script tests the fixed point linear
import os, math, logging
import sys
sys.path.append('/home/ic/TEMP/mase/machop/')
from mase_cocotb.random_test import RandomSource, RandomSink, check_results
from mase_cocotb.runner import mase_runner

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock

debug = True

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=10):
        self.data_in_width = 8
        self.data_in_frac_width = 0
        self.weight_width = 8
        self.weight_frac_width = 0
        self.bias_width = 8
        self.bias_frac_width = 0
        self.data_out_width = 18
        self.data_out_frac_width = 0
        self.has_bias = 0

        self.in_rows = 20
        self.in_columns = 4
        self.weight_rows = self.in_columns
        self.weight_columns = 1
        self.iterations = 3
        self.data_in = RandomSource(
            name="data_in",
            samples=samples * self.iterations,
            num=self.in_rows * self.in_columns,
            max_stalls=0,
            debug=debug,
        )
        self.weight = RandomSource(
            name="weight",
            samples=samples * self.iterations,
            num=self.weight_rows * self.weight_columns,
            max_stalls=0,
            debug=debug,
        )
        self.bias = RandomSource(
            name="bias",
            samples=samples,
            num=self.in_rows * self.weight_columns,
            max_stalls=0,
            debug=debug,
        )
        self.outputs = RandomSink(samples=samples, max_stalls=0, debug=debug)
        self.samples = samples
        self.ref = self.sw_compute()
        # self.ref = self.sw_cast(
        #     inputs=self.ref,
        #     in_width=self.data_in_width
        #     + self.weight_width
        #     + math.ceil(math.log2(self.iterations * self.in_columns))
        #     + self.has_bias,
        #     in_frac_width=self.data_in_frac_width + self.weight_frac_width,
        #     out_width=self.data_out_width,
        #     out_frac_width=self.data_out_frac_width,
        # )

    def get_dut_parameters(self):
        return {
            "IN_WIDTH": self.data_in_width,
            "IN_SIZE": self.in_columns,
            "IN_PARALLELISM": self.in_rows,
            "WEIGHT_PARALLELISM": self.weight_columns,
            "HAS_BIAS": self.has_bias,
            "IN_DEPTH": self.iterations,
            "OUT_WIDTH": self.data_out_width
        }

    def sw_compute(self):
        final = []
        ref = []
        for i in range(self.samples):
            acc = [0 for _ in range(self.in_rows * self.weight_columns)]
            for w in range(self.in_rows):
                for j in range(self.iterations):
                    data_idx = i * self.iterations + j
                    for k in range(self.weight_columns):
                        s = [
                            self.data_in.data[data_idx][w * self.in_columns + h]
                            * self.weight.data[data_idx][k * self.weight_rows + h]
                            for h in range(self.weight_rows)
                        ]
                        acc[w * self.weight_columns + k] += sum(s)
            if self.has_bias:
                for j in range(self.in_rows * self.weight_columns):
                    acc[j] += self.bias.data[i][j] << (
                        self.weight_frac_width
                        + self.data_in_frac_width
                        - self.bias_frac_width
                    )
            ref.append(acc)
        ref.reverse()
        return ref

    def sw_cast(self, inputs, in_width, in_frac_width, out_width, out_frac_width):
        outputs = []
        for j in range(len(inputs)):
            in_list = inputs[j]
            out_list = []
            for i in range(0, len(in_list)):
                in_value = in_list[i]
                if in_frac_width > out_frac_width:
                    in_value = in_value >> (in_frac_width - out_frac_width)
                else:
                    in_value = in_value << (out_frac_width - in_frac_width)
                in_int_width = in_width - in_frac_width
                out_int_width = out_width - out_frac_width
                if in_int_width > out_int_width:
                    if in_value >> (in_frac_width + out_int_width) > 0:
                        in_value = 1 << out_width - 1
                    elif in_value >> (in_frac_width + out_int_width) < 0:
                        in_value = -(1 << out_width - 1)
                    else:
                        in_value = int(in_value % (1 << out_width))
                out_list.append(in_value)
            outputs.append(out_list)
        return outputs


def debug_state(dut, state):
    logger.debug(
        "{} State: (w_ready,w_valid,in_ready,in_valid,out_ready,out_valid) = ({},{},{},{},{},{})".format(
            state,
            dut.data_in_ready.value,
            dut.weight_valid.value,
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )
    )


@cocotb.test()
async def test_llm_int8_tb(dut):
    """Test integer based vector mult"""
    samples = 100
    test_case = VerificationCase(samples=samples)

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
    dut.weight_valid.value = 0
    dut.data_in_valid.value = 0
    dut.data_out_ready.value = 1
    debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    debug_state(dut, "Post-clk")
    debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    debug_state(dut, "Post-clk")

    done = False
    # Set a timeout to avoid deadlock
    for i in range(samples * 50):
        await FallingEdge(dut.clk)
        debug_state(dut, "Post-clk")
        dut.bias_valid.value = test_case.bias.pre_compute()
        dut.weight_valid.value = test_case.weight.pre_compute()
        dut.data_in_valid.value = test_case.data_in.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(
            dut.data_out_valid.value
        )
        debug_state(dut, "Pre-clk")
        await Timer(1, units="ns")
        debug_state(dut, "Post-clk")
        # dut.bias_valid.value, dut.bias.value = test_case.bias.compute(
        #     dut.bias_ready.value
        # )
        dut.weight_valid.value, dut.weight.value = test_case.weight.compute(
            dut.weight_ready.value
        )
        dut.data_in_valid.value, dut.data_in.value = test_case.data_in.compute(
            dut.data_in_ready.value
        )
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )
        # breakpoint()
        debug_state(dut, "Pre-clk")
        if (
            # test_case.bias.is_empty()  // TODO
            test_case.weight.is_empty()
            and test_case.data_in.is_empty()
            and test_case.outputs.is_full()
        ):
            done = True
            break
    assert (
        done
    ), "Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)"

    # check_results_signed(test_case.outputs.data, test_case.ref)
    analyse_results_signed(test_case.outputs.data, test_case.ref, 10)


if __name__ == "__main__":
    tb = VerificationCase()
    mase_runner(
        module_param_list=[tb.get_dut_parameters()],
        extra_build_args=["--unroll-count", "3000"],
    )

