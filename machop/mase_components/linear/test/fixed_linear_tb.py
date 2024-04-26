#!/usr/bin/env python3

# This script tests the fixed point linear
import os, logging

from mase_cocotb.random_test import RandomSource, RandomSink, check_results
from mase_cocotb.runner import mase_runner

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock

debug = False

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=10):
        self.has_bias = 1

        self.data_in_0_precision_0 = 8
        self.data_in_0_precision_1 = 3
        self.data_in_0_tensor_size_dim_0 = 1
        self.data_in_0_parallelism_dim_0 = 1
        self.data_in_0_tensor_size_dim_1 = 8
        self.data_in_0_parallelism_dim_1 = 8
        self.data_in_0_tensor_size_dim_2 = 1
        self.data_in_0_parallelism_dim_2 = 1

        self.weight_precision_0 = 8
        self.weight_precision_1 = 3
        self.weight_tensor_size_dim_0 = 10
        self.weight_parallelism_dim_0 = 10
        self.weight_tensor_size_dim_1 = 8
        self.weight_parallelism_dim_1 = 8
        self.weight_tensor_size_dim_2 = 1
        self.weight_parallelism_dim_2 = 1

        self.data_out_0_precision_0 = 32
        self.data_out_0_precision_1 = 16
        self.data_out_0_tensor_size_dim_0 = 1
        self.data_out_0_parallelism_dim_0 = 1
        self.data_out_0_tensor_size_dim_1 = 10
        self.data_out_0_parallelism_dim_1 = 10
        self.data_out_0_tensor_size_dim_2 = 1

        self.bias_precision_0 = 8
        self.bias_precision_1 = 3
        self.bias_tensor_size_dim_0 = 10
        self.bias_parallelism_dim_0 = 10
        self.bias_tensor_size_dim_1 = 1
        self.bias_parallelism_dim_1 = 1
        self.bias_tensor_size_dim_2 = 1
        self.bias_parallelism_dim_2 = 1

        self.data_in = RandomSource(
            name="data_in",
            samples=samples
            * self.data_in_0_tensor_size_dim_1
            // self.data_in_0_parallelism_dim_1,
            num=self.data_in_0_parallelism_dim_1,
            max_stalls=0,
            debug=debug,
        )
        self.weight = RandomSource(
            name="weight",
            samples=samples
            * self.weight_tensor_size_dim_1
            // self.weight_parallelism_dim_1,
            num=self.weight_parallelism_dim_0 * self.weight_parallelism_dim_1,
            max_stalls=0,
            debug=debug,
        )
        self.bias = RandomSource(
            name="bias",
            samples=samples,
            num=self.bias_parallelism_dim_0,
            max_stalls=0,
            debug=debug,
        )
        self.outputs = RandomSink(samples=samples, max_stalls=0, debug=debug)
        self.samples = samples
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "HAS_BIAS": self.has_bias,
            "DATA_IN_0_PRECISION_0": self.data_in_0_precision_0,
            "DATA_IN_0_PRECISION_1": self.data_in_0_precision_1,
            "DATA_IN_0_TENSOR_SIZE_DIM_0": self.data_in_0_tensor_size_dim_0,
            "DATA_IN_0_PARALLELISM_DIM_0": self.data_in_0_parallelism_dim_0,
            "DATA_IN_0_TENSOR_SIZE_DIM_1": self.data_in_0_tensor_size_dim_1,
            "DATA_IN_0_PARALLELISM_DIM_1": self.data_in_0_parallelism_dim_1,
            "DATA_IN_0_TENSOR_SIZE_DIM_2": self.data_in_0_tensor_size_dim_2,
            "DATA_IN_0_PARALLELISM_DIM_2": self.data_in_0_parallelism_dim_2,
            "WEIGHT_PRECISION_0": self.weight_precision_0,
            "WEIGHT_PRECISION_1": self.weight_precision_1,
            "WEIGHT_TENSOR_SIZE_DIM_0": self.weight_tensor_size_dim_0,
            "WEIGHT_PARALLELISM_DIM_0": self.weight_parallelism_dim_0,
            "WEIGHT_TENSOR_SIZE_DIM_1": self.weight_tensor_size_dim_1,
            "WEIGHT_PARALLELISM_DIM_1": self.weight_parallelism_dim_1,
            "WEIGHT_TENSOR_SIZE_DIM_2": self.weight_tensor_size_dim_2,
            "WEIGHT_PARALLELISM_DIM_2": self.weight_parallelism_dim_2,
            "DATA_OUT_0_PRECISION_0": self.data_out_0_precision_0,
            "DATA_OUT_0_PRECISION_1": self.data_out_0_precision_1,
            "DATA_OUT_0_TENSOR_SIZE_DIM_0": self.data_out_0_tensor_size_dim_0,
            "DATA_OUT_0_PARALLELISM_DIM_0": self.data_out_0_parallelism_dim_0,
            "DATA_OUT_0_TENSOR_SIZE_DIM_1": self.data_out_0_tensor_size_dim_1,
            "DATA_OUT_0_PARALLELISM_DIM_1": self.data_out_0_parallelism_dim_1,
            "DATA_OUT_0_TENSOR_SIZE_DIM_2": self.data_out_0_tensor_size_dim_2,
            "BIAS_PRECISION_0": self.bias_precision_0,
            "BIAS_PRECISION_1": self.bias_precision_1,
            "BIAS_TENSOR_SIZE_DIM_0": self.bias_tensor_size_dim_0,
            "BIAS_PARALLELISM_DIM_0": self.bias_parallelism_dim_0,
            "BIAS_TENSOR_SIZE_DIM_1": self.bias_tensor_size_dim_1,
            "BIAS_PARALLELISM_DIM_1": self.bias_parallelism_dim_1,
            "BIAS_TENSOR_SIZE_DIM_2": self.bias_tensor_size_dim_2,
            "BIAS_PARALLELISM_DIM_2": self.bias_parallelism_dim_2,
        }

    def sw_compute(self):
        ref = []
        for i in range(self.samples):
            acc = [0 for _ in range(self.data_out_0_parallelism_dim_1)]
            for j in range(
                self.data_in_0_tensor_size_dim_1 // self.data_in_0_parallelism_dim_1
            ):
                data_idx = (
                    i
                    * self.data_in_0_tensor_size_dim_1
                    // self.data_in_0_parallelism_dim_1
                    + j
                )
                temp = []
                for k in range(self.data_out_0_parallelism_dim_1):
                    s = [
                        self.data_in.data[data_idx][h]
                        * self.weight.data[data_idx][
                            k * self.data_in_0_parallelism_dim_1 + h
                        ]
                        for h in range(self.data_in_0_parallelism_dim_1)
                    ]
                    acc[k] += sum(s)
            if self.has_bias:
                for k in range(self.bias_parallelism_dim_0):
                    acc[k] += self.bias.data[i][k] << (
                        self.weight_precision_1
                        + self.data_in_0_precision_1
                        - self.bias_precision_1
                    )
            ref.append(acc)
        ref.reverse()
        return ref


def debug_state(dut, state):
    logger.debug(
        "{} State: (bias_ready,bias_valid,bias_ready,bias_valid,data_in_ready,data_in_valid,data_out_ready,data_out_valid) = ({},{},{},{},{},{})".format(
            state,
            dut.bias_ready.value,
            dut.bias_valid.value,
            dut.weight_ready.value,
            dut.weight_valid.value,
            dut.data_in_0_ready.value,
            dut.data_in_0_valid.value,
            dut.data_out_0_ready.value,
            dut.data_out_0_valid.value,
        )
    )


@cocotb.test()
async def test_fixed_linear(dut):
    """Test integer based vector mult"""
    samples = 1000
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
    dut.bias_valid.value = 0
    dut.data_in_0_valid.value = 0
    dut.data_out_0_ready.value = 1
    debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    debug_state(dut, "Post-clk")
    debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    debug_state(dut, "Post-clk")

    done = False
    # Set a timeout to avoid deadlock
    for i in range(samples * 100):
        await FallingEdge(dut.clk)
        debug_state(dut, "Post-clk")
        dut.weight_valid.value = test_case.weight.pre_compute()
        dut.bias_valid.value = test_case.bias.pre_compute()
        dut.data_in_0_valid.value = test_case.data_in.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_0_ready.value = test_case.outputs.pre_compute(
            dut.data_out_0_valid.value
        )
        await Timer(1, units="ns")
        debug_state(dut, "Post-clk")

        dut.bias_valid.value, dut.bias.value = test_case.bias.compute(
            dut.bias_ready.value
        )
        dut.weight_valid.value, dut.weight.value = test_case.weight.compute(
            dut.weight_ready.value
        )
        dut.data_in_0_valid.value, dut.data_in_0.value = test_case.data_in.compute(
            dut.data_in_0_ready.value
        )
        await Timer(1, units="ns")
        dut.data_out_0_ready.value = test_case.outputs.compute(
            dut.data_out_0_valid.value, dut.data_out_0.value
        )
        debug_state(dut, "Pre-clk")

        if (
            (not test_case.has_bias or test_case.bias.is_empty())
            and test_case.weight.is_empty()
            and test_case.data_in.is_empty()
            and test_case.outputs.is_full()
        ):
            done = True
            break
    assert (
        done
    ), "Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)"

    check_results(test_case.outputs.data, test_case.ref)


if __name__ == "__main__":
    tb = VerificationCase()
    mase_runner(module_param_list=[tb.get_dut_parameters()])
