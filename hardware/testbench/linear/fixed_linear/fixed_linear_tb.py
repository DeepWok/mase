#!/usr/bin/env python3

# This script tests the fixed point linear
import random, os, math, logging, sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from RandomTest import RandomSource
from RandomTest import RandomSink
from RandomTest import check_results

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock
from cocotb.runner import get_runner

debug = False

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=10):
        self.data_in_width = 32
        self.weights_width = 16
        self.vector_size = 8
        self.iterations = 3
        self.parallelism = 7
        self.has_bias = 1
        self.data_in = RandomSource(
            name="data_in",
            samples=samples * self.iterations,
            num=self.vector_size,
            max_stalls=2 * samples,
            debug=debug,
        )
        self.weights = RandomSource(
            name="weights",
            samples=samples * self.iterations,
            num=self.vector_size * self.parallelism,
            max_stalls=2 * samples,
            debug=debug,
        )
        self.bias = RandomSource(
            name="bias",
            samples=samples,
            num=self.parallelism,
            max_stalls=2 * samples,
            debug=debug,
        )
        self.outputs = RandomSink(samples=samples, max_stalls=2 * samples, debug=debug)
        self.samples = samples
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "IN_WIDTH": self.data_in_width,
            "W_WIDTH": self.weights_width,
            "IN_SIZE": self.vector_size,
            "IN_DEPTH": self.iterations,
            "PARALLELISM": self.parallelism,
            "HAS_BIAS": self.has_bias,
        }

    def sw_compute(self):
        ref = []
        for i in range(self.samples):
            acc = [0 for _ in range(self.parallelism)]
            for j in range(self.iterations):
                data_idx = i * self.iterations + j
                temp = []
                for k in range(self.parallelism):
                    s = [
                        self.data_in.data[data_idx][h]
                        * self.weights.data[data_idx][k * self.vector_size + h]
                        for h in range(self.vector_size)
                    ]
                    acc[k] += sum(s)
            if self.has_bias:
                for k in range(self.parallelism):
                    acc[k] += self.bias.data[i][k]
            ref.append(acc)
        ref.reverse()
        return ref


# Check if an is_impossible state is reached
def is_impossible_state(
    weights_ready,
    weights_valid,
    data_in_ready,
    data_in_valid,
    data_out_ready,
    data_out_valid,
):
    return False


def debug_state(dut, state):
    logger.debug(
        "{} State: (bias_ready,bias_valid,bias_ready,bias_valid,data_in_ready,data_in_valid,data_out_ready,data_out_valid) = ({},{},{},{},{},{})".format(
            state,
            dut.bias_ready.value,
            dut.bias_valid.value,
            dut.weights_ready.value,
            dut.weights_valid.value,
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )
    )


@cocotb.test()
async def test_fixed_linear(dut):
    """Test integer based vector mult"""
    samples = 20
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
    dut.weights_valid.value = 0
    dut.bias_valid.value = 0
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
    for i in range(samples * 100):
        await FallingEdge(dut.clk)
        debug_state(dut, "Post-clk")
        dut.weights_valid.value = test_case.weights.pre_compute(dut.weights_ready.value)
        dut.bias_valid.value = test_case.bias.pre_compute(dut.bias_ready.value)
        dut.data_in_valid.value = test_case.data_in.pre_compute(dut.data_in_ready.value)
        debug_state(dut, "Pre-clk")
        await Timer(1, units="ns")
        debug_state(dut, "Post-clk")

        dut.bias_valid.value, dut.bias.value = test_case.bias.compute(
            dut.bias_ready.value
        )
        dut.weights_valid.value, dut.weights.value = test_case.weights.compute(
            dut.weights_ready.value
        )
        dut.data_in_valid.value, dut.data_in.value = test_case.data_in.compute(
            dut.data_in_ready.value
        )
        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )
        debug_state(dut, "Pre-clk")

        if (
            (not test_case.has_bias or test_case.bias.is_empty())
            and test_case.weights.is_empty()
            and test_case.data_in.is_empty()
            and test_case.outputs.is_full()
        ):
            done = True
            break
    assert (
        done
    ), "Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)"

    check_results(test_case.outputs.data, test_case.ref)


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = [
        "../../../../hardware/common/fixed_dot_product.sv",
        "../../../../hardware/linear/fixed_linear.sv",
        "../../../../hardware/common/fixed_accumulator.sv",
        "../../../../hardware/common/fixed_vector_mult.sv",
        "../../../../hardware/common/register_slice.sv",
        "../../../../hardware/common/fixed_adder_tree.sv",
        "../../../../hardware/common/fixed_adder_tree_layer.sv",
        "../../../../hardware/common/fixed_mult.sv",
        "../../../../hardware/common/join2.sv",
    ]
    test_case = VerificationCase()

    # set parameters
    extra_args = []
    for k, v in test_case.get_dut_parameters().items():
        extra_args.append(f"-G{k}={v}")
    print(extra_args)
    runner = get_runner(sim)()
    runner.build(
        verilog_sources=verilog_sources,
        toplevel="fixed_linear",
        extra_args=extra_args,
    )

    runner.test(toplevel="fixed_linear", py_module="fixed_linear_tb")


if __name__ == "__main__":
    runner()
