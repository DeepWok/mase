#!/usr/bin/env python3

# This script tests the fixed point linear
import random, os, math, logging, sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from random_test import RandomSource
from random_test import RandomSink
from random_test import check_results

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock
from cocotb.runner import get_runner

debug = True

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=10):
        self.data_in_width = 32
        self.data_in_frac_width = 8
        self.weight_width = 16
        self.weight_frac_width = 8
        self.data_out_width = 32
        self.data_out_frac_width = 8

        self.in_rows = 3
        self.in_columns = 4
        self.weight_rows = self.in_columns
        self.weight_columns = 2
        self.iterations = 3
        self.data_in = RandomSource(
            name="data_in",
            samples=samples * self.iterations,
            num=self.in_rows * self.in_columns,
            max_stalls=2 * samples * self.iterations,
            debug=debug,
        )
        self.weight = RandomSource(
            name="weight",
            samples=samples * self.iterations,
            num=self.weight_rows * self.weight_columns,
            max_stalls=2 * samples * self.iterations,
            debug=debug,
        )
        self.outputs = RandomSink(samples=samples, max_stalls=2 * samples, debug=debug)
        self.samples = samples
        self.ref = self.sw_compute()
        self.ref = self.sw_cast(
            inputs=self.ref,
            in_width=self.data_in_width + self.weight_width,
            in_frac_width=self.data_in_frac_width + self.weight_frac_width,
            out_width=self.data_out_width,
            out_frac_width=self.data_out_frac_width,
        )

    def get_dut_parameters(self):
        return {
            "IN1_WIDTH": self.data_in_width,
            "IN1_FRAC_WIDTH": self.data_in_frac_width,
            "IN2_WIDTH": self.weight_width,
            "IN2_FRAC_WIDTH": self.weight_frac_width,
            "OUT_WIDTH": self.data_out_width,
            "OUT_FRAC_WIDTH": self.data_out_frac_width,
            "IN1_PARALLELISM": self.in_rows,
            "IN_SIZE": self.in_columns,
            "IN2_PARALLELISM": self.weight_columns,
            "IN_DEPTH": self.iterations,
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
            dut.data_in1_ready.value,
            dut.data_in2_valid.value,
            dut.data_in1_ready.value,
            dut.data_in1_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )
    )


@cocotb.test()
async def test_fixed_linear(dut):
    """Test integer based vector mult"""
    samples = 30
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
    dut.data_in2_valid.value = 0
    dut.data_in1_valid.value = 0
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
        dut.data_in2_valid.value = test_case.weight.pre_compute()
        dut.data_in1_valid.value = test_case.data_in.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(
            dut.data_out_valid.value
        )
        debug_state(dut, "Pre-clk")
        await Timer(1, units="ns")
        debug_state(dut, "Post-clk")
        dut.data_in2_valid.value, dut.data_in2.value = test_case.weight.compute(
            dut.data_in2_ready.value
        )
        dut.data_in1_valid.value, dut.data_in1.value = test_case.data_in.compute(
            dut.data_in1_ready.value
        )
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )
        debug_state(dut, "Pre-clk")
        if (
            test_case.weight.is_empty()
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
        "../../../../components/fixed_arith/fixed_matmul_core.sv",
        "../../../../components/common/register_slice.sv",
        "../../../../components/common/join2.sv",
        "../../../../components/linear/fixed_linear.sv",
        "../../../../components/cast/fixed_cast.sv",
        "../../../../components/fixed_arith/fixed_dot_product.sv",
        "../../../../components/fixed_arith/fixed_accumulator.sv",
        "../../../../components/fixed_arith/fixed_vector_mult.sv",
        "../../../../components/fixed_arith/fixed_adder_tree.sv",
        "../../../../components/fixed_arith/fixed_adder_tree_layer.sv",
        "../../../../components/fixed_arith/fixed_mult.sv",
    ]
    test_case = VerificationCase()

    # set parameters
    extra_args = []
    for k, v in test_case.get_dut_parameters().items():
        extra_args.append(f"-G{k}={v}")
    print(extra_args)
    runner = get_runner(sim)
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="fixed_matmul_core",
        build_args=extra_args,
    )

    runner.test(
        hdl_toplevel="fixed_matmul_core",
        test_module="fixed_matmul_core_tb",
    )


if __name__ == "__main__":
    runner()
