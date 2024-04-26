#!/usr/bin/env python3

# This script tests the fixed point dot product
import random, os, math, logging, sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mase_cocotb.random_test import RandomSource
from mase_cocotb.random_test import RandomSink
from mase_cocotb.random_test import check_results_signed
import mase_cocotb.utils
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
        self.data_in_width = 1
        self.weight_width = 1
        self.vector_size = 9
        self.data_in = RandomSource(
            name="data_in",
            samples=samples,
            num=self.vector_size,
            max_stalls=2 * samples,
            debug=debug,
            arithmetic="binary",
        )
        self.weight = RandomSource(
            name="weight",
            samples=samples,
            num=self.vector_size,
            max_stalls=2 * samples,
            debug=debug,
            arithmetic="binary",
        )
        self.outputs = RandomSink(samples=samples, max_stalls=2 * samples, debug=debug)
        self.samples = samples
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "IN_WIDTH": self.data_in_width,
            "WEIGHT_WIDTH": self.weight_width,
            "IN_SIZE": self.vector_size,
        }

    def sw_compute(self):
        ref = []
        for i in range(self.samples):
            s = [
                int(not (self.data_in.data[i][j] ^ self.weight.data[i][j]))
                for j in range(self.vector_size)
            ]
            ref.append(sum([utils.binary_decode(element) for element in s]))
        ref.reverse()
        return ref


def in_out_wave(dut, name):
    logger.debug(
        "{}  State: (w_valid, w_ready, in_valid,in_ready,out_valid,out_ready) = ({},{},{},{},{},{})".format(
            name,
            dut.weight_valid.value,
            dut.weight_ready.value,
            dut.data_in_valid.value,
            dut.data_in_ready.value,
            dut.data_out_valid.value,
            dut.data_out_ready.value,
        )
    )


@cocotb.test()
async def cocotb_test_binary_activation_binary_dot_product(dut):
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
    in_out_wave(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    in_out_wave(dut, "Post-clk")

    in_out_wave(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    in_out_wave(dut, "Post-clk")

    done = False
    # Set a timeout to avoid deadlock
    for i in range(samples * 10):
        await FallingEdge(dut.clk)
        in_out_wave(dut, "Post-clk")
        ## pre_compute
        dut.weight_valid.value = test_case.weight.pre_compute()
        dut.data_in_valid.value = test_case.data_in.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(
            dut.data_out_valid.value
        )
        await Timer(1, units="ns")
        ## compute
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
        in_out_wave(dut, "Pre-clk")

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

    check_results_signed(test_case.outputs.data, test_case.ref)


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = [
        "../../../../components/binary_arith/binary_activation_binary_dot_product.sv",
        "../../../../components/binary_arith/binary_activation_binary_vector_mult.sv",
        "../../../../components/binary_arith/binary_activation_binary_adder_tree.sv",
        "../../../../components/binary_arith/binary_activation_binary_adder_tree_layer.sv",
        "../../../../components/binary_arith/binary_activation_binary_mult.sv",
        "../../../../components/common/register_slice.sv",
        "../../../../components/common/join2.sv",
    ]
    test_case = VerificationCase(samples=100)

    # set parameters
    extra_args = []
    for k, v in test_case.get_dut_parameters().items():
        extra_args.append(f"-G{k}={v}")
    print(extra_args)
    runner = get_runner(sim)
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="binary_activation_binary_dot_product",
        build_args=extra_args,
    )

    runner.test(
        hdl_toplevel="binary_activation_binary_dot_product",
        test_module="binary_activation_binary_dot_product_tb",
    )


if __name__ == "__main__":
    runner()
