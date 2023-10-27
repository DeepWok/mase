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

from z_qlayers import QuantizedConvolution
import torch
import torch.nn as nn

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock
from cocotb.runner import get_runner

from math import ceil

debug = True

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=1):
        self.data_width = 32
        self.data_frac_width = 1
        self.weight_width = 16
        self.weight_frac_width = 1
        self.bias_width = 16
        self.bias_frac_width = 1
        self.out_data_width = 32
        self.out_data_frac_width = 1

        self.in_height = 2
        self.in_width = 4

        self.in_channels = 2
        self.unroll_in_c = 2
        self.unroll_kernel_out = 4

        self.kernel_height = 2
        self.kernel_width = 3
        self.out_channels = 4
        self.unroll_out_c = 2

        self.stride = 3
        self.padding_height = 1
        self.padding_width = 2

        self.out_height = ceil(
            (self.in_height - self.kernel_height + 2 * self.padding_height + 1)
            / self.stride
        )
        self.out_width = ceil(
            (self.in_width - self.kernel_width + 2 * self.padding_width + 1)
            / self.stride
        )

        self.sliding_depth = self.out_width * self.out_height

        self.samples = samples
        test_data_in, test_weight, test_bias, _, _, _ = self.data_generate()
        print(
            "data_in",
            test_data_in,
            "test_weight",
            test_weight,
            "test_bias",
            test_bias,
        )
        self.data_in = RandomSource(
            name="data_in",
            samples=int(
                samples
                * self.in_width
                * self.in_height
                * self.in_channels
                / self.unroll_in_c
            ),
            num=self.unroll_in_c,
            max_stalls=2 * samples,
            debug=debug,
            data_specify=test_data_in,
        )
        self.weight_partition_depth = int(
            self.kernel_height
            * self.kernel_width
            * self.in_channels
            * self.out_channels
            / self.unroll_kernel_out
            / self.unroll_out_c
        )
        self.weight = RandomSource(
            name="weight",
            samples=samples * self.weight_partition_depth,
            num=self.unroll_kernel_out * self.unroll_out_c,
            max_stalls=2 * samples * self.weight_partition_depth,
            data_specify=test_weight,
            debug=debug,
        )
        self.bias = RandomSource(
            name="bias",
            samples=samples * int(self.out_channels / self.unroll_out_c),
            num=self.unroll_out_c,
            max_stalls=2 * samples * int(self.out_channels / self.unroll_out_c),
            data_specify=test_bias,
            debug=debug,
        )
        self.outputs = RandomSink(
            samples=samples
            * int(self.out_channels / self.unroll_out_c)
            * self.sliding_depth,
            max_stalls=2 * samples,
            debug=debug,
        )
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "DATA_WIDTH": self.data_width,
            "DATA_FRAC_WIDTH": self.data_frac_width,
            "W_WIDTH": self.weight_width,
            "W_FRAC_WIDTH": self.weight_frac_width,
            "BIAS_WIDTH": self.bias_width,
            "BIAS_FRAC_WIDTH": self.bias_frac_width,
            "OUT_WIDTH": self.out_data_width,
            "OUT_FRAC_WIDTH": self.out_data_frac_width,
            "IN_X": self.in_width,
            "IN_Y": self.in_height,
            "IN_C": self.in_channels,
            "KERNEL_X": self.kernel_width,
            "KERNEL_Y": self.kernel_height,
            "OUT_C": self.out_channels,
            "UNROLL_OUT_C": self.unroll_out_c,
            "UNROLL_IN_C": self.unroll_in_c,
            "UNROLL_KERNEL_OUT": self.unroll_kernel_out,
            "SLIDING_NUM": self.sliding_depth,
            "STRIDE": self.stride,
            "PADDING_Y": self.padding_height,
            "PADDING_X": self.padding_width,
        }

    def data_generate(self):
        torch.manual_seed(0)
        samples = self.samples
        # in dimension
        in_width = self.in_width
        in_height = self.in_height
        in_channels = self.in_channels
        unroll_in_c = self.unroll_in_c
        # weight dimension
        kernel_width = self.kernel_width
        kernel_height = self.kernel_height
        out_channels = self.out_channels
        unroll_out_c = self.unroll_out_c
        unroll_kernel_out = self.unroll_kernel_out
        # data_pack
        re_data_tensor = torch.randint(30, (samples, in_channels, in_height, in_width))
        # from (samples, c, h, w) to (samples*h*w*c/unroll_in_c, unroll_in_c)
        data_tensor = re_data_tensor.permute(0, 2, 3, 1).reshape(-1, unroll_in_c)
        data_in = data_tensor.type(torch.int).flip(0).tolist()
        # weight_pack
        re_w_tensor = torch.randint(
            30, (samples, out_channels, in_channels, kernel_height, kernel_width)
        )

        # from (oc,ic/unroll_in_c,unroll_in_c,h,w) to (ic/unroll_in_c,h*w,unroll_in_c,oc)
        reorder_w_tensor = re_w_tensor.reshape(
            samples,
            out_channels,
            int(in_channels / unroll_in_c),
            unroll_in_c,
            kernel_height * kernel_width,
        ).permute(0, 2, 4, 3, 1)

        # reverse the final 2 dimension
        # from(samples, int(kernel_height * kernel_width * in_channels / unroll_kernel_out), unroll_kernel_out, int(out_channels/unroll_out_c ), unroll_out_c )
        # to  (samples, int(out_channels/unroll_out_c ), int(kernel_height * kernel_width * in_channels / unroll_kernel_out), unroll_out_c , unroll_kernel_out)
        w_tensor = reorder_w_tensor.reshape(
            samples,
            int(kernel_height * kernel_width * in_channels / unroll_kernel_out),
            unroll_kernel_out,
            int(out_channels / unroll_out_c),
            unroll_out_c,
        ).permute(0, 3, 1, 4, 2)

        w_tensor = w_tensor.reshape(
            -1,
            unroll_out_c * unroll_kernel_out,
        )
        w_in = w_tensor.type(torch.int).flip(0).tolist()
        # bias_pack
        re_bias_tensor = torch.randint(30, (samples, out_channels))
        bias_tensor = re_bias_tensor.reshape(-1, unroll_out_c)
        bias_in = bias_tensor.type(torch.int).flip(0).tolist()
        return (
            data_in,
            w_in,
            bias_in,
            re_data_tensor.type(torch.float),
            re_w_tensor.type(torch.float),
            re_bias_tensor.type(torch.float),
        )

    def out_unpack(self, data_out):
        out_height = self.out_height
        out_width = self.out_width
        data_out = data_out.reshape(self.out_channels, out_height, out_width).permute(
            1, 2, 0
        )
        return data_out.reshape(-1, self.out_channels)

    def sw_compute(self):
        ref = []
        output = []
        _, _, _, data, weight, bias = self.data_generate()
        logger.debug(
            "input data: \n\
            data_in = \n\
            {} \n\
            weight  = \n\
            {} \n\
            bias    = \n\
            {} \n\
            ".format(
                data,
                weight,
                bias,
            )
        )
        for i in range(self.samples):
            kernel_size = (self.kernel_height, self.kernel_width)
            Qconv = QuantizedConvolution(
                self.in_channels,
                self.out_channels,
                kernel_size,
                weight[i],
                bias[i],
                stride=self.stride,
                padding=(self.padding_height, self.padding_width),
                data_width=self.data_width,
                data_frac_width=self.data_frac_width,
                weight_width=self.weight_width,
                weight_frac_width=self.weight_frac_width,
                bias_width=self.bias_width,
                bias_frac_width=self.bias_frac_width,
                out_width=self.out_data_width,
                out_frac_width=self.out_data_frac_width,
            )
            # Turn an input to a single batch input
            data_out = Qconv(data[i].unsqueeze(0))
            data_out = self.out_unpack(data_out)
            output = data_out.reshape(-1, self.unroll_out_c).tolist()
            ref = ref + output
        # ref.reverse()
        return ref


# Check if an is_impossible state is reached
def is_impossible_state(
    weight_ready,
    weight_valid,
    data_in_ready,
    data_in_valid,
    data_out_ready,
    data_out_valid,
):
    return False


def debug_state(dut, state):
    logger.debug(
        "{} State: (bias_ready,bias_valid,weight_ready,weight_valid,data_in_ready,data_in_valid,data_out_ready,data_out_valid) = ({},{},{},{},{},{},{},{})".format(
            state,
            dut.bias_ready.value,
            dut.bias_valid.value,
            dut.weight_ready.value,
            dut.weight_valid.value,
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
    dut.weight_valid.value = 0
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
    count = 0
    for i in range(samples * 1000):
        await FallingEdge(dut.clk)
        debug_state(dut, "Post-clk")
        dut.weight_valid.value = test_case.weight.pre_compute()
        dut.bias_valid.value = test_case.bias.pre_compute()
        dut.data_in_valid.value = test_case.data_in.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(
            dut.data_out_valid.value
        )
        await Timer(1, units="ns")
        debug_state(dut, "Post-clk")

        dut.bias_valid.value, dut.bias.value = test_case.bias.compute(
            dut.bias_ready.value
        )
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
        await Timer(1, units="ns")
        debug_state(dut, "Pre-clk")
        wave_check(dut)
        logger.debug(
            "wave of interface:\n\
                {},{} kernel = {}  \n\
                {},{} rolled_k = {}   \n\
                padding_x = {} \n\
                padding_y = {} \n\
                padding_c = {} \n\
                ".format(
                dut.kernel_valid.value,
                dut.kernel_ready.value,
                [int(i) for i in dut.kernel.value],
                dut.rolled_k_valid.value,
                dut.rolled_k_ready.value,
                [int(i) for i in dut.rolled_k.value],
                int(dut.sw_inst.padding_inst.count_x.value),
                int(dut.sw_inst.padding_inst.count_y.value),
                int(dut.sw_inst.padding_inst.count_c.value),
            )
        )
        if dut.kernel_valid.value == 1 and dut.kernel_ready.value == 1:
            count += 1
        print(count)

        # breakpoint()
        if (
            (test_case.bias.is_empty())
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


def wave_check(dut):
    logger.debug(
        "wave of linear:\n\
            {},{},data_in = {}\n\
            {},{},weight = {}\n\
            {},{},bias = {}\n\
            ".format(
            dut.fl_instance.data_in_valid.value,
            dut.fl_instance.data_in_ready.value,
            [int(i) for i in dut.fl_instance.data_in.value],
            dut.fl_instance.weight_valid.value,
            dut.fl_instance.weight_ready.value,
            [int(i) for i in dut.fl_instance.weight.value],
            dut.fl_instance.bias_valid.value,
            dut.fl_instance.bias_ready.value,
            [int(i) for i in dut.fl_instance.bias.value],
        )
    )


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = [
        "../../../../components/conv/convolution.sv",
        "../../../../components/conv/padding.sv",
        "../../../../components/conv/roller.sv",
        "../../../../components/conv/sliding_window.sv",
        "../../../../components/cast/fixed_cast.sv",
        "../../../../components/linear/fixed_linear.sv",
        "../../../../components/fixed_arith/fixed_dot_product.sv",
        "../../../../components/fixed_arith/fixed_accumulator.sv",
        "../../../../components/fixed_arith/fixed_vector_mult.sv",
        "../../../../components/fixed_arith/fixed_adder_tree.sv",
        "../../../../components/fixed_arith/fixed_adder_tree_layer.sv",
        "../../../../components/fixed_arith/fixed_mult.sv",
        "../../../../components/common/skid_buffer.sv",
        "../../../../components/common/unpaced_skid_buffer.sv",
        "../../../../components/common/join2.sv",
        "../../../../components/common/input_buffer.sv",
        "../../../../components/common/ram_block.sv",
    ]
    test_case = VerificationCase()

    # set parameters
    extra_args = []
    for k, v in test_case.get_dut_parameters().items():
        extra_args.append(f"-G{k}={v}")
    runner = get_runner(sim)
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="convolution",
        build_args=extra_args,
    )

    runner.test(hdl_toplevel="convolution", test_module="convolution_tb")


if __name__ == "__main__":
    runner()
