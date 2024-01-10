#!/usr/bin/env python3

# This script tests the fixed point linear
import random, os, math, logging, sys

# pd is parent directory
pd = os.path.dirname
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
machop_dir = pd(pd(pd(pd(pd(os.path.abspath(__file__)))))) + "/machop"
sys.path.append(machop_dir)
from random_test import RandomSource
from random_test import RandomSink
from random_test import check_results

from chop.passes.graph.transforms.quantize.quantized_modules import Conv2dInteger

import torch

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock
from cocotb.runner import get_runner

from z_qlayers import quantize_to_int as q2i

from math import ceil

debug = True

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=1):
        self.w_config = {
            "data_in_width": 8,
            "data_in_frac_width": 3,
            "weight_width": 8,
            "weight_frac_width": 3,
            "bias_width": 8,
            "bias_frac_width": 3,
        }
        self.data_out_width = 6
        self.data_out_frac_width = 2

        self.in_c = 2
        self.in_y = 4
        self.in_x = 3

        self.out_channels = 4
        self.kernel_y = 3
        self.kernel_x = 2

        self.unroll_in_c = 2
        self.unroll_kernel_out = 4

        self.unroll_out_c = 2

        self.stride = 2
        self.padding_height = 2
        self.padding_width = 1

        self.out_height = ceil(
            (self.in_y - self.kernel_y + 2 * self.padding_height + 1) / self.stride
        )
        self.out_width = ceil(
            (self.in_x - self.kernel_x + 2 * self.padding_width + 1) / self.stride
        )

        self.sliding_num = self.out_width * self.out_height

        self.samples = samples
        self.data_generate()

        self.data_in = RandomSource(
            name="data_in",
            samples=int(samples * self.in_x * self.in_y * self.in_c / self.unroll_in_c),
            num=self.unroll_in_c,
            max_stalls=2 * samples,
            debug=debug,
            data_specify=self.hw_x,
        )
        self.weight_partition_depth = int(
            self.kernel_y
            * self.kernel_x
            * self.in_c
            * self.out_channels
            / self.unroll_kernel_out
            / self.unroll_out_c
        )
        self.weight = RandomSource(
            name="weight",
            samples=samples * self.weight_partition_depth,
            num=self.unroll_kernel_out * self.unroll_out_c,
            max_stalls=2 * samples * self.weight_partition_depth,
            data_specify=self.hw_w,
            debug=debug,
        )
        self.bias = RandomSource(
            name="bias",
            samples=samples * int(self.out_channels / self.unroll_out_c),
            num=self.unroll_out_c,
            max_stalls=2 * samples * int(self.out_channels / self.unroll_out_c),
            data_specify=self.hw_b,
            debug=debug,
        )
        self.outputs = RandomSink(
            samples=samples
            * int(self.out_channels / self.unroll_out_c)
            * self.sliding_num,
            max_stalls=2 * samples,
            debug=debug,
        )
        self.ref = self.sw_compute()

    def get_dut_parameters(self):
        return {
            "DATA_WIDTH": self.w_config["data_in_width"],
            "DATA_FRAC_WIDTH": self.w_config["data_in_frac_width"],
            "W_WIDTH": self.w_config["weight_width"],
            "W_FRAC_WIDTH": self.w_config["weight_frac_width"],
            "BIAS_WIDTH": self.w_config["bias_width"],
            "BIAS_FRAC_WIDTH": self.w_config["bias_frac_width"],
            "OUT_WIDTH": self.data_out_width,
            "OUT_FRAC_WIDTH": self.data_out_frac_width,
            "IN_X": self.in_x,
            "IN_Y": self.in_y,
            "IN_C": self.in_c,
            "KERNEL_X": self.kernel_x,
            "KERNEL_Y": self.kernel_y,
            "OUT_C": self.out_channels,
            "UNROLL_OUT_C": self.unroll_out_c,
            "UNROLL_IN_C": self.unroll_in_c,
            "UNROLL_KERNEL_OUT": self.unroll_kernel_out,
            "SLIDING_NUM": self.sliding_num,
            "STRIDE": self.stride,
            "PADDING_Y": self.padding_height,
            "PADDING_X": self.padding_width,
        }

    def data_generate(self):
        torch.manual_seed(0)
        self.int_conv_layer = Conv2dInteger(
            in_channels=self.in_c,
            out_channels=self.out_channels,
            kernel_size=(self.kernel_y, self.kernel_x),
            stride=self.stride,
            padding=(self.padding_height, self.padding_width),
            config=self.w_config,
        )

        # get parameters with integer format
        self.sw_x = 5 * torch.randn(self.samples, self.in_c, self.in_y, self.in_x)

        self.sw_w = self.int_conv_layer.weight
        self.sw_b = self.int_conv_layer.bias
        # data_in_pack
        x = q2i(
            self.sw_x,
            self.w_config["data_in_width"],
            self.w_config["data_in_frac_width"],
        )
        print("x = ", x)
        # from (samples, c, h, w) to (samples*h*w*c/unroll_in_c, unroll_in_c)
        # flip from convinient debug
        reshape_x = (
            x.permute(0, 2, 3, 1).reshape(-1).flip(0).reshape(-1, self.unroll_in_c)
        )
        self.hw_x = reshape_x.type(torch.int).tolist()
        # parameters packs
        self.hw_w, self.hw_b = self.conv_pack(
            weight=self.sw_w,
            bias=self.sw_b,
            in_channels=self.in_c,
            kernel_size=[self.kernel_y, self.kernel_x],
            out_channels=self.out_channels,
            unroll_in_channels=self.unroll_in_c,
            unroll_kernel_out=self.unroll_kernel_out,
            unroll_out_channels=self.unroll_out_c,
        )

    def sw_compute(self):
        """
        The output of this module should follow channel first output model
        Software level output dimension should be [oc, oh, ow] first
        it should be reshaped to [oh,ow,oc/u_oc,,u_oc]
        then for the purpose of mapping hardware index, should flip the last dimension
        """
        sw_data_out = self.int_conv_layer(self.sw_x)
        print(q2i(sw_data_out, self.data_out_width, self.data_out_frac_width))
        data_out_temp = q2i(sw_data_out, self.data_out_width, self.data_out_frac_width)
        data_out_temp = data_out_temp.permute(0, 2, 3, 1)
        data_out_temp = data_out_temp.reshape(-1, self.unroll_out_c)
        hw_data_out = data_out_temp.flip(-1).tolist()
        return hw_data_out

    def conv_pack(
        self,
        weight,
        bias,
        in_channels,
        kernel_size,
        out_channels,
        unroll_in_channels,
        unroll_kernel_out,
        unroll_out_channels,
    ):
        print("weight = ", weight)
        print("bias = ", bias)
        weight = q2i(
            weight,
            self.w_config["weight_width"],
            self.w_config["weight_frac_width"],
        )
        print("weight = ", weight)
        bias = q2i(
            bias,
            self.w_config["bias_width"],
            self.w_config["bias_frac_width"],
        )
        print("bias = ", bias)
        samples = self.samples
        # requires input as a quantized int format
        # weight_pack
        # from (oc,ic/u_ic,u_ic,h,w) to (ic/u_ic,h*w,u_ic,oc)
        reorder_w_tensor = (
            weight.repeat(samples, 1, 1, 1, 1)
            .reshape(
                samples,
                out_channels,
                int(in_channels / unroll_in_channels),
                unroll_in_channels,
                kernel_size[0] * kernel_size[1],
            )
            .permute(0, 2, 4, 3, 1)
        )

        # reverse the final 2 dimension
        # from(samples, int(kernel_height * kernel_width * in_channels / unroll_kernel_out), unroll_kernel_out, int(out_channels/unroll_out_channels), unroll_out_channels)
        # to  (samples, int(out_channels/unroll_out_channels), int(kernel_height * kernel_width * in_channels / unroll_kernel_out), unroll_out_channels, unroll_kernel_out)
        w_tensor = reorder_w_tensor.reshape(
            samples,
            int(kernel_size[0] * kernel_size[1] * in_channels / unroll_kernel_out),
            unroll_kernel_out,
            int(out_channels / unroll_out_channels),
            unroll_out_channels,
        ).permute(0, 3, 1, 4, 2)

        w_tensor = (
            w_tensor.reshape(-1)
            .flip(0)
            .reshape(
                -1,
                unroll_out_channels * unroll_kernel_out,
            )
        )
        w_in = w_tensor.type(torch.int).tolist()
        # bias_pack
        bias_tensor = (
            bias.repeat(samples, 1).reshape(-1).flip(0).reshape(-1, unroll_out_channels)
        )
        b_in = bias_tensor.type(torch.int).tolist()
        return w_in, b_in


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
            dut.data_in_0_ready.value,
            dut.data_in_0_valid.value,
            dut.data_out_0_ready.value,
            dut.data_out_0_valid.value,
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
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
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
        await Timer(1, units="ns")
        debug_state(dut, "Pre-clk")
        wave_check(dut)
        if dut.ib_weight_valid.value == 1 and dut.ib_weight_ready.value == 1:
            count1 += 1
        if dut.ib_bias_valid.value == 1 and dut.ib_bias_ready.value == 1:
            count2 += 1
        if dut.data_out_0_valid.value == 1 and dut.data_out_0_ready.value == 1:
            count3 += 1
        if dut.ib_rolled_k_valid.value == 1 and dut.ib_rolled_k_ready.value == 1:
            count4 += 1
        print(
            "count:\n\
              c_weight = {}\n\
              c_bias = {}\n\
              c_data_out = {}\n\
              c_linear_in = {}".format(
                count1, count2, count3, count4
            )
        )
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
            dut.fl_instance.data_in_0_valid.value,
            dut.fl_instance.data_in_0_ready.value,
            [int(i) for i in dut.fl_instance.data_in_0.value],
            dut.fl_instance.weight_valid.value,
            dut.fl_instance.weight_ready.value,
            [int(i) for i in dut.fl_instance.weight.value],
            dut.fl_instance.bias_valid.value,
            dut.fl_instance.bias_ready.value,
            [int(i) for i in dut.fl_instance.bias.value],
        )
    )
    logger.debug(
        "wave of interface:\n\
            {},{} data_in = {}  \n\
            {},{} kernel = {}  \n\
            {},{} rolled_k = {}   \n\
            {},{} data_out = {}  \n\
            padding_x = {} \n\
            padding_y = {} \n\
            padding_c = {} \n\
            ".format(
            dut.data_in_0_valid.value,
            dut.data_in_0_ready.value,
            [int(i) for i in dut.data_in_0.value],
            dut.kernel_valid.value,
            dut.kernel_ready.value,
            [int(i) for i in dut.kernel.value],
            dut.rolled_k_valid.value,
            dut.rolled_k_ready.value,
            [int(i) for i in dut.rolled_k.value],
            dut.data_out_0_valid.value,
            dut.data_out_0_ready.value,
            [int(i) for i in dut.data_out_0.value],
            int(dut.sw_inst.padding_inst.count_x.value),
            int(dut.sw_inst.padding_inst.count_y.value),
            int(dut.sw_inst.padding_inst.count_c.value),
        )
    )


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = [
        "../../../../components/conv/convolution.sv",
        "../../../../components/conv/padding.sv",
        "../../../../components/conv/roller.sv",
        "../../../../components/conv/sliding_window.sv",
        "../../../../components/cast/fixed_rounding.sv",
        "../../../../components/linear/fixed_linear.sv",
        "../../../../components/fixed_arith/fixed_dot_product.sv",
        "../../../../components/fixed_arith/fixed_accumulator.sv",
        "../../../../components/fixed_arith/fixed_vector_mult.sv",
        "../../../../components/fixed_arith/fixed_adder_tree.sv",
        "../../../../components/fixed_arith/fixed_adder_tree_layer.sv",
        "../../../../components/fixed_arith/fixed_mult.sv",
        "../../../../components/common/join2.sv",
        "../../../../components/common/input_buffer.sv",
        "../../../../components/common/skid_buffer.sv",
        "../../../../components/common/unpacked_skid_buffer.sv",
        "../../../../components/common/blk_mem_gen_0.sv",
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
