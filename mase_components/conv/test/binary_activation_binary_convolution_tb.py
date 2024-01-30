#!/usr/bin/env python3

# This script tests the fixed point linear
import random, os, math, logging, sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from random_test import RandomSource
from random_test import RandomSink
from random_test import check_results_signed

from Qconv import QuantizedConvolution
import torch
import torch.nn as nn
import utils

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
        self.data_width = 1
        self.data_frac_width = 0
        self.weight_width = 1
        self.weight_frac_width = 0
        self.bias_width = 16
        self.bias_frac_width = 0

        self.in_width = 4
        self.in_height = 2

        self.in_channels = 2
        self.in_size = 2
        self.weight_size = 4

        self.kernel_width = 3
        self.kernel_height = 2
        self.out_channels = 3

        self.stride = 2
        self.padding_width = 2
        self.padding_height = 1

        self.out_height = ceil(
            (self.in_height - self.kernel_height + 2 * self.padding_height + 1)
            / self.stride
        )
        self.out_width = ceil(
            (self.in_width - self.kernel_width + 2 * self.padding_width + 1)
            / self.stride
        )

        self.sliding_depth = self.out_width * self.out_height
        print(
            "output_height = {}, output_width = {}".format(
                self.out_height, self.out_width
            )
        )

        # test_data_in = [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,11],[12,12],[13,13],[14,14],[15,15]]
        # test_data_in = [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7]]
        # test_weight = [[0,0,4,4],[1,1,5,5],[2,2,6,6],[3,3,7,7],[0,0,4,4],[1,1,5,5],[2,2,6,6],[3,3,7,7]]
        # test_bias = [[1,1]]
        # test_data_in.reverse()
        # test_weight.reverse()
        self.samples = samples
        test_data_in, test_weight, test_bias, _, _, _ = self.data_generate()
        self.data_in = RandomSource(
            name="data_in",
            samples=int(
                samples
                * self.in_width
                * self.in_height
                * self.in_channels
                / self.in_size
            ),
            num=self.in_size,
            max_stalls=2 * samples,
            debug=debug,
            data_specify=test_data_in,
            arithmetic="binary",  # TODO: the model starts shotting random data to the layer... still unsure why
        )
        self.weight = RandomSource(
            name="weight",
            samples=int(
                samples
                * self.kernel_height
                * self.kernel_width
                * self.in_channels
                / self.weight_size
            ),
            num=self.weight_size * self.out_channels,
            max_stalls=2 * samples,
            data_specify=test_weight,
            debug=debug,
            arithmetic="binary",  # TODO: the model starts shotting random data to the layer... still unsure why
        )
        self.bias = RandomSource(
            name="bias",
            samples=samples,
            num=self.out_channels,
            max_stalls=2 * samples,
            data_specify=test_bias,
            debug=debug,
        )
        self.outputs = RandomSink(
            samples=samples * self.sliding_depth, max_stalls=2 * samples, debug=debug
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
            "IN_WIDTH": self.in_width,
            "IN_HEIGHT": self.in_height,
            "IN_CHANNELS": self.in_channels,
            "KERNEL_WIDTH": self.kernel_width,
            "KERNEL_HEIGHT": self.kernel_height,
            "OUT_CHANNELS": self.out_channels,
            "IN_SIZE": self.in_size,
            "W_SIZE": self.weight_size,
            "SLIDING_SIZE": self.sliding_depth,
            "STRIDE": self.stride,
            "PADDING_HEIGHT": self.padding_height,
            "PADDING_WIDTH": self.padding_width,
        }

    def data_generate(self):
        torch.manual_seed(0)
        samples = self.samples
        # in dimension
        in_width = self.in_width
        in_height = self.in_height
        in_channels = self.in_channels
        in_size = self.in_size
        # weight dimension
        kernel_width = self.kernel_width
        kernel_height = self.kernel_height
        out_channels = self.out_channels
        w_size = self.weight_size
        # data_pack
        re_data_tensor = torch.randint(
            0, 2, size=(samples, in_channels, in_height, in_width)
        )
        data_tensor = re_data_tensor.permute(0, 2, 3, 1)
        data_tensor = data_tensor.reshape(
            samples * in_height * in_width * int(in_channels / in_size), in_size
        )
        data_in = data_tensor.type(torch.int).flip(0).tolist()
        # weight_pack
        re_w_tensor = torch.randint(
            0, 2, size=(samples, out_channels, in_channels, kernel_height, kernel_width)
        )

        reorder_w_tensor = re_w_tensor.reshape(
            samples,
            out_channels,
            int(in_channels / in_size),
            in_size,
            kernel_height * kernel_width,
        ).permute(0, 2, 4, 3, 1)
        """
        (
            samples, 
            int(in_channels / in_size), 
            kernel_height * kernel_width, 
            in_size, 
            out_channels
        )
        """

        # reverse the final 2 dimension
        w_tensor = reorder_w_tensor.reshape(
            samples,
            int(kernel_height * kernel_width * in_channels / w_size),
            w_size,
            out_channels,
        ).permute(0, 1, 3, 2)
        """
        (
            samples, 
            int(kernel_height * kernel_width * in_channels / w_size), 
            out_channels, 
            w_size
        )
        """

        w_tensor = w_tensor.reshape(
            int(samples * kernel_height * kernel_width * in_channels / w_size),
            out_channels * w_size,
        )
        w_in = w_tensor.type(torch.int).flip(0).tolist()
        # bias_pack
        re_bias_tensor = torch.randint(30, (samples, out_channels))
        bias_in = re_bias_tensor.type(torch.int).flip(0).tolist()
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
            "sw_compute w/o conversion: \n \
            input data: \n\
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
                torch.where(weight[i] == 0, torch.tensor(-1.0), torch.tensor(1.0)),
                bias[i],
                stride=self.stride,
                padding=(self.padding_height, self.padding_width),
                DWidth=self.data_width,
                DFWidth=self.data_frac_width,
                WWidth=self.weight_width,
                WFWidth=self.weight_frac_width,
                BWidth=self.bias_width,
                BFWidth=self.bias_frac_width,
            )
            data_out = Qconv(
                torch.where(
                    data[i] == 0, torch.tensor(-1.0), torch.tensor(1.0)
                ).unsqueeze(0)
            )
            logger.debug(
                """
                        sw_compute:
                         data_in = {}
                         weight = {}
                         bia = {}
                         data_out = {}
                         """.format(
                    torch.where(
                        data[i] == 0, torch.tensor(-1.0), torch.tensor(1.0)
                    ).unsqueeze(0),
                    torch.where(weight[i] == 0, torch.tensor(-1.0), torch.tensor(1.0)),
                    bias[i],
                    data_out,
                )
            )
            data_out = self.out_unpack(data_out)
            output = data_out.tolist()
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


def wave_check(dut, instance):
    if instance == "sliding_window":
        logger.debug(
            "wave of sliding_window:\n\
                {},{},data_in = {}\n\
                {},{},data_out = {}\n\
                ".format(
                dut.sw_inst.data_in_valid.value,
                dut.sw_inst.data_in_ready.value,
                [int(i) for i in dut.sw_inst.data_in.value],
                dut.sw_inst.data_out_valid.value,
                dut.sw_inst.data_out_ready.value,
                [int(i) for i in dut.sw_inst.data_out.value],
            )
        )
    elif instance == "roller":
        logger.debug(
            "wave of roller:\n\
                {},{},data_in = {}\n\
                {},{},data_out = {}\n\
                ".format(
                dut.roller_inst.data_in_valid.value,
                dut.roller_inst.data_in_ready.value,
                [int(i) for i in dut.roller_inst.data_in.value],
                dut.roller_inst.data_out_valid.value,
                dut.roller_inst.data_out_ready.value,
                [int(i) for i in dut.roller_inst.data_out.value],
            )
        )
    else:
        logger.debug(
            "wave of linear:\n\
                {},{},data_in = {}\n\
                {},{},weight = {}\n\
                {},{},bias = {}\n\
                {},{},data_out_linear = {}\n\
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
                dut.fl_instance.data_out_valid.value,
                dut.fl_instance.data_out_ready.value,
                [int(i) for i in dut.fl_instance.data_out.value],
            )
        )


@cocotb.test()
async def test_binary_activation_binary_convolution(dut):
    """Test integer based vector mult"""
    samples = 20
    test_case = VerificationCase(samples=samples)
    print("data_in_hw", test_case.data_in.data)
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
        print("dut.data_in.value {}".format(dut.data_in.value))
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )
        await Timer(1, units="ns")
        debug_state(dut, "Pre-clk")
        wave_check(dut, "sliding_window")
        wave_check(dut, "roller")
        wave_check(dut, "linear")
        # logger.debug(
        #     "wave of interface:\n\
        #         {},{} kernel = {}  \n\
        #         {},{} rolled_k = {}   \n\
        #         padding_x = {} \n\
        #         padding_y = {} \n\
        #         padding_c = {} \n\
        #         ".format(
        #         dut.kernel_valid.value,
        #         dut.kernel_ready.value,
        #         [int(i) for i in dut.kernel.value],
        #         dut.rolled_k_valid.value,
        #         dut.rolled_k_ready.value,
        #         [int(i) for i in dut.rolled_k.value],
        #         int(dut.sw_inst.padding_inst.count_x.value),
        #         int(dut.sw_inst.padding_inst.count_y.value),
        #         int(dut.sw_inst.padding_inst.count_c.value),
        #     )
        # )
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

    check_results_signed(test_case.outputs.data, test_case.ref)


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = [
        "../../../../components/conv/binary_activation_binary_convolution.sv",
        "../../../../components/conv/padding.sv",
        "../../../../components/conv/roller.sv",
        "../../../../components/conv/sliding_window.sv",
        "../../../../components/cast/fixed_cast.sv",
        "../../../../components/cast/integer_cast.sv",
        "../../../../components/linear/binary_activation_binary_linear.sv",
        "../../../../components/binary_arith/binary_activation_binary_dot_product.sv",
        "../../../../components/fixed_arith/fixed_accumulator.sv",
        "../../../../components/binary_arith/binary_activation_binary_vector_mult.sv",
        "../../../../components/binary_arith/binary_activation_binary_adder_tree.sv",
        "../../../../components/binary_arith/binary_activation_binary_adder_tree_layer.sv",
        "../../../../components/binary_arith/binary_activation_binary_mult.sv",
        "../../../../components/common/register_slice.sv",
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
        hdl_toplevel="binary_activation_binary_convolution",
        build_args=extra_args,
    )

    runner.test(
        hdl_toplevel="binary_activation_binary_convolution",
        test_module="binary_activation_binary_convolution_tb",
    )


if __name__ == "__main__":
    runner()
