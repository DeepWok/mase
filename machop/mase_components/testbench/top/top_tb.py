#!/usr/bin/env python3

import math
import os, sys, logging

import cocotb
from cocotb.triggers import Timer
from cocotb.runner import get_runner
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock

import torch
import torch.nn as nn
from Qlinear import QuantizedLinear


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from random_test import RandomSource, RandomSink, check_results_signed
import utils

debug = True

# create a logger client
logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# --------------------------------------------------
#   Model specifications
#   prefer small models for fast test
# --------------------------------------------------
class MLP(torch.nn.Module):
    """
    Toy quantized FC model for digit recognition on MNIST
    """

    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(7 * 7, 7 * 7)
        self.fc2 = nn.Linear(7 * 7, 7 * 7 * 4)
        # self.fc3 = nn.Linear(28 * 28 * 4, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = torch.nn.functional.relu(self.fc1(x))
        # x = torch.nn.functional.relu(self.fc2(x))
        # x = self.fc3(x)
        x = self.fc2(x)
        return x


class VerificationCase:
    def __init__(self, samples=10):
        # For now we are binarizing all the layers
        self.data_in_width = 32
        self.data_in_frac_width = 16

        # Apply for both fc1 and fc2
        self.weight_width = 1
        self.weight_frac_width = 0
        self.has_bias = 0
        self.bias_width = 32
        self.bias_frac_width = 16

        # fc1
        self.in_feature = 49  # 784
        self.out_feature = 49  # 784
        self.vector_size = 1
        self.iterations = 49  # 784
        self.parallelism = 49  # 784

        # fc2
        self.fc2_in_feature = 49  # 784
        self.fc2_out_feature = 196  # 3136
        self.fc2_vector_size = 1
        self.fc2_iterations = 49  # 784
        self.fc2_parallelism = 196  # 3136

        # fc1
        self.data_in = RandomSource(
            name="data_in",
            samples=samples * self.iterations,
            num=self.vector_size,
            max_stalls=2 * samples,
            debug=debug,
            fix_seed=True,
        )
        self.weight = RandomSource(
            name="weight",
            samples=samples * self.iterations,
            num=self.vector_size * self.parallelism,
            max_stalls=2 * samples,
            debug=debug,
            arithmetic="binary",
            fix_seed=True,
        )
        self.bias = RandomSource(
            name="bias",
            samples=samples,
            num=self.parallelism,
            max_stalls=2 * samples,
            debug=debug,
            fix_seed=True,
        )

        # fc2
        self.fc2_weight = RandomSource(
            name="fc2_weight",
            samples=samples * self.fc2_iterations,
            num=self.fc2_vector_size * self.fc2_parallelism,
            max_stalls=2 * samples,
            debug=debug,
            arithmetic="binary",
            fix_seed=True,
        )
        self.fc2_bias = RandomSource(
            name="fc2_bias",
            samples=samples,
            num=self.fc2_parallelism,
            max_stalls=2 * samples,
            debug=debug,
            fix_seed=True,
        )

        self.outputs = RandomSink(samples=samples, max_stalls=2 * samples, debug=debug)
        self.samples = samples
        # self.ref = self.sw_compute() // TODO: This only support sample 1
        self.ref_array = self.sw_compute_array()

    def get_dut_parameters(self):
        return {
            # fc1_width
            "fc1_IN_0_WIDTH": self.data_in_width,
            "fc1_IN_0_FRAC_WIDTH": self.data_in_frac_width,
            "fc1_WEIGHT_WIDTH": self.weight_width,
            "fc1_WEIGHT_FRAC_WIDTH": self.weight_frac_width,
            # self.data_in_width
            # + math.log2(196 * 196)
            # + self.has_bias,  # self.weight_frac_width,
            "fc1_BIAS_WIDTH": self.bias_width,
            "fc1_BIAS_FRAC_WIDTH": self.bias_frac_width,
            # fc1_configuration
            "fc1_HAS_BIAS": 0,  # self.has_bias,
            "fc1_OUT_0_WIDTH": self.data_in_width
            + math.ceil(math.log2(49))
            + 0,  # same with input width
            "fc1_WEIGHT_SIZE": 49,  # 784,
            "fc1_BIAS_SIZE": 49,  # 784,
            "fc1_OUT_0_SIZE": 49,  # 784,
            "fc1_PARALLELISM": 49,  # 784,
            "fc1_IN_0_SIZE": 1,
            "fc1_IN_0_DEPTH": 49,  # 784, # Total cycle needed
            # fc2_width
            "fc2_IN_0_WIDTH": self.data_in_width,
            "fc2_IN_0_FRAC_WIDTH": self.data_in_frac_width,
            "fc2_WEIGHT_WIDTH": self.weight_width,
            "fc2_WEIGHT_FRAC_WIDTH": self.weight_frac_width,
            "fc2_BIAS_WIDTH": self.bias_width,
            "fc2_BIAS_FRAC_WIDTH": self.bias_frac_width,
            # fc2_configuration
            "fc2_HAS_BIAS": 0,  # self.has_bias,
            "fc2_OUT_0_WIDTH": self.data_in_width + math.ceil(math.log2(49)) + 0,
            "fc2_WEIGHT_SIZE": 196,  # All weight: 784 * 196,  # 3136,
            "fc2_BIAS_SIZE": 196,  # 3136,
            "fc2_OUT_0_SIZE": 196,  # 3136,
            "fc2_IN_0_SIZE": 1,  # TODO: This need to be change
            "fc2_PARALLELISM": 196,  # 3136,
            "fc2_IN_0_DEPTH": 49,  # 784,  # Total cycle needed
            # relu configuration
            "relu_IN_0_WIDTH": self.data_in_width,
            "relu_IN_0_FRAC_WIDTH": self.data_in_frac_width,
            "relu_IN_0_SIZE": 49,  # 784,
            "relu_OUT_0_WIDTH": 32,  # same with output_width in the fc1
            "relu_OUT_0_SIZE": 49,  # 784,
        }

    def sw_compute(self):
        """
        Use quantised layer
        """
        ref = []
        output = []
        for i in range(self.samples):
            fc1_weight = torch.where(
                torch.tensor(self.weight.data).type(torch.float) == 0,
                torch.tensor(-1.0),
                torch.tensor(1.0),
            ).T
            fc2_weight = torch.where(
                torch.tensor(self.fc2_weight.data).type(torch.float) == 0,
                torch.tensor(-1.0),
                torch.tensor(1.0),
            ).T
            fc1_bias = torch.tensor(self.bias.data).type(
                torch.float
            )  # TODO: check dimension
            # print("fc1_bias: {}".format(fc1_bias))
            fc2_bias = torch.tensor(self.fc2_bias.data).type(
                torch.float
            )  # TODO: check dimension

            # model
            QLinear = QuantizedLinear(
                in_channels=self.in_feature,
                out_channels=self.out_feature,
                bias=True,
                bias_data=fc1_bias,
                weights=fc1_weight,
                DWidth=32,
                DFWidth=0,
                WWidth=1,
                WFWidth=0,
                BWidth=32,
                BFWidth=0,
            )
            fc2_QLinear = QuantizedLinear(
                in_channels=self.in_feature,
                out_channels=self.out_feature,
                bias=True,
                bias_data=fc2_bias,
                weights=fc2_weight,
                DWidth=32,
                DFWidth=0,
                WWidth=1,
                WFWidth=0,
                BWidth=32,
                BFWidth=0,
            )
            fc1_data_out = QLinear(torch.tensor(self.data_in.data).type(torch.float).T)
            relu1_data_out = torch.nn.functional.relu(fc1_data_out)
            fc2_data_out = fc2_QLinear(relu1_data_out)
            output = fc2_data_out.flatten().type(torch.int).tolist()
            print(
                """
                Shape:
                  fc1_data_in: {},
                  fc1_weight: {},
                  fc1_bias: {},
                  fc2_data_in: {},
                  fc2_weight: {},
                  fc2_bias: {},
            """.format(
                    torch.tensor(self.data_in.data).type(torch.float).T.shape,
                    fc1_weight.shape,
                    fc1_bias.shape,
                    relu1_data_out.shape,
                    fc2_weight.shape,
                    fc2_bias.shape,
                )
            )
            ref.append(output)
        ref.reverse()
        return ref

    def sw_compute_array(self):
        """
        Use array to amulate the calculation
        """
        # mlp = MLP()
        ref = []
        for i in range(self.samples):
            # fc_1
            acc = [0 for _ in range(self.parallelism)]
            for j in range(self.iterations):
                data_idx = i * self.iterations + j
                for k in range(self.parallelism):
                    s = [
                        self.data_in.data[data_idx][h]
                        * utils.binary_decode(
                            self.weight.data[data_idx][k * self.vector_size + h]
                        )
                        for h in range(self.vector_size)
                    ]
                    acc[k] += sum(s)
            # print("acc: {}".format(acc))
            if self.has_bias:
                for k in range(self.parallelism):
                    acc[k] += self.bias.data[i][k] << (
                        self.weight_frac_width
                        + self.data_in_frac_width
                        - self.bias_frac_width
                    )
            # print("acc_bias: {}".format(acc))
            # relu_1 [32][784]
            relu_1_out = [0 if element < 0 else element for element in acc]
            # print("relu output:{} with length:{}".format(relu_1_out, len(relu_1_out)))

            # fc2
            fc2_acc = [0 for _ in range(self.fc2_parallelism)]
            for j in range(self.fc2_iterations):
                data_idx = i * self.fc2_iterations + j
                for k in range(self.fc2_parallelism):
                    # TODO: this is 99% incorrect. fc2_vector_size is been set to 1
                    s = [
                        relu_1_out[j]
                        * utils.binary_decode(
                            self.fc2_weight.data[data_idx][k * self.fc2_vector_size + h]
                        )
                        for h in range(self.fc2_vector_size)
                    ]
                    fc2_acc[k] += sum(s)
            if self.has_bias:
                for k in range(self.fc2_parallelism):
                    fc2_acc[k] += self.fc2_bias.data[i][k] << (
                        self.weight_frac_width
                        + self.data_in_frac_width
                        - self.bias_frac_width
                    )

            # print("fc2 output:{} with length:{}".format(fc2_acc, len(fc2_acc)))
            ref.append(fc2_acc)
        ref.reverse()
        return ref


def debug_state(dut, state):
    logger.debug(
        """{} State: (
            fc1_bias_ready,
            fc1_bias_valid,
            fc1_weight_ready,
            fc1_weight_valid,
            fc2_bias_ready,
            fc2_bias_valid,
            fc2_weight_ready,
            fc2_weight_valid,
            data_in_ready,
            data_in_valid,
            data_out_ready,
            data_out_valid
            ) = ({},{},{},{},{},{},{},{},{},{},{},{})""".format(
            state,
            dut.fc1_bias_ready.value,
            dut.fc1_bias_valid.value,
            dut.fc1_weight_ready.value,
            dut.fc1_weight_valid.value,
            dut.fc2_bias_ready.value,
            dut.fc2_bias_valid.value,
            dut.fc2_weight_ready.value,
            dut.fc2_weight_valid.value,
            dut.data_in_ready.value,
            dut.data_in_valid.value,
            dut.data_out_ready.value,
            dut.data_out_valid.value,
        )
    )


@cocotb.test()
async def test_binary_toy_model(dut):
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
    dut.data_in_valid.value = 0

    dut.fc1_weight_valid.value = 0
    dut.fc1_bias_valid.value = 0

    dut.fc2_weight_valid.value = 0
    dut.fc2_bias_valid.value = 0

    dut.data_out_ready.value = 1

    debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    debug_state(dut, "Post-clk")
    debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    debug_state(dut, "Post-clk")

    done = False
    for i in range(samples * 1000):
        await FallingEdge(dut.clk)
        debug_state(dut, "Post-clk")
        dut.data_in_valid.value = test_case.data_in.pre_compute()
        dut.fc1_weight_valid.value = test_case.weight.pre_compute()
        dut.fc1_bias_valid.value = test_case.bias.pre_compute()
        dut.fc2_weight_valid.value = test_case.fc2_weight.pre_compute()
        dut.fc2_bias_valid.value = test_case.fc2_bias.pre_compute()

        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(
            dut.data_out_valid.value
        )
        await Timer(1, units="ns")
        debug_state(dut, "Post-clk")

        dut.data_in_valid.value, dut.data_in.value = test_case.data_in.compute(
            dut.data_in_ready.value
        )
        dut.fc1_weight_valid.value, dut.fc1_weight.value = test_case.weight.compute(
            dut.fc1_weight_ready.value
        )
        dut.fc1_bias_valid.value, dut.fc1_bias.value = test_case.bias.compute(
            dut.fc1_bias_ready.value
        )
        dut.fc2_bias_valid.value, dut.fc2_bias.value = test_case.fc2_bias.compute(
            dut.fc2_bias_ready.value
        )
        dut.fc2_weight_valid.value, dut.fc2_weight.value = test_case.fc2_weight.compute(
            dut.fc2_weight_ready.value
        )
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )

        await Timer(1, units="ns")
        debug_state(dut, "Pre-clk")

        if (
            (test_case.bias.is_empty())
            and test_case.weight.is_empty()
            and test_case.data_in.is_empty()
            and test_case.outputs.is_full()
        ):
            done = True
            break
    print(len(test_case.outputs.data))
    assert (
        done
    ), "Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)"
    check_results_signed(test_case.outputs.data, test_case.ref_array)
    """ """


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = [
        "../../../components/testbench/top/top.sv",
        "../../../components/conv/roller.sv",
        "../../../components/cast/fixed_cast.sv",
        "../../../components/rename/fixed_activation_binary_linear.sv",
        "../../../components/rename/fixed_relu.sv",
        "../../../components/binary_arith/fixed_activation_binary_dot_product.sv",
        "../../../components/fixed_arithmetic/fixed_accumulator.sv",
        "../../../components/binary_arith/fixed_activation_binary_vector_mult.sv",
        "../../../components/fixed_arithmetic/fixed_adder_tree.sv",
        "../../../components/fixed_arithmetic/fixed_adder_tree_layer.sv",
        "../../../components/binary_arith/fixed_activation_binary_mult.sv",
        "../../../components/common/register_slice.sv",
        "../../../components/common/join2.sv",
    ]

    test_case = VerificationCase()
    extra_args = []
    for k, v in test_case.get_dut_parameters().items():
        extra_args.append(f"-G{k}={v}")
    print(extra_args)

    runner = get_runner(sim)
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="top",
        build_args=extra_args,
    )
    runner.test(
        hdl_toplevel="top",
        test_module="top_tb",
    )
    # assert test_case.ref == test_case.ref_array


if __name__ == "__main__":
    runner()
