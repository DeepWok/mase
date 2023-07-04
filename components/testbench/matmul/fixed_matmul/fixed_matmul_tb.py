#!/usr/bin/env python3

# This script tests the fixed point linear
import random, os, math, logging, sys
import numpy as np

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
    def __init__(self, samples=1):
        self.samples = samples
        self.data_in_width = 32
        self.data_in_frac_width = 1
        self.weight_width = 16
        self.weight_frac_width = 1
        self.data_out_width = 32
        self.data_out_frac_width = 1

        self.in_parallelism = 4
        self.in_num_parallelism = 3

        self.in_size = 2
        self.weight_size = self.in_size

        self.w_parallelism = 5
        self.w_num_parallelism = 3
        self.weight_columns = self.w_parallelism * self.w_num_parallelism
        self.iterations = 4
        self.data_in = RandomSource(
            name="data_in",
            samples=samples * self.iterations * self.in_num_parallelism,
            num=self.in_parallelism * self.in_size,
            max_stalls=2 * samples,
            debug=debug,
        )
        self.weight = RandomSource(
            name="weight",
            samples=samples * self.iterations * self.w_num_parallelism,
            num=self.w_parallelism * self.weight_size,
            max_stalls=2 * samples,
            debug=debug,
        )

        self.outputs = RandomSink(
            samples=samples * self.in_num_parallelism * self.w_num_parallelism,
            max_stalls=2 * samples,
            debug=debug,
        )
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
            "IN1_PARALLELISM": self.in_parallelism,
            "IN1_NUM_PARALLELISM": self.in_num_parallelism,
            "IN2_PARALLELISM": self.w_parallelism,
            "IN2_NUM_PARALLELISM": self.w_num_parallelism,
            "IN_SIZE": self.in_size,
            "IN_DEPTH": self.iterations,
        }

    def sw_compute(self):
        all_in_matrix = self.data_in.data
        all_weight_matrix = self.weight.data

        ref = []
        for i in range(self.samples):
            for j in range(self.in_num_parallelism):
                for k in range(self.w_num_parallelism):
                    list_attach = []
                    for t in range(self.iterations):
                        # get each in
                        _in = all_in_matrix[
                            (i * self.in_num_parallelism + j) * self.iterations + t
                        ]
                        # let(_in = [in_parallelism * in_size -1 : 0]  to list_in = [in_parallelism -1 :0][in_size - 1:0])
                        list_in = []
                        for s in range(self.in_parallelism):
                            list_in.append(
                                [_in[s * self.in_size + m] for m in range(self.in_size)]
                            )
                        # concat list_in to list_attach = [in_parallelism -1 :0][in_size * iterations - 1:0]
                        list_attach = list_attach + list_in
                    rlist_in = []
                    for t in range(self.in_parallelism):
                        iteration_add = []
                        for m in range(self.iterations):
                            iteration_add = (
                                iteration_add + list_attach[m * self.in_parallelism + t]
                            )
                        rlist_in.append(iteration_add)

                    array_in = np.array(rlist_in).reshape(
                        self.in_parallelism,
                        self.iterations * self.in_size,
                    )
                    w_attach = []
                    for t in range(self.iterations):
                        _w = all_weight_matrix[
                            (i * self.w_num_parallelism + k) * self.iterations + t
                        ]
                        list_w = []
                        for s in range(self.w_parallelism):
                            list_w.append(
                                [_w[s * self.in_size + m] for m in range(self.in_size)]
                            )
                        w_attach = w_attach + list_w
                    rlist_w = []
                    for t in range(self.w_parallelism):
                        iteration_add = []
                        for m in range(self.iterations):
                            iteration_add = (
                                iteration_add + w_attach[m * self.w_parallelism + t]
                            )
                            # breakpoint()
                        rlist_w.append(iteration_add)

                    array_w = np.array(rlist_w).reshape(
                        self.w_parallelism,
                        self.iterations * self.weight_size,
                    )

                    result = np.matmul(array_in, array_w.T).reshape(-1)
                    ref.append(result.tolist())
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


# Check if an is_impossible state is reached
def is_impossible_state(
    data_in2_ready,
    data_in2_valid,
    data_in1_ready,
    data_in1_valid,
    data_out_ready,
    data_out_valid,
):
    return False


def debug_state(dut, state):
    logger.debug(
        "{} State: (data_in2_ready,data_in2_valid,data_in1_ready,data_in1_valid,data_out_ready,data_out_valid) = ({},{},{},{},{},{})".format(
            state,
            dut.data_in2_ready.value,
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
    # breakpoint()
    logger.debug(
        "data_in = {}\n\
        weight = {}\n\
        ".format(
            [int(i[0]) for i in test_case.data_in.data],
            [int(i[0]) for i in test_case.weight.data],
        )
    )
    for i in range(samples * 80):
        await FallingEdge(dut.clk)
        debug_state(dut, "Post-clk")
        dut.data_in2_valid.value = test_case.weight.pre_compute()
        dut.data_in1_valid.value = test_case.data_in.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(dut.data_out_valid)
        await Timer(1, units="ns")
        # start input data
        #
        dut.data_in2_valid.value, dut.data_in2.value = test_case.weight.compute(
            dut.data_in2_ready.value
        )
        await Timer(1, units="ns")
        dut.data_in1_valid.value, dut.data_in1.value = test_case.data_in.compute(
            dut.data_in1_ready.value
        )

        await Timer(1, units="ns")
        # breakpoint()

        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )
        await Timer(1, units="ns")
        debug_state(dut, "Pre-clk")
        logger.debug(
            "ib_data_in = {}\n\
            ib_data_in_valid = {}\n\
            ib_data_in_ready = {}\n\
            ib_weight = {}\n\
            ib_weight_valid = {}\n\
            ib_weight_ready = {}\n\
            ".format(
                [int(i) for i in dut.ib_data_in.value],
                dut.ib_data_in_valid.value,
                dut.ib_data_in_ready.value,
                [int(i) for i in dut.ib_weight.value],
                dut.ib_weight_valid.value,
                dut.ib_weight_ready.value,
            )
        )
        # breakpoint()
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
        "../../../../components/matmul/fixed_matmul.sv",
        "../../../../components/common/input_buffer.sv",
        "../../../../components/common/ram_block.sv",
        "../../../../components/common/register_slice.sv",
        "../../../../components/common/join2.sv",
        "../../../../components/linear/fixed_linear.sv",
        "../../../../components/cast/fixed_cast.sv",
        "../../../../components/fixed_arith/fixed_matmul_core.sv",
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
        hdl_toplevel="fixed_matmul",
        build_args=extra_args,
    )
    for _ in range(1):
        runner.test(
            hdl_toplevel="fixed_matmul",
            test_module="fixed_matmul_tb",
        )


if __name__ == "__main__":
    runner()
