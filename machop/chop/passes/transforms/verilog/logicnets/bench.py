#  Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from functools import reduce


def generate_lut_bench(input_fanin_bits, output_bits, lut_string):
    lut_neuron_template = """\
{input_string}\
{output_string}\
{lut_string}"""
    input_string = ""
    for i in range(input_fanin_bits):
        input_string += f"INPUT(M0[{i}])\n"
    output_string = ""
    for i in range(output_bits):
        output_string += f"OUTPUT(M1[{i}])\n"
    return lut_neuron_template.format(
        input_string=input_string, output_string=output_string, lut_string=lut_string
    )


def generate_lut_input_string(input_fanin_bits):
    lut_input_string = ""
    for i in range(input_fanin_bits):
        if i == 0:
            lut_input_string += f"( M0[{i}]"
        elif i == input_fanin_bits - 1:
            lut_input_string += f", M0[{i}] )\n"
        else:
            lut_input_string += f", M0[{i}]"
    return lut_input_string


def sort_to_bench(input_state_space_bin_str, bin_output_states):
    sorted_bin_output_states = bin_output_states.tolist()
    input_state_space_flat_int = list(
        map(lambda l: int(reduce(lambda a, b: a + b, l), 2), input_state_space_bin_str)
    )
    zipped_io_states = list(zip(input_state_space_flat_int, sorted_bin_output_states))
    zipped_io_states.sort(key=lambda x: x[0], reverse=True)
    sorted_bin_output_states = list(map(lambda x: x[1], zipped_io_states))
    return sorted_bin_output_states
