import logging
import os
import shutil
import torch.nn as nn

from chop.passes.utils import init_project
from chop.passes.transforms.quantize.quantized_modules.linear import LinearLogicNets
from ..internal_file_dependences import INTERNAL_RTL_DEPENDENCIES

from .util import (
    generate_lut_verilog,
    generate_neuron_connection_verilog,
    layer_connection_verilog,
    generate_logicnets_verilog,
    generate_register_verilog,
    get_bin_str,
)

from .bench import generate_lut_bench, generate_lut_input_string, sort_to_bench

logger = logging.getLogger(__name__)


class LogicNetsLinearVerilog:
    def __init__(self, logicnets_linear: nn.Module) -> None:
        super(LogicNetsLinearVerilog, self).__init__()
        self.module = logicnets_linear

    # TODO: Move the verilog string templates to elsewhere
    # TODO: Update this code to support custom bitwidths per input/output
    def gen_layer_verilog(self, module_prefix, directory, generate_bench: bool = False):
        # TODO: Implement input_quant and output_quant!
        # _, input_bitwidth = self.module.input_quant.get_scale_factor_bits()
        # _, output_bitwidth = self.module.output_quant.get_scale_factor_bits()
        # input_bitwidth, output_bitwidth = int(input_bitwidth), int(output_bitwidth)
        input_bitwidth, output_bitwidth = self.module.x_width, self.module.y_width
        total_input_bits = self.module.in_features * input_bitwidth
        total_output_bits = self.module.out_features * output_bitwidth
        layer_contents = f"module {module_prefix} (input [{total_input_bits-1}:0] M0, output [{total_output_bits-1}:0] M1);\n\n"
        output_offset = 0
        for index in range(self.module.out_features):
            module_name = f"{module_prefix}_N{index}"
            indices, _, _ = self.module.neuron_truth_tables[index]
            neuron_verilog = self.gen_neuron_verilog(
                index, module_name
            )  # Generate the contents of the neuron verilog
            with open(f"{directory}/{module_name}.v", "w") as f:
                f.write(neuron_verilog)
            if generate_bench:
                neuron_bench = self.module.gen_neuron_bench(
                    index, module_name
                )  # Generate the contents of the neuron verilog
                with open(f"{directory}/{module_name}.bench", "w") as f:
                    f.write(neuron_bench)
            connection_string = generate_neuron_connection_verilog(
                indices, input_bitwidth
            )  # Generate the string which connects the synapses to this neuron
            wire_name = f"{module_name}_wire"
            connection_line = f"wire [{len(indices)*input_bitwidth-1}:0] {wire_name} = {{{connection_string}}};\n"
            inst_line = f"{module_name} {module_name}_inst (.M0({wire_name}), .M1(M1[{output_offset+output_bitwidth-1}:{output_offset}]));\n\n"
            layer_contents += connection_line + inst_line
            output_offset += output_bitwidth
        layer_contents += "endmodule"
        with open(f"{directory}/{module_prefix}.v", "w") as f:
            f.write(layer_contents)
        return total_input_bits, total_output_bits

    # TODO: Move the verilog string templates to elsewhere
    def gen_neuron_verilog(self, index, module_name):
        indices, input_perm_matrix, bin_output_states = self.module.neuron_truth_tables[
            index
        ]
        # TODO: Implement input_quant and output_quant!
        # _, input_bitwidth = self.module.input_quant.get_scale_factor_bits()
        # _, output_bitwidth = self.module.output_quant.get_scale_factor_bits()
        input_bitwidth, output_bitwidth = self.module.x_width, self.module.y_width
        cat_input_bitwidth = len(indices) * input_bitwidth
        lut_string = ""
        num_entries = input_perm_matrix.shape[0]
        for i in range(num_entries):
            entry_str = ""
            for idx in range(len(indices)):
                val = input_perm_matrix[i, idx]
                # TODO: Implement input_quant and output_quant!
                # entry_str += self.input_quant.get_bin_str(val)
                entry_str += get_bin_str(val, input_bitwidth)

            # TODO: Implement input_quant and output_quant!
            # res_str = self.module.output_quant.get_bin_str(bin_output_states[i])
            res_str = get_bin_str(bin_output_states[i], output_bitwidth)
            lut_string += f"\t\t\t{int(cat_input_bitwidth)}'b{entry_str}: M1r = {int(output_bitwidth)}'b{res_str};\n"
        return generate_lut_verilog(
            module_name, int(cat_input_bitwidth), int(output_bitwidth), lut_string
        )

    # # TODO: Move the string templates to bench.py
    # def gen_neuron_bench(self, index, module_name):
    #     indices, input_perm_matrix, bin_output_states = self.module.neuron_truth_tables[index]
    #     _, input_bitwidth = self.module.input_quant.get_scale_factor_bits()
    #     _, output_bitwidth = self.module.output_quant.get_scale_factor_bits()
    #     cat_input_bitwidth = len(indices)*input_bitwidth
    #     lut_string = ""
    #     num_entries = input_perm_matrix.shape[0]
    #     # Sort the input_perm_matrix to match the bench format
    #     input_state_space_bin_str = list(map(lambda y: list(map(lambda z: self.module.input_quant.get_bin_str(z), y)), input_perm_matrix))
    #     sorted_bin_output_states = sort_to_bench(input_state_space_bin_str, bin_output_states)
    #     # Generate the LUT for each output
    #     for i in range(int(output_bitwidth)):
    #         lut_string += f"M1[{i}]       = LUT 0x"
    #         output_bin_str = reduce(lambda b,c: b+c, map(lambda a: self.module.output_quant.get_bin_str(a)[int(output_bitwidth)-1-i], sorted_bin_output_states))
    #         lut_hex_string = f"{int(output_bin_str,2):0{int(num_entries/4)}x} "
    #         lut_string += lut_hex_string
    #         lut_string += generate_lut_input_string(int(cat_input_bitwidth))
    #     return generate_lut_bench(int(cat_input_bitwidth), int(output_bitwidth), lut_string)
