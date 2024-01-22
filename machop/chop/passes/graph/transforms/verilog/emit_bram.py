import logging
import math
import os
import struct
import time

import torch

from chop.passes.graph.utils import vf, v2p, get_module_by_name, init_project
from chop.passes.graph.transforms.quantize.quantizers import integer_quantizer_for_hw

logger = logging.getLogger(__name__)
from pathlib import Path


def iceil(x):
    return int(math.ceil(x))


def clog2(x):
    return iceil(math.log2(x))


def _cap(name):
    """
    capitalize a string
    """
    return str(name).upper()


def emit_parameters_in_mem_internal(node, param_name, file_name, data_name):
    """
    Emit single-port ROM hardware components for each parameter
    (Mostly because Vivado does not support string type parameters...)
    """

    # TODO: Force bias to have a depth of 1 for now
    if param_name != "bias":
        # out_depth = node.meta["mase"].parameters["hardware"]["verilog_param"][
        #     "DATA_IN_0_DEPTH"
        # ]
        out_depth = 1
    else:
        out_depth = 1
    addr_width = clog2(out_depth) + 1
    total_size = math.prod(
        node.meta["mase"].parameters["common"]["args"][param_name]["shape"]
    )
    # The depth of parameters must match with the input depth
    assert (
        total_size % out_depth == 0
    ), f"Cannot partition imperfect size for now {node.name}.{param_name} = {total_size} / {out_depth}."
    out_size = iceil(total_size / out_depth)
    # Assume the first index is the total width
    out_width = node.meta["mase"].parameters["common"]["args"][param_name]["precision"][
        0
    ]

    node_name = vf(node.name)
    node_param_name = f"{node_name}_{param_name}"
    time_to_emit = time.strftime("%d/%m/%Y %H:%M:%S")

    rom_verilog = f"""
// =====================================
//     Mase Hardware
//     Parameter: {node_param_name}
//     {time_to_emit}
// =====================================

`timescale 1 ns / 1 ps
module {node_param_name}_rom #(
  parameter DWIDTH = {out_size*out_width},
  parameter MEM_SIZE = {out_depth},
  parameter AWIDTH = $clog2(MEM_SIZE) + 1
) (
    input clk,
    input logic [AWIDTH-1:0] addr0,
    input ce0,
    output logic [DWIDTH-1:0] q0
);

  logic [DWIDTH-1:0] ram[0:MEM_SIZE-1];
  logic [DWIDTH-1:0] q0_t0;
  logic [DWIDTH-1:0] q0_t1;

  initial begin
    $readmemh("{data_name}", ram);
  end

  assign q0 = q0_t1;

  always_ff @(posedge clk) if (ce0) q0_t1 <= q0_t0;
  always_ff @(posedge clk) if (ce0) q0_t0 <= ram[addr0];

endmodule

`timescale 1 ns / 1 ps
module {node_param_name} #(
  parameter DATA_WIDTH = 32'd{out_width*out_size},
  parameter ADDR_RANGE = 32'd{out_depth},
  parameter ADDR_WIDTH = $clog2(ADDR_RANGE) + 1
) (
  input reset,
  input clk,
  input logic [ADDR_WIDTH - 1:0] address0,
  input ce0,
  output logic [DATA_WIDTH - 1:0] q0
);

  {node_param_name}_rom {node_param_name}_rom_U (
      .clk(clk),
      .addr0(address0),
      .ce0(ce0),
      .q0(q0)
  );

endmodule


`timescale 1ns / 1ps
module {node_param_name}_source #(
    parameter {_cap(param_name)}_TENSOR_SIZE_DIM_0  = 32,
    parameter {_cap(param_name)}_TENSOR_SIZE_DIM_1  = 1,
    parameter {_cap(param_name)}_PRECISION_0 = 16,

    parameter {_cap(param_name)}_PARALLELISM_DIM_0 = 1,
    parameter {_cap(param_name)}_PARALLELISM_DIM_1 = 1,
    parameter OUT_DEPTH = {_cap(param_name)}_TENSOR_SIZE_DIM_0 / {_cap(param_name)}_PARALLELISM_DIM_0
) (
    input clk,
    input rst,

    output logic [{_cap(param_name)}_PRECISION_0-1:0] data_out      [{_cap(param_name)}_PARALLELISM_DIM_0 * {_cap(param_name)}_PARALLELISM_DIM_1-1:0],
    output                       data_out_valid,
    input                        data_out_ready
);
  // 1-bit wider so IN_DEPTH also fits.
  localparam COUNTER_WIDTH = $clog2(OUT_DEPTH);
  logic [COUNTER_WIDTH:0] counter;

  always_ff @(posedge clk)
    if (rst) counter <= 0;
    else begin
      if (data_out_ready) begin
        if (counter == OUT_DEPTH - 1) counter <= 0;
        else counter <= counter + 1;
      end
    end

  logic ce0;
  assign ce0 = 1;

  logic [{_cap(param_name)}_PRECISION_0*{_cap(param_name)}_TENSOR_SIZE_DIM_0-1:0] data_vector;
  {node_param_name} #(
      .DATA_WIDTH({_cap(param_name)}_PRECISION_0 * {_cap(param_name)}_TENSOR_SIZE_DIM_0),
      .ADDR_RANGE(OUT_DEPTH)
  ) {node_param_name}_mem (
      .clk(clk),
      .reset(rst),
      .address0(counter),
      .ce0(ce0),
      .q0(data_vector)
  );

  // Cocotb/verilator does not support array flattening, so
  // we need to manually add some reshaping process.
  for (genvar j = 0; j < {_cap(param_name)}_TENSOR_SIZE_DIM_0; j++)
    assign data_out[j] = data_vector[{_cap(param_name)}_PRECISION_0*j+{_cap(param_name)}_PRECISION_0-1:{_cap(param_name)}_PRECISION_0*j];

  assign data_out_valid = 1;

endmodule
"""

    with open(file_name, "w", encoding="utf-8") as outf:
        outf.write(rom_verilog)
    logger.debug(f"ROM module {param_name} successfully written into {file_name}")
    assert os.path.isfile(file_name), "ROM Verilog generation failed."
    os.system(f"verible-verilog-format --inplace {file_name}")


def emit_parameters_in_dat_internal(node, param_name, file_name):
    """
    Emit initialised data for the ROM block. Each element must be in 8 HEX digits.
    """
    total_size = math.prod(
        node.meta["mase"].parameters["common"]["args"][param_name]["shape"]
    )

    if "IN_DEPTH" in node.meta["mase"].parameters["hardware"]["verilog_param"].keys():
        if param_name == "bias":
            out_depth = 1
        else:
            out_depth = node.meta["mase"].parameters["hardware"]["verilog_param"][
                "IN_DEPTH"
            ]
    else:
        out_depth = total_size

    out_size = iceil(total_size / out_depth)
    # The depth of parameters must match with the input depth of data
    assert (
        total_size % out_depth == 0
    ), f"Cannot partition imperfect size for now {node.name}.{param_name} = {total_size} / {out_depth}."
    # Assume the first index is the total width
    out_width = node.meta["mase"].parameters["common"]["args"][param_name]["precision"][
        0
    ]

    data_buff = ""
    param_data = node.meta["mase"].module.get_parameter(param_name).data
    if node.meta["mase"].parameters["hardware"]["interface"][param_name]["transpose"]:
        param_data = torch.reshape(
            param_data,
            (
                node.meta["mase"].parameters["hardware"]["verilog_param"][
                    "DATA_OUT_0_SIZE"
                ],
                node.meta["mase"].parameters["hardware"]["verilog_param"][
                    "DATA_IN_0_DEPTH"
                ],
                node.meta["mase"].parameters["hardware"]["verilog_param"][
                    "DATA_IN_0_SIZE"
                ],
            ),
        )
        param_data = torch.transpose(param_data, 0, 1)
    param_data = torch.flatten(param_data).tolist()

    if node.meta["mase"].parameters["common"]["args"][param_name]["type"] == "fixed":
        width = node.meta["mase"].parameters["common"]["args"][param_name]["precision"][
            0
        ]
        frac_width = node.meta["mase"].parameters["common"]["args"][param_name][
            "precision"
        ][1]

        scale = 2**frac_width
        thresh = 2**width
        for i in range(0, out_depth):
            line_buff = ""
            for j in range(0, out_size):
                value = param_data[i * out_size + out_size - 1 - j]
                value = integer_quantizer_for_hw(
                    torch.tensor(value), width, frac_width
                ).item()
                value = str(bin(int(value * scale) % thresh))
                value_bits = value[value.find("0b") + 2 :]
                value_bits = "0" * (width - len(value_bits)) + value_bits
                assert len(value_bits) == width
                line_buff += value_bits

            hex_buff = hex(int(line_buff, 2))
            data_buff += hex_buff[hex_buff.find("0x") + 2 :] + "\n"
    else:
        assert False, "Emitting non-fixed parameters is not supported."

    with open(file_name, "w", encoding="utf-8") as outf:
        outf.write(data_buff)
    logger.debug(f"Init data {param_name} successfully written into {file_name}")
    assert os.path.isfile(file_name), "ROM data generation failed."


def emit_parameters_in_dat_hls(node, param_name, file_name):
    """
    Emit initialised data for the ROM block. Each element must be in 8 HEX digits.
    """
    total_size = math.prod(
        node.meta["mase"].parameters["common"]["args"][param_name]["shape"]
    )
    out_depth = total_size
    out_size = 1
    out_width = node.meta["mase"].parameters["hardware"]["verilog_param"][
        "{}_WIDTH".format(param_name.upper())
    ]

    data_buff = ""
    param_data = node.meta["mase"].module.get_parameter(param_name).data
    param_data = torch.flatten(param_data).tolist()

    if node.meta["mase"].parameters["common"]["args"][param_name]["type"] == "fixed":
        width = node.meta["mase"].parameters["common"]["args"][param_name]["precision"][
            0
        ]
        frac_width = node.meta["mase"].parameters["common"]["args"][param_name][
            "precision"
        ][1]

        scale = 2**frac_width
        thresh = 2**width
        for i in range(0, out_depth):
            line_buff = ""
            for j in range(0, out_size):
                value = param_data[i * out_size + out_size - 1 - j]
                value = integer_quantizer_for_hw(
                    torch.tensor(value), width, frac_width
                ).item()
                value = str(bin(int(value * scale) % thresh))
                value_bits = value[value.find("0b") + 2 :]
                value_bits = "0" * (width - len(value_bits)) + value_bits
                assert len(value_bits) == width
                line_buff += value_bits

            hex_buff = hex(int(line_buff, 2))
            data_buff += hex_buff[hex_buff.find("0x") + 2 :] + "\n"
    elif node.meta["mase"].parameters["common"]["args"][param_name]["type"] == "float":
        width = node.meta["mase"].parameters["common"]["args"][param_name]["precision"][
            0
        ]
        assert width == 32, "Only float32 is supported for now."

        for i in range(0, out_depth):
            line_buff = ""
            value = param_data[i]
            hex_buff = hex(struct.unpack("<I", struct.pack("<f", value))[0])
            # Double will then be:
            # hex(struct.unpack('<Q', struct.pack('<d', value))[0])
            data_buff += hex_buff[hex_buff.find("0x") + 2 :] + "\n"
    else:
        assert False, "Emitting unknown type of parameters is not supported."

    with open(file_name, "w", encoding="utf-8") as outf:
        outf.write(data_buff)
    logger.debug(f"Init data {param_name} successfully written into {file_name}")
    assert os.path.isfile(file_name), "ROM data generation failed."


def emit_bram_handshake(node, rtl_dir):
    """
    Enumerate input parameters of the internal node and emit a ROM block
    with handshake interface for each parameter
    """
    node_name = vf(node.name)
    for param_name, parameter in node.meta["mase"].module.named_parameters():
        if (
            node.meta["mase"].parameters["hardware"]["interface"][param_name]["storage"]
            == "BRAM"
        ):
            logger.debug(
                f"Emitting DAT file for node: {node_name}, parameter: {param_name}"
            )
            verilog_name = os.path.join(rtl_dir, f"{node_name}_{param_name}_source.sv")
            data_name = os.path.join(rtl_dir, f"{node_name}_{param_name}_rom.dat")
            emit_parameters_in_mem_internal(node, param_name, verilog_name, data_name)
            emit_parameters_in_dat_internal(node, param_name, data_name)
        else:
            assert False, "Emtting parameters in non-BRAM hardware is not supported."


def emit_parameters_in_mem_hls(node, param_name, file_name, data_name):
    """
    Emit single-port ROM hardware components for each parameter
    (Mostly because Vivado does not support string type parameters...)
    """

    # The depth of parameters matches with the input depth
    total_size = math.prod(
        node.meta["mase"].parameters["common"]["args"][param_name]["shape"]
    )
    out_depth = total_size
    addr_width = clog2(out_depth) + 1
    total_size = math.prod(
        node.meta["mase"].parameters["common"]["args"][param_name]["shape"]
    )
    out_size = iceil(total_size / out_depth)
    assert (
        total_size % out_depth == 0
    ), f"Cannot partition imperfect size for now = {total_size} / {out_depth}."
    # Assume the first index is the total width
    out_width = node.meta["mase"].parameters["hardware"]["verilog_param"][
        "{}_WIDTH".format(param_name.upper())
    ]

    node_name = vf(node.name)
    node_param_name = f"{node_name}_{param_name}"
    time_to_emit = time.strftime("%d/%m/%Y %H:%M:%S")

    rom_verilog = f"""
// =====================================
//     Mase Hardware
//     Parameter: {node_param_name}
//     {time_to_emit}
// =====================================

`timescale 1 ns / 1 ps
module {node_param_name}_rom #(
  parameter DWIDTH = {out_size*out_width},
  parameter MEM_SIZE = {out_depth},
  parameter AWIDTH = $clog2(MEM_SIZE) + 1
) (
    input clk,
    input logic [AWIDTH-1:0] addr0,
    input ce0,
    output logic [DWIDTH-1:0] q0
);

  logic [DWIDTH-1:0] ram[0:MEM_SIZE-1];
  logic [DWIDTH-1:0] q0_t0;
  logic [DWIDTH-1:0] q0_t1;

  initial begin
    $readmemh("{data_name}", ram);
  end

  assign q0 = q0_t1;

  always_ff @(posedge clk) if (ce0) q0_t1 <= q0_t0;
  always_ff @(posedge clk) if (ce0) q0_t0 <= ram[addr0];

endmodule

`timescale 1 ns / 1 ps
module {node_param_name}_source #(
  parameter DATA_WIDTH = 32'd{out_width*out_size},
  parameter ADDR_RANGE = 32'd{out_depth},
  parameter ADDR_WIDTH = $clog2(ADDR_RANGE) + 1
) (
  input reset,
  input clk,
  input logic [ADDR_WIDTH - 1:0] address0,
  input ce0,
  output logic [DATA_WIDTH - 1:0] q0
);

  {node_param_name}_rom {node_param_name}_rom_U (
      .clk(clk),
      .addr0(address0),
      .ce0(ce0),
      .q0(q0)
  );

endmodule
"""

    with open(file_name, "w", encoding="utf-8") as outf:
        outf.write(rom_verilog)
    logger.debug(f"ROM module {param_name} successfully written into {file_name}")
    assert os.path.isfile(file_name), "ROM Verilog generation failed."
    os.system(f"verible-verilog-format --inplace {file_name}")


def emit_bram_hls(node, rtl_dir):
    """
    Enumerate input parameters of the hls node and emit a ROM block
    with handshake interface for each parameter
    """
    node_name = vf(node.name)
    for param_name, parameter in node.meta["mase"].module.named_parameters():
        if (
            node.meta["mase"].parameters["hardware"]["interface"][param_name]["storage"]
            == "BRAM"
        ):
            # Verilog code of the ROM has been emitted using mlir passes
            verilog_name = os.path.join(rtl_dir, f"{node_name}_{param_name}.sv")
            data_name = os.path.join(rtl_dir, f"{node_name}_{param_name}_rom.dat")
            emit_parameters_in_mem_hls(node, param_name, verilog_name, data_name)
            emit_parameters_in_dat_hls(node, param_name, data_name)
        else:
            assert False, "Emtting parameters in non-BRAM hardware is not supported."


def emit_bram_transform_pass(graph, pass_args={}):
    """
    Enumerate input parameters of the node and emit a ROM block with
    handshake interface for each parameter


    :param graph: a MaseGraph
    :type graph: MaseGraph
    :param pass_args: this pass requires additional arguments which is explained below, defaults to {}
    :type pass_args: _type_, optional
    :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
    :rtype: tuple(MaseGraph, Dict)


    - pass_args
        - project_dir -> str : the directory of the project for cosimulation
        - top_name -> str : name of the top module
    """

    logger.info("Emitting BRAM...")
    project_dir = (
        pass_args["project_dir"]
        if "project_dir" in pass_args.keys()
        else Path.home() / ".mase" / "top"
    )
    top_name = pass_args["top_name"] if "top_name" in pass_args.keys() else "top"

    init_project(project_dir)
    rtl_dir = os.path.join(project_dir, "hardware", "rtl")

    for node in graph.fx_graph.nodes:
        if node.meta["mase"].parameters["hardware"]["is_implicit"]:
            continue
        # Only modules have internal parameters
        if node.meta["mase"].module is None:
            continue
        if "INTERNAL" in node.meta["mase"].parameters["hardware"]["toolchain"]:
            emit_bram_handshake(node, rtl_dir)
        elif "MLIR_HLS" in node.meta["mase"].parameters["hardware"]["toolchain"]:
            emit_bram_hls(node, rtl_dir)

    return graph, None
