import logging
from typing import Tuple, Dict
import math
import os
import time
from multiprocessing import Process, Queue

import torch.fx as fx
from chop.passes.graph.utils import vf, v2p, init_project
import mase_components.helper.generate_memory as gen_lut
import torch.nn as nn
logger = logging.getLogger(__name__)
from chop.nn.quantized.modules.layer_norm import LayerNormIntegerFloor
from chop.nn.quantized.modules.attention import ViTAttentionInteger
from pathlib import Path
from chop.passes.graph.transforms.verilog.emit_top import _cap, _remove_last_comma, get_verilog_parameters, VerilogParameterEmitter, VerilogEmitter

def emit_folded_bram(folded_gragh, stream_name, folded_name, reuse_times):
    def _emit_module_parameters_top_internal(key, node, stream_name, folded_name, reuse_times):
        node_name = vf(node.name)
        component_name = f"{node_name}_{key}_source"
        component_name_inst = f"{component_name}_0"

        # verilog_param = node_name+"_"+_cap(key)
        def get_image_depth(key, param_list, node_name):
            if "weight" in key:
                image_depth = (
                    param_list[f"{_cap(key)}_TENSOR_SIZE_DIM_0"]
                    * param_list[f"{_cap(key)}_TENSOR_SIZE_DIM_1"]
                    / (
                        param_list[f"{_cap(key)}_PARALLELISM_DIM_0"]
                        * param_list[f"{_cap(key)}_PARALLELISM_DIM_1"]
                    )
                )
            elif "bias" in key:
                if "norm" in node_name:
                    image_depth = (
                        param_list[f"{_cap(key)}_TENSOR_SIZE_DIM_0"]
                        * param_list[f"{_cap(key)}_TENSOR_SIZE_DIM_1"]
                        / (
                            param_list[f"{_cap(key)}_PARALLELISM_DIM_0"]
                            * param_list[f"{_cap(key)}_PARALLELISM_DIM_1"]
                        )
                    )
                else:
                    image_depth = (
                        param_list[f"{_cap(key)}_TENSOR_SIZE_DIM_0"]
                        / param_list[f"{_cap(key)}_PARALLELISM_DIM_0"]
                    )
            else:
                raise NotImplementedError
            return image_depth

        image_depth = get_image_depth(
            key, node.meta["mase"].parameters["hardware"]["verilog_param"], node.name
        )
        parameters = ""
        for param in node.meta["mase"].parameters["hardware"]["verilog_param"].keys():
            if f"{_cap(key)}_" in param:
                parameters += f"    .{param}({param}),\n"
        parameters = _remove_last_comma(parameters)
        modules = ""
        signal = ""
        for i in range(reuse_times):
            new_node_name = node_name.replace(stream_name, folded_name + f"_{i}_" + stream_name)
            new_componet_name = component_name.replace(stream_name, folded_name + f"_{i}_" + stream_name)
            new_component_name_inst = component_name_inst.replace(
                stream_name, folded_name + f"_{i}_" + stream_name
            )
            signal += f"""
logic [{_cap(key)}_PRECISION_0 - 1:0] {new_node_name}_m{key} [{_cap(key)}_PARALLELISM_DIM_0*{_cap(key)}_PARALLELISM_DIM_1 - 1:0];
logic [{_cap(key)}_PRECISION_1 - 1:0] {new_node_name}_e{key};
logic {new_node_name}_{key}_valid, {new_node_name}_{key}_ready;
"""
            modules += f"""
{new_componet_name} #(
{parameters}
) {new_component_name_inst} (
    .clk(clk),
    .rst(rst),
    .mdata_out({new_node_name}_m{key}),
    .edata_out({new_node_name}_e{key}),
    .data_out_ready({new_node_name}_{key}_ready),
    .data_out_valid({new_node_name}_{key}_valid)
);

    """

        output_connections = f"""
always_comb begin"""
        
        for element in ["m", "e"]:
            output_connections += f"""
    {element}data_out = (counter<IMAGE_DEPTH)?"""
            for i in range(reuse_times - 1):
                new_node_name = node_name.replace(stream_name, folded_name + f"_{i}_" + stream_name)
                output_connections += f"""
    {node_name.replace(stream_name, folded_name + f"_{i}_" + stream_name)}_{element}{key}: (counter<{i+2}*IMAGE_DEPTH)?"""
            output_connections += f"""
    {node_name.replace(stream_name, folded_name + f"_{reuse_times - 1}_" + stream_name)}_{element}{key}: {node_name.replace(stream_name, folded_name + f"_{0}_" + stream_name)}_{element}{key};
"""
        for item in ["_valid"]:
            output_connections += f"""
    data_out{item} = (counter<IMAGE_DEPTH)?"""
            for i in range(reuse_times - 1):
                new_node_name = node_name.replace(stream_name, folded_name + f"_{i}_" + stream_name)
                output_connections += f"""
    {node_name.replace(stream_name, folded_name + f"_{i}_" + stream_name)}_{key}{item}: (counter<{i+2}*IMAGE_DEPTH)?"""
            output_connections += f"""
    {node_name.replace(stream_name, folded_name + f"_{reuse_times - 1}_" + stream_name)}_{key}{item}: {node_name.replace(stream_name, folded_name + f"_{0}_" + stream_name)}_{key}{item};
"""
        output_connections += f"end \n"
        input_connections = """
always_comb begin
    """
        for i in range(reuse_times):
            input_connections += f"""
{node_name.replace(stream_name, folded_name + f"_{i}_" + stream_name)}_{key}_ready = (({i}*IMAGE_DEPTH<=counter) && (counter<{i+1}*IMAGE_DEPTH))? data_out_ready:0; """
        input_connections += """
end
    """
        connections = input_connections + output_connections

        new_module = f"""
`timescale 1ns / 1ps
module {component_name} #(
    parameter {_cap(key)}_TENSOR_SIZE_DIM_0  = -1,
    parameter {_cap(key)}_TENSOR_SIZE_DIM_1  = -1,
    parameter {_cap(key)}_PRECISION_0 = -1,
    parameter {_cap(key)}_PRECISION_1 = -1,

    parameter {_cap(key)}_PARALLELISM_DIM_0 = -1,
    parameter {_cap(key)}_PARALLELISM_DIM_1 = -1
) (
    input clk,
    input rst,

    output logic [{_cap(key)}_PRECISION_0-1:0] mdata_out      [{_cap(key)}_PARALLELISM_DIM_0 * {_cap(key)}_PARALLELISM_DIM_1-1:0],
    output logic [{_cap(key)}_PRECISION_1-1:0] edata_out,
    output logic                 data_out_valid,
    input                        data_out_ready
);
localparam REPEAT_TIMES = {reuse_times};
localparam IMAGE_DEPTH = {int(image_depth)};
localparam COUNTER_DEPTH = REPEAT_TIMES*IMAGE_DEPTH;
logic [$clog2(COUNTER_DEPTH) - 1 : 0] counter;
{signal}
always_ff @(posedge clk)
    if (rst) counter <= 0;
    else
        if (counter == COUNTER_DEPTH) counter <= 0;
        else if (data_out_valid && data_out_ready) counter <= counter + 1;

{modules}

{connections}

endmodule
        """
        return new_module

    top_bram = ""
    for node in folded_gragh.fx_graph.nodes:
        if node.meta["mase"].parameters["hardware"]["is_implicit"] == False:
            for arg, arg_info in node.meta["mase"].parameters["common"]["args"].items():
                if "data_in" in arg:
                    continue
                if not isinstance(arg_info, dict):
                    continue
                top_bram += _emit_module_parameters_top_internal(
                    arg, node, stream_name, folded_name, reuse_times
                )
    return top_bram


def emit_verilog_folded_top(graph, reuse_times, top_name):
    parameter_map = get_verilog_parameters(graph)

    # get_top_map
    parameters = f"""    parameter REPEAT_TIMES = {reuse_times},\n"""
    parameters += VerilogParameterEmitter(graph).emit(graph, parameter_map)
    top = f"""
`timescale 1ns/1ps
module {top_name} #(
{parameters}
) (
    input clk,
    input rst,
    input logic [DATA_IN_0_PRECISION_0-1:0]  mdata_in_0       [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input logic [DATA_IN_0_PRECISION_1-1:0]  edata_in_0,
    input logic                             data_in_0_valid,
    output logic                             data_in_0_ready,
    output logic [DATA_IN_0_PRECISION_0-1:0]  mdata_out_0       [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    output logic [DATA_IN_0_PRECISION_1-1:0]  edata_out_0,
    output logic                             data_out_0_valid,
    input logic                             data_out_0_ready
);
localparam IMAGE_DEPTH = DATA_IN_0_TENSOR_SIZE_DIM_0 * DATA_IN_0_TENSOR_SIZE_DIM_1 / (DATA_IN_0_PARALLELISM_DIM_1*DATA_IN_0_PARALLELISM_DIM_0);
localparam COUNTER_DEPTH = REPEAT_TIMES*IMAGE_DEPTH;
localparam MAN_WIDTH = DATA_IN_0_PRECISION_0;
localparam EXP_WIDTH = DATA_IN_0_PRECISION_1;
localparam IN_SIZE = DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1;


logic [DATA_IN_0_PRECISION_0-1:0]  top_block_mdata_in_0       [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0];
logic [DATA_IN_0_PRECISION_1-1:0]  top_block_edata_in_0;
logic                             top_block_data_in_0_valid;
logic                             top_block_data_in_0_ready;

logic [DATA_IN_0_PRECISION_0-1:0]  top_block_mdata_out_0       [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0];
logic [DATA_IN_0_PRECISION_1-1:0]  top_block_edata_out_0;
logic                             top_block_data_out_0_valid;
logic                             top_block_data_out_0_ready;

logic [DATA_IN_0_PRECISION_0-1:0]  a_fifo_mdata_out_0       [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0];
logic [DATA_IN_0_PRECISION_1-1:0]  a_fifo_edata_out_0;
logic                             a_fifo_data_out_0_valid;
logic                             a_fifo_data_out_0_ready;

logic [$clog2(COUNTER_DEPTH) - 1 : 0] counter_in, counter_out;
always_ff @(posedge clk)
    if (rst) counter_in <= 0;
    else
        if (counter_in == COUNTER_DEPTH) counter_in <= 0;
        else if (top_block_data_in_0_valid && top_block_data_in_0_ready) counter_in <= counter_in + 1;

always_ff @(posedge clk)
    if (rst) counter_out <= 0;
    else
        if (counter_out == COUNTER_DEPTH) counter_out <= 0;
        else if (a_fifo_data_out_0_valid && a_fifo_data_out_0_ready) counter_out <= counter_out + 1;

top_block top_block_inst (
    .clk(clk),
    .rst(rst),
    .mdata_in_0(top_block_mdata_in_0),
    .edata_in_0(top_block_edata_in_0),
    .data_in_0_valid(top_block_data_in_0_valid),
    .data_in_0_ready(top_block_data_in_0_ready),
    .mdata_out_0(top_block_mdata_out_0),
    .edata_out_0(top_block_edata_out_0),
    .data_out_0_valid(top_block_data_out_0_valid),
    .data_out_0_ready(top_block_data_out_0_ready)
);


  unpacked_mx_fifo #(
    .MAN_WIDTH(DATA_IN_0_PRECISION_0),
    .EXP_WIDTH(DATA_IN_0_PRECISION_1), 
    .IN_SIZE(DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1),
    .DEPTH(IMAGE_DEPTH)  // Minimum depth for breaking timing path
  ) data_fifo (
    .clk(clk),
    .rst(rst),
    .mdata_in(top_block_mdata_out_0),
    .edata_in(top_block_edata_out_0),
    .data_in_valid(top_block_data_out_0_valid),
    .data_in_ready(top_block_data_out_0_ready),
    .mdata_out(a_fifo_mdata_out_0),
    .edata_out(a_fifo_edata_out_0),
    .data_out_valid(a_fifo_data_out_0_valid),
    .data_out_ready(a_fifo_data_out_0_ready)
  );

always_comb begin
    top_block_mdata_in_0 = (counter_in < IMAGE_DEPTH)? mdata_in_0: a_fifo_mdata_out_0;
    top_block_edata_in_0 = (counter_in < IMAGE_DEPTH)? edata_in_0: a_fifo_edata_out_0;
    top_block_data_in_0_valid = (counter_in < IMAGE_DEPTH)? data_in_0_valid: a_fifo_data_out_0_valid;
    data_in_0_ready = (counter_in < IMAGE_DEPTH)? top_block_data_in_0_ready: 1'b0;
end
always_comb begin
    mdata_out_0 = a_fifo_mdata_out_0;
    edata_out_0 = a_fifo_edata_out_0;
    data_out_0_valid = (counter_out < (REPEAT_TIMES - 1)*IMAGE_DEPTH)? 0: a_fifo_data_out_0_valid;
    a_fifo_data_out_0_ready = (counter_out >= (REPEAT_TIMES - 1)*IMAGE_DEPTH)? data_out_0_ready: (counter_in < IMAGE_DEPTH) ? 0 : top_block_data_in_0_ready; 
end
endmodule
    """
    return top


def emit_mxint_folded_top_file(graph, top_name, pass_args):
    stream_graph = pass_args["stream_graph"]
    folded_name = pass_args["folded_name"]
    stream_name = pass_args["stream_name"]
    reuse_times = pass_args["reuse_times"]
    top_block = VerilogEmitter(stream_graph).emit(stream_graph, "top_block")
    top_bram = emit_folded_bram(stream_graph, stream_name, folded_name, reuse_times)
    top = emit_verilog_folded_top(graph, reuse_times, top_name)
    top_file = f"""
    {top}
    {top_block}
    {top_bram}
    """
    return top_file

def mxint_folded_top_generation(graph, pass_args={}):
    """Emit the top-level model design in Verilog

    :param graph: a MaseGraph
    :type graph: MaseGraph
    :param pass_args: this pass requires additional arguments which is explained below, defaults to {}
    :type pass_args: _type_, optional
    :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
    :rtype: tuple(MaseGraph, Dict)


    - pass_args
        - project_dir -> str : the directory of the project for cosimulation
        - top_name -> str : top-level name
    """

    logger.info("Emitting Verilog...")

    # Create project directory, and the verilog is emmited to {project_name}/hardware/rtl
    project_dir = (
        pass_args["project_dir"]
        if "project_dir" in pass_args.keys()
        else Path.home() / ".mase" / "top"
    )
    top_name = pass_args["top_name"] if "top_name" in pass_args.keys() else "top"
    init_project(project_dir)
    rtl_dir = os.path.join(project_dir, "hardware", "rtl")
    top = emit_mxint_folded_top_file(graph, top_name, pass_args)

    top_file = os.path.join(rtl_dir, f"{top_name}.sv")
    with open(top_file, "w") as top_design:
        top_design.write(top)