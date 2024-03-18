`timescale 1ns / 1ps
module fixed_silu #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 4,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 10,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,

    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 4,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 10,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1

) (
    /* verilator lint_off UNUSEDSIGNAL */
    input rst,
    input clk,
    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0,
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0,

    input  logic data_in_0_valid,
    output logic data_in_0_ready,
    output logic data_out_0_valid,
    input  logic data_out_0_ready
);
  
  logic [DATA_IN_0_PRECISION_0-1:0] silu_lut [256];
  assign realaddr = $signed(data_in_0) + 128;
  // initial begin
  //   $readmemb("/workspace/machop/mase_components/activations/rtl/silu_map.mem", silu_lut);
  //   $display("input %b %d", data_in_0, data_in_0);
  //   $display("realladdr %b %d", realaddr, realaddr);
  //   $display("SILU at 3 %b", silu_lut[realaddr]);
  //   $writememb("/workspace/machop/mase_components/activations/rtl/memory_binary.txt", silu_lut);
  // end
  assign data_out_0 = silu_lut[realaddr];
  assign data_out_0_valid = data_in_0_valid;
  assign data_in_0_ready  = data_out_0_ready;

endmodule
