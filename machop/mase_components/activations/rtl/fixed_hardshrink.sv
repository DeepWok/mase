`timescale 1ns / 1ps
module fixed_hardshrink #(
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
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1,
    parameter LAMBDA = 0.5,  //the threshold
    parameter FX_LAMBDA = $rtoi(LAMBDA * 2 ** (DATA_IN_0_PRECISION_1))  //the threshold
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input rst,
    input clk,
    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],

    input  logic data_in_0_valid,
    output logic data_in_0_ready,
    output logic data_out_0_valid,
    input  logic data_out_0_ready
);
  logic [DATA_IN_0_PRECISION_0-1:0] fx_lambda;
  logic [DATA_IN_0_PRECISION_0-1:0] cast_data [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0];

  for (
      genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1; i++
  ) begin : HardShrink
    always_comb begin
      if ($signed(data_in_0[i]) < -1 * FX_LAMBDA) cast_data[i] = data_in_0[i];
      else if ($signed(data_in_0[i]) > FX_LAMBDA) cast_data[i] = data_in_0[i];
      else cast_data[i] = '0;
      $display("%d", cast_data[i]);
    end
  end

  fixed_rounding #(
      .IN_SIZE(DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1),
      .IN_WIDTH(DATA_IN_0_PRECISION_0),
      .IN_FRAC_WIDTH(DATA_IN_0_PRECISION_1),
      .OUT_WIDTH(DATA_OUT_0_PRECISION_0),
      .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1)
  ) data_out_cast (
      .data_in (cast_data),
      .data_out(data_out_0)
  );


  assign data_out_0_valid = data_in_0_valid;
  assign data_in_0_ready  = data_out_0_ready;

endmodule
