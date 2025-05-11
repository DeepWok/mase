`timescale 1ns / 1ps
module fixed_difflogic_flatten #(
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0  = 2,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1  = 2,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 4
) (
    input wire clk,
    input wire rst,

    input wire [DATA_IN_0_TENSOR_SIZE_DIM_0-1:0] data_in_0[0:DATA_IN_0_TENSOR_SIZE_DIM_1-1],
    input wire data_in_0_valid,
    output wire data_in_0_ready,

    output reg [(DATA_IN_0_TENSOR_SIZE_DIM_0*DATA_IN_0_TENSOR_SIZE_DIM_1-1):0] data_out_0,
    output reg data_out_0_valid,
    input reg data_out_0_ready
);

  genvar i;
  genvar j;
  generate
    for (i = 0; i < DATA_IN_0_TENSOR_SIZE_DIM_1; i = i + 1) begin : ROW
      for (j = 0; j < DATA_IN_0_TENSOR_SIZE_DIM_0; j = j + 1) begin : COL
        assign data_out_0[i*DATA_IN_0_TENSOR_SIZE_DIM_1+j] = data_in_0[i][j];
      end
    end
  endgenerate

  assign data_out_0_valid = data_in_0_valid;
  assign data_in_0_ready  = data_out_0_ready;

endmodule
