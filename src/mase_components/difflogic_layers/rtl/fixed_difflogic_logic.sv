`timescale 1ns / 1ps
module fixed_difflogic_logic #(
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 4,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 4,
    parameter [3:0] LAYER_OP_CODES[0:(DATA_OUT_0_TENSOR_SIZE_DIM_0-1)] = '{
        4'd0,
        4'd1,
        4'd6,
        4'd7
    },
    parameter [$clog2(
DATA_IN_0_TENSOR_SIZE_DIM_0
)-1:0] IND_A[0:(DATA_OUT_0_TENSOR_SIZE_DIM_0-1)] = '{2'b00, 2'b01, 2'b10, 2'b11},
    parameter [$clog2(
DATA_IN_0_TENSOR_SIZE_DIM_0
)-1:0] IND_B[0:(DATA_OUT_0_TENSOR_SIZE_DIM_0-1)] = '{2'b11, 2'b10, 2'b01, 2'b00}
) (
    input wire clk,
    input wire rst,

    input wire [(DATA_IN_0_TENSOR_SIZE_DIM_0-1):0] data_in_0,
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    output reg [(DATA_OUT_0_TENSOR_SIZE_DIM_0-1):0] data_out_0,
    output logic data_out_0_valid,
    input logic data_out_0_ready
);

  genvar i;
  generate

    for (i = 0; i < DATA_OUT_0_TENSOR_SIZE_DIM_0; i = i + 1) begin : gen_block
      fixed_difflogic_logic_neuron #(
          .OP_CODE(LAYER_OP_CODES[i])
      ) neuron_inst (
          .clk(clk),
          .rst(rst),
          .a  (data_in_0[IND_A[i]]),
          .b  (data_in_0[IND_B[i]]),
          .res(data_out_0[i])
      );
    end

  endgenerate

  assign data_out_0_valid = data_in_0_valid;
  assign data_in_0_ready  = data_out_0_ready;

endmodule
