`timescale 1ns / 1ps
module fixed_hardswish #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 4,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 8,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,

    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 4,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 0,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 0,

    parameter INPLACE = 0
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

  logic [DATA_IN_0_PRECISION_0-1:0] tmp_0[DATA_IN_0_TENSOR_SIZE_DIM_0-1:0];
  logic [DATA_IN_0_PRECISION_0-1:0] tmp_1[DATA_IN_0_TENSOR_SIZE_DIM_0-1:0];
  logic [DATA_IN_0_PRECISION_0-1:0] tmp_2[DATA_IN_0_TENSOR_SIZE_DIM_0-1:0];
  logic [DATA_IN_0_PRECISION_0-1:0] tmp_3[DATA_IN_0_TENSOR_SIZE_DIM_0-1:0];

  for (
      genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1; i++
  ) begin : HardSwish

    always_comb begin
      // Default values
      tmp_0[i] = '0;
      tmp_1[i] = '0;
      tmp_2[i] = '0;

      // negative value, put to zero
      if ($signed(data_in_0[i]) < -3) data_out_0[i] = '0;
      else if ($signed(data_in_0[i]) >= 3) data_out_0[i] = data_in_0[i];
      else begin
        //swish
        tmp_0[i] = 3 <<< DATA_IN_0_PRECISION_1;  // 3 in the same fx
        tmp_1[i] = data_in_0[i] + tmp_0[i];  // x + 3
        tmp_2[i] = (tmp_1[i] >>> 3) + (tmp_1[i] >>> 4);  // tmp/8 + tmp/16 ~ tmp/6
        data_out_0[i] = tmp_3[i];  // dout = x(x+3) * 3/16 [Original HardSwish is x(x+3)/6]
      end
    end

    fixed_mult #(
        .IN_A_WIDTH(DATA_IN_0_PRECISION_0),
        .IN_B_WIDTH(DATA_IN_0_PRECISION_0)
    ) fixed_mult_inst (
        .data_a (data_in_0[i]),
        .data_b (tmp_2[i]),
        .product(tmp_3[i])
    );
  end

  assign data_out_0_valid = data_in_0_valid;
  assign data_in_0_ready  = data_out_0_ready;

endmodule
