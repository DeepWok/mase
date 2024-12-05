`timescale 1ns / 1ps
/*
  Currently, we dont' want to support parallelism
  Cause in attention, it's actually not in parallel
*/
module mxint_gelu_element #(
    parameter IN_MAN_WIDTH = 4,
    parameter IN_EXP_WIDTH = 8,
    parameter OUT_MAN_WIDTH = 4,
    parameter OUT_EXP_WIDTH = 8
) (
    input logic[IN_MAN_WIDTH-1:0] mdata_in_0,
    input logic[IN_EXP_WIDTH-1:0] edata_in_0,
    output logic[OUT_MAN_WIDTH-1:0] mdata_out_0,
    output logic[OUT_EXP_WIDTH-1:0] edata_out_0
);
    localparam VALID_WIDTH = IN_MAN_WIDTH + 2;
    localparam logic[VALID_WIDTH-1:0] MIN_VAL = -(2 ** (OUT_MAN_WIDTH - 1));
    localparam logic[VALID_WIDTH-1:0] MAX_VAL = (2 ** (OUT_MAN_WIDTH - 1)) - 1;

    logic[VALID_WIDTH - 1:0] real_x ;
    logic[VALID_WIDTH - 1:0] real_x_v [0:0];

    logic[OUT_MAN_WIDTH-1:0] lut_out;
    logic [OUT_MAN_WIDTH-1:0] shifted_lut_out_v[0:0];
    logic [OUT_MAN_WIDTH-1:0] shifted_lut_out;

    optimized_right_shift #(
        .IN_WIDTH(IN_MAN_WIDTH),
        .SHIFT_WIDTH(IN_EXP_WIDTH),
        .OUT_WIDTH(VALID_WIDTH),
        .BLOCK_SIZE(1)
    ) data_in_shift_inst (
        .data_in({mdata_in_0}),
        .shift_value(edata_in_0),
        .data_out(real_x_v)
    );

    assign real_x = real_x_v[0];

    gelu_lut #(
        .DATA_IN_0_PRECISION_0(VALID_WIDTH),
        .DATA_IN_0_PRECISION_1(VALID_WIDTH - 3),
        .DATA_OUT_0_PRECISION_0(OUT_MAN_WIDTH),
        .DATA_OUT_0_PRECISION_1(OUT_MAN_WIDTH - 1)
    ) gelu_lut_inst (
        .data_in_0(real_x),
        .data_out_0(lut_out)
    );

    optimized_right_shift #(
        .IN_WIDTH(OUT_MAN_WIDTH),
        .SHIFT_WIDTH(IN_EXP_WIDTH),
        .OUT_WIDTH(OUT_MAN_WIDTH),
        .BLOCK_SIZE(1)
    ) lut_out_shift_inst (
        .data_in({lut_out}),
        .shift_value(edata_in_0),
        .data_out(shifted_lut_out_v)
    );
    assign shifted_lut_out = shifted_lut_out_v[0];

    always_comb begin
        if (real_x == MAX_VAL) begin
            mdata_out_0 = mdata_in_0;
            edata_out_0 = edata_in_0;
        end else if (real_x == MIN_VAL) begin
            mdata_out_0 = 0;
            edata_out_0 = edata_in_0;
        end else begin
            mdata_out_0 = shifted_lut_out;
            edata_out_0 = edata_in_0;
        end
    end
endmodule

module mxint_gelu #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 4,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 10,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_2 = 1,

    parameter IN_0_DEPTH = $rtoi($ceil(DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0)),

    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 4,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_2 = DATA_IN_0_TENSOR_SIZE_DIM_2,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_2 = DATA_IN_0_PARALLELISM_DIM_2
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input clk,
    input rst,

    input logic data_in_0_valid,
    output logic data_in_0_ready,
    input logic [DATA_IN_0_PRECISION_0-1:0] mdata_in_0[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input logic [DATA_IN_0_PRECISION_1-1:0] edata_in_0,

    output logic data_out_0_valid,
    input logic data_out_0_ready,
    output logic [DATA_OUT_0_PRECISION_0-1:0] mdata_out_0[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic [DATA_OUT_0_PRECISION_1-1:0] edata_out_0
);

  logic [DATA_OUT_0_PRECISION_0-1:0] gelu_mdata_out [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [DATA_OUT_0_PRECISION_1-1:0] gelu_edata_out;

  for (genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1; i++) begin : gelu
    mxint_gelu_element #(
        .IN_MAN_WIDTH(DATA_IN_0_PRECISION_0),
        .IN_EXP_WIDTH(DATA_IN_0_PRECISION_1),
        .OUT_MAN_WIDTH(DATA_OUT_0_PRECISION_0),
        .OUT_EXP_WIDTH(DATA_OUT_0_PRECISION_1)
    ) gelu_inst (
        .mdata_in_0(mdata_in_0[i]),
        .edata_in_0(edata_in_0),
        .mdata_out_0(gelu_mdata_out[i]),
        .edata_out_0()
    );
  end
  assign gelu_edata_out = edata_in_0;

  mxint_cast #(
      .IN_MAN_WIDTH(DATA_OUT_0_PRECISION_0),
      .IN_EXP_WIDTH(DATA_OUT_0_PRECISION_1),
      .OUT_MAN_WIDTH(DATA_OUT_0_PRECISION_0),
      .OUT_EXP_WIDTH(DATA_OUT_0_PRECISION_1),
      .BLOCK_SIZE(DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1)
  ) cast_inst (
      .clk(clk),
      .rst(rst),
      .mdata_in(gelu_mdata_out),
      .edata_in(gelu_edata_out),
      .data_in_valid(data_in_0_valid),
      .data_in_ready(data_in_0_ready),
      .mdata_out(mdata_out_0),
      .edata_out(edata_out_0),
      .data_out_valid(data_out_0_valid),
      .data_out_ready(data_out_0_ready)
  );

endmodule