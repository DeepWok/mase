`timescale 1ns / 1ps
// block floating point add
module bfp_adder #(
    // precision_0 represent mantissa width
    // precision_1 represent exponent width
    // 
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 8,
    parameter WEIGHT_PRECISION_0 = 8,
    parameter WEIGHT_PRECISION_1 = 8,
    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 - 1,
    parameter DATA_OUT_0_PRECISION_1 = (DATA_IN_0_PRECISION_1 > WEIGHT_PRECISION_1)? DATA_IN_0_PRECISION_1 + 2 : WEIGHT_PRECISION_1 + 2,
    parameter BLOCK_SIZE = 6
) (
    input clk,
    input rst,
    // m -> mantissa, e -> exponent
    input logic [DATA_IN_0_PRECISION_0-1:0] mdata_in_0[BLOCK_SIZE - 1:0],
    input logic [DATA_IN_0_PRECISION_1-1:0] edata_in_0,
    input data_in_0_valid,
    output data_in_0_ready,

    input logic [WEIGHT_PRECISION_0-1:0] mweight[BLOCK_SIZE - 1:0],
    input logic [WEIGHT_PRECISION_1-1:0] eweight,
    input weight_valid,
    output weight_ready,

    output logic [DATA_OUT_0_PRECISION_0-1:0] mdata_out_0,
    output logic [DATA_OUT_0_PRECISION_1-1:0] edata_out_0,
    output data_out_0_valid,
    input data_out_0_ready
);
  localparam PRODUCT_PRECISION_0 = DATA_OUT_0_PRECISION_0;
  localparam PRODUCT_PRECISION_1 = DATA_OUT_0_PRECISION_1;
  logic [PRODUCT_PRECISION_0-1:0] mpv      [BLOCK_SIZE-1:0];
  logic [PRODUCT_PRECISION_1-1:0] epv      [BLOCK_SIZE-1:0];
  logic                           pv_valid;
  logic                           pv_ready;
  bfp_vector_mult #(
      .DATA_IN_0_PRECISION_0(DATA_IN_0_PRECISION_0),
      .DATA_IN_0_PRECISION_1(DATA_IN_0_PRECISION_1),
      .WEIGHT_PRECISION_0(WEIGHT_PRECISION_0),
      .WEIGHT_PRECISION_1(WEIGHT_PRECISION_1),
      .BLOCK_SIZE(BLOCK_SIZE)
  ) bfp_vector_mult_inst (
      .clk(clk),
      .rst(rst),
      .mdata_in_0(mdata_in_0),
      .edata_in_0(edata_in_0),
      .data_in_0_valid(data_in_0_valid),
      .data_in_0_ready(data_in_0_ready),
      .mweight(mweight),
      .eweight(eweight),
      .weight_valid(weight_valid),
      .weight_ready(weight_ready),
      .mdata_out_0(mpv),
      .edata_out_0(epv),
      .data_out_0_valid(pv_valid),
      .data_out_0_ready(pv_ready)
  );

  localparam SUM_WIDTH = PRODUCT_PRECISION_0 + $clog2(BLOCK_SIZE);
  // sum the products
  logic [SUM_WIDTH-1:0] sum;
  logic                 sum_valid;
  logic                 sum_ready;
  // sum = sum(pv)
  fixed_adder_tree #(
      .IN_SIZE (BLOCK_SIZE),
      .IN_WIDTH(PRODUCT_PRECISION_0)
  ) fixed_adder_tree_inst (
      .clk(clk),
      .rst(rst),
      .data_in(mpv),
      .data_in_valid(pv_valid),
      .data_in_ready(pv_ready),

      .data_out(sum),
      .data_out_valid(sum_valid),
      .data_out_ready(sum_ready)
  );
  always_comb begin : output
    mdata_out_0 = sum >> $clog2(BLOCK_SIZE);
    edata_out_0 = epv + $clog2(BLOCK_SIZE);
    data_out_0_valid = sum_valid;
    data_out_0_ready = sum_ready;
  end



endmodule
