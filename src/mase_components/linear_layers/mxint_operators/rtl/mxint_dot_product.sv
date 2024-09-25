`timescale 1ns / 1ps
// block floating point add
module mxint_dot_product #(
    // precision_0 represent mantissa width
    // precision_1 represent exponent width
    // 
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 8,
    parameter WEIGHT_PRECISION_0 = 8,
    parameter WEIGHT_PRECISION_1 = 8,
    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 - 1,
    parameter DATA_OUT_0_PRECISION_1 = (DATA_IN_0_PRECISION_1 > WEIGHT_PRECISION_1)? DATA_IN_0_PRECISION_1 + 1 : WEIGHT_PRECISION_1 + 1,
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
  localparam PRODUCT_PRECISION_1 = DATA_OUT_0_PRECISION_1 - 1;
  logic [PRODUCT_PRECISION_0-1:0] mpv[BLOCK_SIZE-1:0];
  logic [PRODUCT_PRECISION_1-1:0] epv, epv_out;


endmodule
