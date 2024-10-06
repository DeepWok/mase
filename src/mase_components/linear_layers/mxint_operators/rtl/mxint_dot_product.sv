`timescale 1ns / 1ps
module mxint_dot_product #(
    // precision_0 represent mantissa width
    // precision_1 represent exponent width
    // 
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 8,
    parameter WEIGHT_PRECISION_0 = 8,
    parameter WEIGHT_PRECISION_1 = 8,
    parameter BLOCK_SIZE = 6,
    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
        BLOCK_SIZE
    ),
    parameter DATA_OUT_0_PRECISION_1 = (DATA_IN_0_PRECISION_1 > WEIGHT_PRECISION_1)? DATA_IN_0_PRECISION_1 + 1 : WEIGHT_PRECISION_1 + 1
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
  logic [DATA_OUT_0_PRECISION_0-1:0] mdp [BLOCK_SIZE-1:0];
  logic [DATA_OUT_0_PRECISION_1-1:0] edp;
  logic mdp_valid, mdp_ready;
  logic mdata_in_0_valid, mdata_in_0_ready;
  logic edata_in_0_valid, edata_in_0_ready;
  logic [DATA_IN_0_PRECISION_1 - 1:0] buffer_edata_in_0;
  logic buffer_edata_in_0_valid, buffer_edata_in_0_ready;
  logic mweight_valid, mweight_ready;
  logic eweight_valid, eweight_ready;
  logic [WEIGHT_PRECISION_1-1:0] buffer_eweight;
  logic buffer_eweight_valid, buffer_eweight_ready;
  logic mdata_out_0_valid, mdata_out_0_ready;
  split2 #() data_in_split_i (
      .data_in_valid (data_in_0_valid),
      .data_in_ready (data_in_0_ready),
      .data_out_valid({mdata_in_0_valid, edata_in_0_valid}),
      .data_out_ready({mdata_in_0_ready, edata_in_0_ready})
  );
  logic [DATA_IN_0_PRECISION_0 - 1:0] mdata_in_0_reg_out[BLOCK_SIZE - 1:0];
  logic mdata_in_0_reg_out_valid, mdata_in_0_reg_out_ready;
  logic [WEIGHT_PRECISION_0 - 1:0] mweight_reg_out[BLOCK_SIZE - 1:0];
  logic mweight_reg_out_valid, mweight_reg_out_ready;
  unpacked_skid_buffer #(
      .DATA_WIDTH(DATA_IN_0_PRECISION_0),
      .IN_NUM(BLOCK_SIZE)
  ) mdata_in_register_slice (
      .clk           (clk),
      .rst           (rst),
      .data_in       (mdata_in_0),
      .data_in_valid (mdata_in_0_valid),
      .data_in_ready (mdata_in_0_ready),
      .data_out      (mdata_in_0_reg_out),
      .data_out_valid(mdata_in_0_reg_out_valid),
      .data_out_ready(mdata_in_0_reg_out_ready)
  );
  unpacked_skid_buffer #(
      .DATA_WIDTH(WEIGHT_PRECISION_0),
      .IN_NUM(BLOCK_SIZE)
  ) mweight_register_slice (
      .clk           (clk),
      .rst           (rst),
      .data_in       (mweight),
      .data_in_valid (mweight_valid),
      .data_in_ready (mweight_ready),
      .data_out      (mweight_reg_out),
      .data_out_valid(mweight_reg_out_valid),
      .data_out_ready(mweight_reg_out_ready)
  );
  split2 #() weight_split_i (
      .data_in_valid (weight_valid),
      .data_in_ready (weight_ready),
      .data_out_valid({mweight_valid, eweight_valid}),
      .data_out_ready({mweight_ready, eweight_ready})
  );
  fifo #(
      .DEPTH($clog2(BLOCK_SIZE)),
      .DATA_WIDTH(DATA_IN_0_PRECISION_1)
  ) data_in_0_ff_inst (
      .clk(clk),
      .rst(rst),
      .in_data(edata_in_0),
      .in_valid(edata_in_0_valid),
      .in_ready(edata_in_0_ready),
      .out_data(buffer_edata_in_0),
      .out_valid(buffer_edata_in_0_valid),
      .out_ready(buffer_edata_in_0_ready),
      .empty(),
      .full()
  );
  fifo #(
      .DEPTH($clog2(BLOCK_SIZE)),
      .DATA_WIDTH(WEIGHT_PRECISION_1)
  ) weight_ff_inst (
      .clk(clk),
      .rst(rst),
      .in_data(eweight),
      .in_valid(eweight_valid),
      .in_ready(eweight_ready),
      .out_data(buffer_eweight),
      .out_valid(buffer_eweight_valid),
      .out_ready(buffer_eweight_ready),
      .empty(),
      .full()
  );
  assign edata_out_0 = $signed(buffer_eweight) + $signed(buffer_edata_in_0);
  fixed_dot_product #(
      .IN_WIDTH(DATA_IN_0_PRECISION_0),
      .WEIGHT_WIDTH(WEIGHT_PRECISION_0),
      .IN_SIZE(BLOCK_SIZE)
  ) fdp_inst (
      .clk(clk),
      .rst(rst),
      .data_in(mdata_in_0_reg_out),
      .data_in_valid(mdata_in_0_reg_out_valid),
      .data_in_ready(mdata_in_0_reg_out_ready),
      .weight(mweight_reg_out),
      .weight_valid(mweight_reg_out_valid),
      .weight_ready(mweight_reg_out_ready),
      .data_out(mdata_out_0),
      .data_out_valid(mdata_out_0_valid),
      .data_out_ready(mdata_out_0_ready)
  );
  join_n #(
      .NUM_HANDSHAKES(3)
  ) join_inst (
      .data_in_ready ({mdata_out_0_ready, buffer_eweight_ready, buffer_edata_in_0_ready}),
      .data_in_valid ({mdata_out_0_valid, buffer_eweight_valid, buffer_edata_in_0_valid}),
      .data_out_valid(data_out_0_valid),
      .data_out_ready(data_out_0_ready)
  );

endmodule
