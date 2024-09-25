`timescale 1ns / 1ps
// block floating point mult

module mxint_register_slice #(
    // precision represent mantissa width
    // precision_1 represent exponent width
    // 
    parameter DATA_PRECISION_0 = 8,
    parameter DATA_PRECISION_1 = 8,
    parameter BLOCK_SIZE = 6

) (
    input clk,
    input rst,
    // m -> mantissa, e -> exponent
    input logic [DATA_PRECISION_0-1:0] mdata_in[BLOCK_SIZE - 1:0],
    input logic [DATA_PRECISION_1-1:0] edata_in,
    input data_in_valid,
    output data_in_ready,

    output logic [DATA_PRECISION_0-1:0] mdata_out[BLOCK_SIZE - 1:0],
    output logic [DATA_PRECISION_1-1:0] edata_out,
    output data_out_valid,
    input data_out_ready
);

  unpacked_register_slice #(
      .DATA_WIDTH(DATA_PRECISION_0),
      .IN_SIZE(BLOCK_SIZE)
  ) mregister_slice (
      .clk           (clk),
      .rst           (rst),
      .data_in       (mdata_in),
      .data_in_valid (data_in_valid),
      .data_in_ready (data_in_ready),
      .data_out      (mdata_out),
      .data_out_valid(data_out_valid),
      .data_out_ready(data_out_ready)
  );

  register_slice #(
      .DATA_WIDTH(DATA_PRECISION_1)
  ) eregister_slice (
      .clk           (clk),
      .rst           (rst),
      .data_in       (edata_in),
      .data_in_valid (data_in_valid),
      .data_in_ready (),
      .data_out      (edata_out),
      .data_out_valid(),
      .data_out_ready(data_out_ready)
  );
endmodule
