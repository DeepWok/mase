`timescale 1ns / 1ps
/*
Module      : mxint_accumulator
Description : The accumulator for mxint.
              When inputing different exponent, the mantissa will cast to the same bitwidth then accumulate.
*/
module mxint_accumulator #(
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 4,
    parameter BLOCK_SIZE = 4,
    parameter IN_DEPTH = 2,
    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0 + 2 ** DATA_IN_0_PRECISION_1 + $clog2(
        IN_DEPTH
    ),
    parameter DATA_OUT_0_PRECISION_1 = DATA_IN_0_PRECISION_1
) (
    input logic clk,
    input logic rst,

    input  logic [DATA_IN_0_PRECISION_0-1:0] mdata_in_0     [BLOCK_SIZE - 1:0],
    input  logic [DATA_IN_0_PRECISION_1-1:0] edata_in_0,
    input  logic                             data_in_0_valid,
    output logic                             data_in_0_ready,

    output logic [DATA_OUT_0_PRECISION_0-1:0] mdata_out_0     [BLOCK_SIZE - 1:0],
    output logic [DATA_OUT_0_PRECISION_1-1:0] edata_out_0,
    output logic                              data_out_0_valid,
    input  logic                              data_out_0_ready
);

  // 1-bit wider so IN_DEPTH also fits.
  localparam COUNTER_WIDTH = $clog2(IN_DEPTH);
  logic [COUNTER_WIDTH:0] counter;

  /* verilator lint_off WIDTH */
  assign data_in_0_ready  = (counter != IN_DEPTH) || data_out_0_ready;
  assign data_out_0_valid = (counter == IN_DEPTH);
  /* verilator lint_on WIDTH */

  // mantissa shift
  logic [DATA_OUT_0_PRECISION_0 - 1:0] shifted_mdata_in_0[BLOCK_SIZE - 1:0];
  logic [DATA_OUT_0_PRECISION_0 - 1:0] shifted_mdata_out_0[BLOCK_SIZE - 1:0];

  logic no_value_in_register;
  logic [DATA_IN_0_PRECISION_1 - 1:0] exp_min;

  assign no_value_in_register =(counter == 0 || (data_out_0_valid && data_out_0_ready && data_in_0_valid));
  assign exp_min = ($signed(edata_out_0) > $signed(edata_in_0)) ? edata_in_0 : edata_out_0;
  // counter
  always_ff @(posedge clk)
    if (rst) counter <= 0;
    else begin
      if (data_out_0_valid) begin
        if (data_out_0_ready) begin
          if (data_in_0_valid) counter <= 1;
          else counter <= 0;
        end
      end else if (data_in_0_valid && data_in_0_ready) counter <= counter + 1;
    end
  // mantissa

  for (genvar i = 0; i < BLOCK_SIZE; i++) begin : mantissa_block
    // mantissa shift
    for (genvar j = 0; j < 2 ** DATA_IN_0_PRECISION_1; j++) begin : static_shift
      always_comb begin
        if (($signed(edata_in_0) - $signed(exp_min)) == j)
          shifted_mdata_in_0[i] = no_value_in_register ? $signed(
              mdata_in_0[i]
          ) : $signed(
              mdata_in_0[i]
          ) <<< j;
        if (($signed(edata_out_0) - $signed(exp_min)) == j)
          shifted_mdata_out_0[i] = $signed(mdata_out_0[i]) <<< j;
      end
    end
    // mantissa out
    always_ff @(posedge clk)
      if (rst) mdata_out_0[i] <= '0;
      else begin
        if (data_out_0_valid) begin
          if (data_out_0_ready) begin
            if (data_in_0_valid) mdata_out_0[i] <= shifted_mdata_in_0[i];
            else mdata_out_0[i] <= '0;
          end
        end else if (data_in_0_valid && data_in_0_ready)
          mdata_out_0[i] <= $signed(shifted_mdata_out_0[i]) + $signed(shifted_mdata_in_0[i]);
      end
  end
  localparam signed [DATA_IN_0_PRECISION_1 - 1:0] MAXIMUM_EXPONENTIAL = 2**(DATA_IN_0_PRECISION_1 - 1) - 1;
  // exponent
  always_ff @(posedge clk)
    if (rst) edata_out_0 <= MAXIMUM_EXPONENTIAL;
    else if (data_out_0_valid) begin
      if (data_out_0_ready) begin
        if (data_in_0_valid) edata_out_0 <= edata_in_0;
        else edata_out_0 <= MAXIMUM_EXPONENTIAL;
      end
    end else if (data_in_0_valid && data_in_0_ready) edata_out_0 <= exp_min;

endmodule
