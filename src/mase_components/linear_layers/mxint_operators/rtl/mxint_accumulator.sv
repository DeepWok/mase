`timescale 1ns / 1ps
/*
Module      : mxint_accumulator
Description :
  - The accumulator is designed to accumulate a block of MxInt values.
*/

module mxint_accumulator #(
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 4,
    parameter BLOCK_SIZE = 4,
    parameter IN_DEPTH = 2,
    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0 + 2 ** DATA_IN_0_PRECISION_1 + $clog2(
        IN_DEPTH
    ),
    parameter DATA_OUT_0_PRECISION_1 = DATA_IN_0_PRECISION_1 + $clog2($clog2(IN_DEPTH) + 1),
    localparam COUNTER_WIDTH = $clog2(IN_DEPTH)
) (
    input logic clk,
    input logic rst,

    // Input Data
    input  logic [DATA_IN_0_PRECISION_0-1:0] mdata_in_0     [BLOCK_SIZE - 1:0],
    input  logic [DATA_IN_0_PRECISION_1-1:0] edata_in_0,
    input  logic                             data_in_0_valid,
    output logic                             data_in_0_ready,

    // Output Data
    output logic [DATA_OUT_0_PRECISION_0-1:0] mdata_out_0     [BLOCK_SIZE - 1:0],
    output logic [DATA_OUT_0_PRECISION_1-1:0] edata_out_0,
    output logic                              data_out_0_valid,
    input  logic                              data_out_0_ready,
    output logic [           COUNTER_WIDTH:0] accum_count
);

  localparam RIGHT_PADDING = 2 ** DATA_IN_0_PRECISION_1;
  localparam LEFT_PADDING = $clog2(IN_DEPTH);

  localparam EXP_IN_BIAS = 2 ** (DATA_IN_0_PRECISION_1 - 1) - 1;
  localparam EXP_OUT_BIAS = 2 ** (DATA_OUT_0_PRECISION_1 - 1) - 1;

  /* verilator lint_off WIDTH */
  assign data_in_0_ready  = (accum_count != IN_DEPTH) || data_out_0_ready;
  assign data_out_0_valid = (accum_count == IN_DEPTH);
  /* verilator lint_on WIDTH */

  logic signed [DATA_OUT_0_PRECISION_0 - 1:0] padded_mdata_in_0  [BLOCK_SIZE - 1:0];
  logic signed [DATA_OUT_0_PRECISION_0 - 1:0] shifted_mdata_in_0 [BLOCK_SIZE - 1:0];
  logic signed [DATA_OUT_0_PRECISION_0 - 1:0] shifted_mdata_out_0[BLOCK_SIZE - 1:0];
  logic signed [DATA_OUT_0_PRECISION_0 - 1:0] tmp_accumulator    [BLOCK_SIZE - 1:0];

  logic                                       exponent_increment [BLOCK_SIZE - 1:0];

  logic        [ DATA_IN_0_PRECISION_1 - 1:0] max_exponent_d;
  logic        [ DATA_IN_0_PRECISION_1 - 1:0] max_exponent_q;

  logic signed [DATA_OUT_0_PRECISION_1 - 1:0] shift;
  logic                                       no_reg_value;


  // =============================
  // Exponent Calculation
  // =============================
  assign no_reg_value =(accum_count == 0 || (data_out_0_valid && data_out_0_ready && data_in_0_valid));
  assign max_exponent_d = (max_exponent_q < edata_in_0) ? edata_in_0 : max_exponent_q;
  assign shift = max_exponent_q - edata_in_0;

  always_ff @(posedge clk)
    if (rst) accum_count <= 0;
    else begin

      if (data_out_0_valid) begin

        if (data_out_0_ready) begin
          if (data_in_0_valid) accum_count <= 1;
          else accum_count <= 0;
        end

      end else if (data_in_0_valid && data_in_0_ready) accum_count <= accum_count + 1;
    end

  // =============================
  // Mantissa Shift and Accumulation
  // =============================
  for (genvar i = 0; i < BLOCK_SIZE; i++) begin

    always_comb begin

      padded_mdata_in_0[i] = {
        {LEFT_PADDING{mdata_in_0[i][DATA_IN_0_PRECISION_0-1]}}, mdata_in_0[i], {RIGHT_PADDING{1'b0}}
      };

      if (no_reg_value) begin
        shifted_mdata_in_0[i]  = padded_mdata_in_0[i];
        shifted_mdata_out_0[i] = '0;
      end else if (shift > 0) begin
        shifted_mdata_in_0[i]  = padded_mdata_in_0[i] >>> shift;
        shifted_mdata_out_0[i] = mdata_out_0[i];
      end else begin
        shifted_mdata_in_0[i]  = padded_mdata_in_0[i];
        shifted_mdata_out_0[i] = $signed(mdata_out_0[i]) >>> -shift;
      end

    end

  end

  // =============================
  // Mantissa Output Update Logic
  // =============================

  genvar i;
  for (i = 0; i < BLOCK_SIZE; i++) begin
    always_ff @(posedge clk) begin
      if (rst) mdata_out_0[i] <= '0;
      else begin
        if (data_out_0_valid) begin
          if (data_out_0_ready) begin

            if (data_in_0_valid) mdata_out_0[i] <= shifted_mdata_in_0[i];
            else mdata_out_0[i] <= '0;

          end
        end else if (data_in_0_valid && data_in_0_ready)
          mdata_out_0[i] <= shifted_mdata_out_0[i] + shifted_mdata_in_0[i];
      end
    end
  end

  // =============================
  // Exponent Output Update Logic
  // =============================

  always_ff @(posedge clk) begin

    if (rst) begin
      edata_out_0 <= '0;
      max_exponent_q <= '0;

    end else if (data_out_0_valid) begin
      if (data_out_0_ready) begin

        if (data_in_0_valid) begin
          edata_out_0 <= edata_in_0 - EXP_IN_BIAS + EXP_OUT_BIAS + LEFT_PADDING;
          max_exponent_q <= edata_in_0;
        end else begin
          edata_out_0 <= '0;
          max_exponent_q <= '0;
        end

      end
    end else if (data_in_0_valid && data_in_0_ready) begin
      max_exponent_q <= max_exponent_d;
      edata_out_0 <= max_exponent_d - EXP_IN_BIAS + EXP_OUT_BIAS + LEFT_PADDING;
    end

  end

endmodule
