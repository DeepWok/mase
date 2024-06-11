/*
Module      : softermax_lpw_pow2
Description : This module implements 2^x with linear piecewise approximation.

              Uses 4 linear pieces between [0, 1) for the fraction then shifts
              it by the integer part.

              TODO: need to support (-inf, 1) -> (0, 2)
*/

`timescale 1ns / 1ps

module softermax_lpw_pow2 #(
    parameter IN_WIDTH = 8,
    parameter IN_FRAC_WIDTH = 4,
    parameter OUT_WIDTH = 8,
    parameter OUT_FRAC_WIDTH = 4
) (
    input logic clk,
    input logic rst,

    input  logic [IN_WIDTH-1:0] in_data,
    input  logic                in_valid,
    output logic                in_ready,

    output logic [OUT_WIDTH-1:0] out_data,
    output logic                 out_valid,
    input  logic                 out_ready
);

  // -----
  // Parameters
  // -----

  // Input: x
  localparam INT_WIDTH = IN_WIDTH - IN_FRAC_WIDTH;

  // Slope: m
  localparam SLOPE_FRAC_WIDTH = OUT_WIDTH;
  localparam SLOPE_WIDTH = 2 + SLOPE_FRAC_WIDTH;

  // Mult: mx
  localparam MULT_FRAC_WIDTH = IN_FRAC_WIDTH + SLOPE_FRAC_WIDTH;
  localparam MULT_WIDTH = IN_WIDTH + SLOPE_WIDTH;

  // Intercept (need to match mx frac): c
  localparam INTERCEPT_FRAC_WIDTH = MULT_FRAC_WIDTH;
  localparam INTERCEPT_WIDTH = 2 + INTERCEPT_FRAC_WIDTH;

  // Output width: mx + c
  localparam LPW_WIDTH = MULT_WIDTH + 1;
  localparam LPW_FRAC_WIDTH = MULT_FRAC_WIDTH;  // == INTERCEPT_FRAC_WIDTH

  // PARAMETERS BELOW ONLY USED IN 1/2-BIT CASE
  // Output result of 2^[0,1] is in [1,2] which requires 2 integer bits
  localparam LUT_WIDTH = IN_FRAC_WIDTH + 2;


  initial begin
    assert (INT_WIDTH > 0);  // Untested for 0 int width
    assert (OUT_WIDTH > OUT_FRAC_WIDTH);  // Untested for 0 out frac width
    assert (IN_WIDTH > 0);
    assert (IN_FRAC_WIDTH >= 0);
  end

  // Wires
  logic [INT_WIDTH-1:0] in_data_int;  // Q INT.0
  logic [IN_FRAC_WIDTH-1:0] in_data_frac;  // Q 0.FRAC

  logic [OUT_WIDTH-1:0] result_data;
  logic result_valid, result_ready;


  // Function to generate LUT (Only used 1/2-bit case)
  function automatic logic [OUT_WIDTH-1:0] pow2_func(real x);
    real res, res_shifted;
    bit [OUT_WIDTH-1:0] return_val;
    res = 2.0 ** x;

    // Output cast
    res_shifted = res * (2.0 ** OUT_FRAC_WIDTH);
    return_val = logic'(res_shifted);
    return return_val;
  endfunction

  // Function to generate slope variable (m)
  function automatic logic [SLOPE_WIDTH-1:0] slope(real x1, real x2);
    real y1, y2, res, res_shifted;
    bit [SLOPE_WIDTH-1:0] return_val;
    y1 = 2.0 ** x1;
    y2 = 2.0 ** x2;
    res = (y2 - y1) / (x2 - x1);

    // Output cast
    res_shifted = res * (2.0 ** SLOPE_FRAC_WIDTH);
    return_val = logic'(res_shifted);
    return return_val;
  endfunction

  // Function to intercept variable (c)
  function automatic logic [INTERCEPT_WIDTH-1:0] intercept(real x1, real x2);
    real m, y1, y2, res, res_shifted;
    bit [INTERCEPT_WIDTH-1:0] return_val;
    y1 = 2.0 ** x1;
    y2 = 2.0 ** x2;
    m = (y2 - y1) / (x2 - x1);
    res = y1 - (m * x1);

    // Output cast
    res_shifted = res * (2 ** INTERCEPT_FRAC_WIDTH);
    return_val = logic'(res_shifted);
    return return_val;
  endfunction

  // -----
  // Logic
  // -----

  assign {in_data_int, in_data_frac} = in_data;

  generate
    if (IN_FRAC_WIDTH <= 1) begin : one_bit_frac

      logic [OUT_WIDTH-1:0] lookup_result;
      always_comb begin
        case (in_data_frac)
          1'b0: lookup_result = pow2_func(0.0);
          1'b1: lookup_result = pow2_func(0.5);
        endcase
        // TODO: Fix pipelining for the shifter
        result_data = lookup_result >> -in_data_int;
        result_valid = in_valid;
        in_ready = result_ready;
      end

    end else if (IN_FRAC_WIDTH == 2) begin : two_bit_frac

      logic [OUT_WIDTH-1:0] lookup_result;
      always_comb begin
        case (in_data_frac)
          2'b00: lookup_result = pow2_func(0.0);
          2'b01: lookup_result = pow2_func(0.25);
          2'b10: lookup_result = pow2_func(0.5);
          2'b11: lookup_result = pow2_func(0.75);
        endcase
        // TODO: Fix pipelining for the shifter
        result_data = lookup_result >> -in_data_int;
        result_valid = in_valid;
        in_ready = result_ready;
      end

    end else begin : lpw_approx

      // Split out the top two bits of the frac again to figure out which
      // piecewise part if lies on
      logic [1:0] frac_top_in, frac_top_out;
      assign frac_top_in = in_data_frac[IN_FRAC_WIDTH-1:IN_FRAC_WIDTH-2];

      logic [INT_WIDTH-1:0] in_data_int_buff[1:0];

      logic [MULT_WIDTH-1:0] mult_in, mult_out;
      logic mult_out_valid, mult_out_ready;
      logic intercept_out_valid, intercept_out_ready;

      logic [LPW_WIDTH-1:0] lpw_int_in, lpw_int_out, lpw_result;
      logic [LPW_WIDTH-1:0] lpw_out_data;
      logic lpw_out_valid, lpw_out_ready;

      logic [OUT_WIDTH:0] lpw_cast_out;

      always_comb begin
        // Multiplication Stage
        case (frac_top_in)
          2'b00: mult_in = in_data_frac * slope(0.00, 0.25);
          2'b01: mult_in = in_data_frac * slope(0.25, 0.50);
          2'b10: mult_in = in_data_frac * slope(0.50, 0.75);
          2'b11: mult_in = in_data_frac * slope(0.75, 1.00);
        endcase
      end

      // Buffer multiplication, top frac bits, and int part
      skid_buffer #(
          .DATA_WIDTH(MULT_WIDTH + 2 + INT_WIDTH)
      ) out_reg (
          .clk(clk),
          .rst(rst),
          .data_in({mult_in, frac_top_in, in_data_int}),
          .data_in_valid(in_valid),
          .data_in_ready(in_ready),
          .data_out({mult_out, frac_top_out, in_data_int_buff[0]}),
          .data_out_valid(mult_out_valid),
          .data_out_ready(mult_out_ready)
      );

      // Add Intercept
      always_comb begin
        case (frac_top_out)
          2'b00: lpw_int_in = mult_out + intercept(0.00, 0.25);
          2'b01: lpw_int_in = mult_out + intercept(0.25, 0.50);
          2'b10: lpw_int_in = mult_out + intercept(0.50, 0.75);
          2'b11: lpw_int_in = mult_out + intercept(0.75, 1.00);
        endcase
      end

      skid_buffer #(
          .DATA_WIDTH(LPW_WIDTH + INT_WIDTH)
      ) intercept_reg (
          .clk(clk),
          .rst(rst),
          .data_in({lpw_int_in, in_data_int_buff[0]}),
          .data_in_valid(mult_out_valid),
          .data_in_ready(mult_out_ready),
          .data_out({lpw_int_out, in_data_int_buff[1]}),
          .data_out_valid(intercept_out_valid),
          .data_out_ready(intercept_out_ready)
      );

      // TODO: Shift up for positive x
      assign lpw_result = lpw_int_out >> -in_data_int_buff[1];

      skid_buffer #(
          .DATA_WIDTH(LPW_WIDTH)
      ) lpw_reg (
          .clk(clk),
          .rst(rst),
          .data_in(lpw_result),
          .data_in_valid(intercept_out_valid),
          .data_in_ready(intercept_out_ready),
          .data_out(lpw_out_data),
          .data_out_valid(lpw_out_valid),
          .data_out_ready(lpw_out_ready)
      );

      fixed_signed_cast #(
          .IN_WIDTH(LPW_WIDTH + 1),
          .IN_FRAC_WIDTH(LPW_FRAC_WIDTH),
          .OUT_WIDTH(OUT_WIDTH + 1),
          .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),
          .SYMMETRIC(0),
          .ROUND_FLOOR(1)
      ) fixed_cast (
          .in_data ({1'b0, lpw_out_data}),
          .out_data(lpw_cast_out)
      );

      assign result_data   = lpw_cast_out[OUT_WIDTH-1:0];
      assign result_valid  = lpw_out_valid;
      assign lpw_out_ready = result_ready;

    end

  endgenerate


  // Output Register
  skid_buffer #(
      .DATA_WIDTH(OUT_WIDTH)
  ) out_reg (
      .clk(clk),
      .rst(rst),
      .data_in(result_data),
      .data_in_valid(result_valid),
      .data_in_ready(result_ready),
      .data_out(out_data),
      .data_out_valid(out_valid),
      .data_out_ready(out_ready)
  );

endmodule
