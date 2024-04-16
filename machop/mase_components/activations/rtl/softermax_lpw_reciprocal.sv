/*
Module      : softermax_lpw_reciprocal
Description : This module implements 1/x using linear piecewise approximation.

              The softermax module allows us to assume:
              - Input is unsigned. (x >= 0)
              - Therefore, the output is also unsigned. (y >= 0)

              This module calculates 1/x using Newton-Raphson iteration in the
              domain: [1, 2). It will shift all numbers into that range and then
              shift the number back once the 1/x calculation is done.
*/

`timescale 1ns/1ps

module softermax_lpw_reciprocal #(
    parameter IN_WIDTH = 8,
    parameter IN_FRAC_WIDTH = 4,
    parameter OUT_WIDTH = 8,
    parameter OUT_FRAC_WIDTH = 5
) (
    input  logic clk,
    input  logic rst,

    // Input streaming interface
    input  logic [IN_WIDTH-1:0]  in_data,
    input  logic                 in_valid,
    output logic                 in_ready,

    output logic [OUT_WIDTH-1:0] out_data,
    output logic                 out_valid,
    input  logic                 out_ready
);

// -----
// Parameters
// -----

// Range reduced num
localparam RANGE_REDUCED_WIDTH = IN_WIDTH;
localparam RANGE_REDUCED_FRAC_WIDTH = IN_WIDTH;

localparam MSB_WIDTH = $clog2(IN_WIDTH);


// Input: x
// localparam INT_WIDTH = IN_WIDTH - IN_FRAC_WIDTH;

// Slope: m
localparam SLOPE_WIDTH = 1 + RANGE_REDUCED_FRAC_WIDTH;
localparam SLOPE_FRAC_WIDTH = RANGE_REDUCED_FRAC_WIDTH;

// Mult: mx
localparam MULT_WIDTH = IN_WIDTH + SLOPE_WIDTH;
localparam MULT_FRAC_WIDTH = IN_FRAC_WIDTH + SLOPE_FRAC_WIDTH;

// Intercept (need to match mx frac): c
localparam INTERCEPT_FRAC_WIDTH = MULT_FRAC_WIDTH;
localparam INTERCEPT_WIDTH = 2 + INTERCEPT_FRAC_WIDTH;

// Output width: mx + c
localparam LPW_WIDTH = MULT_WIDTH + 1;
localparam LPW_FRAC_WIDTH = MULT_FRAC_WIDTH; // == INTERCEPT_FRAC_WIDTH


// -----
// Wires
// -----

logic [RANGE_REDUCED_WIDTH-1:0] range_reduced_num [1:0];
logic [MSB_WIDTH-1:0] msb [2:0];
logic msb_not_found;
logic range_reduce_out_valid, range_reduce_out_ready;

logic [1:0] frac_top_in, frac_top_out;

logic [MULT_WIDTH-1:0] mult_in, mult_out;
logic mult_in_valid, mult_in_ready;
logic mult_out_valid, mult_out_ready;


// -----
// Functions
// -----

// Function to generate slope variable (m)
function logic [SLOPE_WIDTH-1:0] slope (real x1, real x2);
    real y1, y2, res, res_shifted;
    int res_int;
    y1 = 1.0 / x1;
    y2 = 1.0 / x2;
    res = (y2 - y1) / (x2 - x1);

    // Output cast
    res_shifted = res * (2 ** SLOPE_FRAC_WIDTH);
    res_int = int'(res_shifted);
    return SLOPE_WIDTH'(res_int);
endfunction

// Function to intercept variable (c)
function logic [INTERCEPT_WIDTH-1:0] intercept (real x1, real x2);
    real m, y1, y2, res, res_shifted;
    int res_int;
    y1 = 1.0 / x1;
    y2 = 1.0 / x2;
    m = (y2 - y1) / (x2 - x1);
    res = y1 - (m * x1);

    // Output cast
    res_shifted = res * (2 ** INTERCEPT_FRAC_WIDTH);
    res_int = int'(res_shifted);
    return INTERCEPT_WIDTH'(res_int);
endfunction


// -----
// Modules
// -----

fixed_range_reduction #(
    .WIDTH(IN_WIDTH)
) range_reduce (
    .data_a(in_data),
    .data_out(range_reduced_num[0]), // This num is in the format Q1.(IN_WIDTH-1)
    .msb_index(msb),
    .not_found(msb_not_found) // if msb_not_found, then x = 0
);

skid_buffer #(
    .DATA_WIDTH(RANGE_REDUCED_WIDTH + MSB_WIDTH)
) range_reduce_reg (
    .clk(clk),
    .rst(rst),
    .data_in({range_reduced_num[0], msb[0]}),
    .data_in_valid(in_valid),
    .data_in_ready(in_ready),
    .data_out({range_reduced_num[1], msb[1]}),
    .data_out_valid(range_reduce_out_valid),
    .data_out_ready(range_reduce_out_ready)
);

assign frac_top_in = range_reduced_num[1][RANGE_REDUCED_WIDTH-2:RANGE_REDUCED_WIDTH-3];

// Multiplication Stage
always_comb begin
    case (frac_top_in)
        2'b00: mult_in = range_reduced_num[1] * slope(1.00, 1.25);
        2'b01: mult_in = range_reduced_num[1] * slope(1.25, 1.50);
        2'b10: mult_in = range_reduced_num[1] * slope(1.50, 1.75);
        2'b11: mult_in = range_reduced_num[1] * slope(1.75, 2.00);
    endcase
end

skid_buffer #(
    .DATA_WIDTH(MULT_WIDTH + 2 + MSB_WIDTH)
) mult_stage_reg (
    .clk(clk),
    .rst(rst),
    .data_in({mult_in, frac_top_in, msb[1]}),
    .data_in_valid(),
    .data_in_ready(),
    .data_out({mult_out, frac_top_out, msb[2]}),
    .data_out_valid(),
    .data_out_ready()
);

// Add Intercept & Shift stage
always_comb begin
    case (frac_top_out)
        2'b00: lpw_int = mult_out + intercept(1.00, 1.25);
        2'b01: lpw_int = mult_out + intercept(1.25, 1.50);
        2'b10: lpw_int = mult_out + intercept(1.50, 1.75);
        2'b11: lpw_int = mult_out + intercept(1.75, 2.00);
    endcase
    lpw_result = lpw_int >> -in_data_int_buff; // TODO: Shift up for positive x
end



initial begin
    // $display("reciprocal(0.25) = %d (%f)", reciprocal(0.25), real'(reciprocal(0.25)) / (2 ** OUT_FRAC_WIDTH));
    // $display("reciprocal(0.5) = %d (%f)", reciprocal(0.5), real'(reciprocal(0.5)) / (2 ** OUT_FRAC_WIDTH));
    $display("slope(1.00, 1.25) = %b = -%d", slope(1.00, 1.25), ~slope(1.00, 1.25)+1'b1);
    $display("slope(1.25, 1.50) = %b = -%d", slope(1.25, 1.50), ~slope(1.25, 1.50)+1'b1);
    $display("slope(1.50, 1.75) = %b = -%d", slope(1.50, 1.75), ~slope(1.50, 1.75)+1'b1);
    $display("slope(1.75, 2.00) = %b = -%d", slope(1.75, 2.00), ~slope(1.75, 2.00)+1'b1);
    $finish;
end



endmodule
