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

// Range reduced num: x
localparam RANGE_REDUCED_WIDTH = IN_WIDTH;
localparam RANGE_REDUCED_FRAC_WIDTH = IN_WIDTH - 1;

// Slope: m
localparam SLOPE_WIDTH = 1 + RANGE_REDUCED_FRAC_WIDTH;
localparam SLOPE_FRAC_WIDTH = RANGE_REDUCED_FRAC_WIDTH;

// Mult: mx
localparam MULT_WIDTH = IN_WIDTH + SLOPE_WIDTH;
localparam MULT_FRAC_WIDTH = RANGE_REDUCED_FRAC_WIDTH + SLOPE_FRAC_WIDTH;

// Intercept (need to match mx frac): c
localparam INTERCEPT_FRAC_WIDTH = MULT_FRAC_WIDTH;
localparam INTERCEPT_WIDTH = 2 + INTERCEPT_FRAC_WIDTH; // Needs 2 integer bits

// Output width: mx + c
localparam LPW_WIDTH = MULT_WIDTH + 1;
localparam LPW_FRAC_WIDTH = MULT_FRAC_WIDTH; // == INTERCEPT_FRAC_WIDTH


// Recip width calculation: Need to pad extra 2 * max(intwidth, fracwidth) to
// make sure recip is not shifted out
localparam IN_INT_WIDTH = IN_WIDTH - IN_FRAC_WIDTH;
localparam EXTRA_WIDTH = IN_INT_WIDTH > IN_FRAC_WIDTH ? IN_INT_WIDTH : IN_FRAC_WIDTH;
localparam RECIP_WIDTH = LPW_WIDTH + EXTRA_WIDTH;
localparam RECIP_FRAC_WIDTH = LPW_FRAC_WIDTH;

// Shift num widths
localparam MSB_WIDTH = $clog2(IN_WIDTH);
localparam SHIFT_WIDTH = MSB_WIDTH + 1;

initial begin
    assert (IN_WIDTH > IN_FRAC_WIDTH);
    assert (IN_FRAC_WIDTH >= 2);
    assert (OUT_WIDTH > OUT_FRAC_WIDTH);
    assert (OUT_FRAC_WIDTH >= 2);
end

// -----
// Wires
// -----

logic [RANGE_REDUCED_WIDTH-1:0] range_reduced_num [1:0];
logic [MSB_WIDTH-1:0] msb [2:0];
logic msb_not_found [4:0];
logic range_reduce_out_valid, range_reduce_out_ready;

logic [1:0] frac_top_in, frac_top_out;

logic [MULT_WIDTH-1:0] mult_in, mult_out;
logic mult_out_valid, mult_out_ready;

logic [LPW_WIDTH-1:0] lpw_in_data, lpw_out_data;
logic lpw_out_valid, lpw_out_ready;

logic [SHIFT_WIDTH-1:0] shift_amt_in, shift_amt_out;

logic [RECIP_WIDTH-1:0] recip_in_data, recip_out_data;
logic recip_out_valid, recip_out_ready;

logic [OUT_WIDTH:0] cast_out_data;

logic [OUT_WIDTH-1:0] output_reg_in_data;


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
    .msb_index(msb[0]),
    .not_found(msb_not_found[0]) // if msb_not_found, then x = 0
);

skid_buffer #(
    .DATA_WIDTH(RANGE_REDUCED_WIDTH + MSB_WIDTH + 1)
) range_reduce_reg (
    .clk(clk),
    .rst(rst),
    .data_in({range_reduced_num[0], msb[0], msb_not_found[0]}),
    .data_in_valid(in_valid),
    .data_in_ready(in_ready),
    .data_out({range_reduced_num[1], msb[1], msb_not_found[1]}),
    .data_out_valid(range_reduce_out_valid),
    .data_out_ready(range_reduce_out_ready)
);

assign frac_top_in = range_reduced_num[1][RANGE_REDUCED_WIDTH-2:RANGE_REDUCED_WIDTH-3];

// Multiplication Stage
always_comb begin
    case (frac_top_in)
        2'b00: mult_in = $signed({1'b0, range_reduced_num[1]}) * $signed(slope(1.00, 1.25));
        2'b01: mult_in = $signed({1'b0, range_reduced_num[1]}) * $signed(slope(1.25, 1.50));
        2'b10: mult_in = $signed({1'b0, range_reduced_num[1]}) * $signed(slope(1.50, 1.75));
        2'b11: mult_in = $signed({1'b0, range_reduced_num[1]}) * $signed(slope(1.75, 2.00));
    endcase
end

skid_buffer #(
    .DATA_WIDTH(MULT_WIDTH + 2 + MSB_WIDTH + 1)
) mult_stage_reg (
    .clk(clk),
    .rst(rst),
    .data_in({mult_in, frac_top_in, msb[1], msb_not_found[1]}),
    .data_in_valid(range_reduce_out_valid),
    .data_in_ready(range_reduce_out_ready),
    .data_out({mult_out, frac_top_out, msb[2], msb_not_found[2]}),
    .data_out_valid(mult_out_valid),
    .data_out_ready(mult_out_ready)
);

always_comb begin
    // Add Intercept to Mult
    case (frac_top_out)
        2'b00: lpw_in_data = $signed(mult_out) + $signed({1'b0, intercept(1.00, 1.25)});
        2'b01: lpw_in_data = $signed(mult_out) + $signed({1'b0, intercept(1.25, 1.50)});
        2'b10: lpw_in_data = $signed(mult_out) + $signed({1'b0, intercept(1.50, 1.75)});
        2'b11: lpw_in_data = $signed(mult_out) + $signed({1'b0, intercept(1.75, 2.00)});
    endcase

    // Also convert MSB into a shift amount
    shift_amt_in = IN_FRAC_WIDTH - msb[2];
end

// 001.0000 // shift_amt = 0 // frac = 4, msb_index = 4
// 000.0100 // shift_amt = 2 // frac = 4, msb_index = 2
// 100.0100 // shift_amt = -2 // frac = 4, msb_index = 6

skid_buffer #(
    .DATA_WIDTH(LPW_WIDTH + SHIFT_WIDTH + 1)
) lpw_stage_reg (
    .clk(clk),
    .rst(rst),
    .data_in({lpw_in_data, shift_amt_in, msb_not_found[2]}),
    .data_in_valid(mult_out_valid),
    .data_in_ready(mult_out_ready),
    .data_out({lpw_out_data, shift_amt_out, msb_not_found[3]}),
    .data_out_valid(lpw_out_valid),
    .data_out_ready(lpw_out_ready)
);


always_comb begin
    // Shift stage
    if ($signed(shift_amt_out) >= 0) begin
        recip_in_data = $signed(lpw_out_data) <<< shift_amt_out;
    end else begin
        recip_in_data = $signed(lpw_out_data) >>> -shift_amt_out;
    end
end

skid_buffer #(
    .DATA_WIDTH(RECIP_WIDTH + 1)
) recip_stage_reg (
    .clk(clk),
    .rst(rst),
    .data_in({recip_in_data, msb_not_found[3]}),
    .data_in_valid(lpw_out_valid),
    .data_in_ready(lpw_out_ready),
    .data_out({recip_out_data, msb_not_found[4]}),
    .data_out_valid(recip_out_valid),
    .data_out_ready(recip_out_ready)
);


// TODO: change to unsigned cast
fixed_signed_cast #(
    .IN_WIDTH(RECIP_WIDTH + 1),
    .IN_FRAC_WIDTH(LPW_FRAC_WIDTH),
    .OUT_WIDTH(OUT_WIDTH + 1),
    .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),
    .SYMMETRIC(0),
    .ROUND_FLOOR(1)
) signed_cast (
    .in_data({1'b0, recip_out_data}),
    .out_data(cast_out_data)
);

// Mux between INT_MAX and 1/x result (edge case for 1/0)
assign output_reg_in_data = (msb_not_found[4]) ? '1 : cast_out_data[OUT_WIDTH-1:0];

skid_buffer #(
    .DATA_WIDTH(OUT_WIDTH)
) output_reg (
    .clk(clk),
    .rst(rst),
    .data_in(output_reg_in_data),
    .data_in_valid(recip_out_valid),
    .data_in_ready(recip_out_ready),
    .data_out(out_data),
    .data_out_valid(out_valid),
    .data_out_ready(out_ready)
);

// initial begin
//     // $display("reciprocal(0.25) = %d (%f)", reciprocal(0.25), real'(reciprocal(0.25)) / (2 ** OUT_FRAC_WIDTH));
//     // $display("reciprocal(0.5) = %d (%f)", reciprocal(0.5), real'(reciprocal(0.5)) / (2 ** OUT_FRAC_WIDTH));
//     $display("slope(1.00, 1.25) = %b = -%d", slope(1.00, 1.25), ~slope(1.00, 1.25)+1'b1);
//     $display("slope(1.25, 1.50) = %b = -%d", slope(1.25, 1.50), ~slope(1.25, 1.50)+1'b1);
//     $display("slope(1.50, 1.75) = %b = -%d", slope(1.50, 1.75), ~slope(1.50, 1.75)+1'b1);
//     $display("slope(1.75, 2.00) = %b = -%d", slope(1.75, 2.00), ~slope(1.75, 2.00)+1'b1);
//     $finish;
// end

endmodule
