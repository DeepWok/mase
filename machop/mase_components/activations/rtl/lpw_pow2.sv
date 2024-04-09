/*
Module      : lpw_pow2
Description : This module implements 2^x with linear piecewise approximation.

              Uses 4 linear pieces.
*/

module lpw_pow2 #(
    parameter DATA_WIDTH = 8,
    parameter DATA_FRAC_WIDTH = 4,
    parameter LUT_FRAC_WIDTH = 8
) (
    input  logic clk,
    input  logic rst,

    input  logic [DATA_WIDTH-1:0] in_data,
    input  logic                  in_valid,
    output logic                  in_ready,

    output logic [DATA_WIDTH-1:0] out_data,
    output logic                  out_valid,
    input  logic                  out_ready
);

// -----
// Parameters
// -----

// Input: x
localparam INT_WIDTH = DATA_WIDTH - DATA_FRAC_WIDTH;

// Slope: m
localparam SLOPE_WIDTH = 2 + LUT_FRAC_WIDTH;

// Mult: mx
localparam MULT_WIDTH = DATA_WIDTH + SLOPE_WIDTH;
localparam MULT_FRAC_WIDTH = DATA_FRAC_WIDTH + LUT_FRAC_WIDTH;

// Intercept (need to match mx frac): c
localparam INTERCEPT_FRAC_WIDTH = MULT_FRAC_WIDTH;
localparam INTERCEPT_WIDTH = 2 + INTERCEPT_FRAC_WIDTH;

// Output width: mx + c
localparam LPW_WIDTH = MULT_WIDTH + 1;
localparam LPW_FRAC_WIDTH = MULT_FRAC_WIDTH; // == INTERCEPT_FRAC_WIDTH

localparam LPW_WIDTH = MULT_WIDTH;
localparam LPW_FRAC_WIDTH = DATA_WIDTH - 2;

// PARAMETERS BELOW ONLY USED IN 1/2-BIT CASE
// Output result of 2^[0,1] is in [1,2] which requires 2 integer bits
localparam LUT_WIDTH = LUT_FRAC_WIDTH + 2;


initial begin
    assert (INT_WIDTH > 0); // Untested for 0 int width
    assert (DATA_WIDTH > 0);
    assert (DATA_FRAC_WIDTH >= 0);
end

// Wires
logic [INT_WIDTH-1:0] in_data_int; // Q INT.0
logic [DATA_FRAC_WIDTH-1:0] in_data_frac; // Q 0.FRAC

logic [DATA_WIDTH-1:0] result_data;
logic result_valid, result_ready;


// Function to generate LUT (Only used 1/2-bit case)
function logic [LUT_WIDTH-1:0] pow2_func (real x);
    real res, res_shifted;
    int res_int;
    res = 2.0 ** x;

    // Output cast
    res_shifted = res * (2 ** LUT_FRAC_WIDTH);
    res_int = int'(res_shifted);
    // $display("res = %f", res);
    // $display("res_shifted = %f (%f)", res_shifted, res_shifted / (2 ** LPW_SHIFT_AMT));
    // $display("res_int = %d (%f)", res_int, real'(res_int) / (2 ** LPW_SHIFT_AMT));
    return LUT_WIDTH'(res_int);
endfunction

// Function to generate slope variable (m)
function logic [SLOPE_WIDTH-1:0] slope (real x1, real x2);
    real y1, y2, res, res_shifted;
    int res_int;
    y1 = 2.0 ** x1;
    y2 = 2.0 ** x2;
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
    y1 = 2.0 ** x1;
    y2 = 2.0 ** x2;
    m = (y2 - y1) / (x2 - x1);
    res = y1 - (m * x1);

    // Output cast
    res_shifted = res * (2 ** INTERCEPT_FRAC_WIDTH);
    res_int = int'(res_shifted);
    return INTERCEPT_WIDTH'(res_int);
endfunction

// -----
// Logic
// -----

assign {in_data_int, in_data_frac} = in_data;

generate
if (DATA_FRAC_WIDTH <= 1) begin : one_bit_frac

    logic [LUT_WIDTH-1:0] lpw_result; // Q 0.(FRAC+INT)
    always_comb begin
        case (in_data_frac)
            1'b0: lpw_result = pow2_func(0.0);
            1'b1: lpw_result = pow2_func(0.5);
        endcase
        result_data = lpw_result << in_data_int; // TODO: SIGNED
        result_valid = in_valid;
        in_ready = result_ready;
    end

end else if (DATA_FRAC_WIDTH == 2) begin : two_bit_frac

    logic [LUT_WIDTH-1:0] lpw_result; // Q 0.(FRAC+INT)
    always_comb begin
        case (in_data_frac)
            2'b00: lpw_result = pow2_func(0.0);
            2'b01: lpw_result = pow2_func(0.25);
            2'b10: lpw_result = pow2_func(0.5);
            2'b11: lpw_result = pow2_func(0.75);
        endcase
        result_data = lpw_result << in_data_int; // TODO: SIGNED
        result_valid = in_valid;
        in_ready = result_ready;
    end

end else begin : lpw_approx

    localparam MULT_WIDTH = LPW_WIDTH + DATA_FRAC_WIDTH;
    localparam ADDITION_WIDTH = MULT_WIDTH + 1;

    // Split out the top two bits of the frac again to figure out which
    // piecewise part if lies on
    logic [1:0] frac_top_in, frac_top_out;
    assign frac_top_in = in_data_frac[DATA_FRAC_WIDTH-1:DATA_FRAC_WIDTH-2];

    logic [MULT_WIDTH-1:0] mult_in, mult_out;
    logic mult_out_valid, mult_out_ready;

    logic [ADDITION_WIDTH-1:0] lpw_addition;

    always_comb begin
        // Multiplication Stage
        case (frac_top_in)
            2'b00: mult_in = in_data_frac * slope(0.00, 0.25);
            2'b01: mult_in = in_data_frac * slope(0.25, 0.50);
            2'b10: mult_in = in_data_frac * slope(0.50, 0.75);
            2'b11: mult_in = in_data_frac * slope(0.75, 1.00);
        endcase

        // Add Intercept & Shift stage
        case (frac_top_out)
            2'b00: lpw_addition = mult_out + intercept(0.00, 0.25);
            2'b01: lpw_addition = mult_out + intercept(0.25, 0.50);
            2'b10: lpw_addition = mult_out + intercept(0.50, 0.75);
            2'b11: lpw_addition = mult_out + intercept(0.75, 1.00);
        endcase
        result_data = lpw_addition << in_data_int; // TODO: SIGNED
        result_valid = mult_out_valid;
        mult_out_ready = result_ready;
    end

    // Multiplication Reg
    skid_buffer #(
        .DATA_WIDTH(MULT_WIDTH + 2) // Buffer multiplication & top frac bits
    ) out_reg (
        .clk(clk),
        .rst(rst),
        .data_in({mult_in, frac_top_in}),
        .data_in_valid(mult_in_valid),
        .data_in_ready(mult_in_ready),
        .data_out({mult_out, frac_top_out}),
        .data_out_valid(mult_out_valid),
        .data_out_ready(mult_out_ready)
    );

end

endgenerate


// Output Register
skid_buffer #(
    .DATA_WIDTH(DATA_WIDTH)
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


initial begin
    $display("lookup(0.0) = %d", pow2_func(0.0));
    $display("lookup(0.5) = %d", pow2_func(0.5));
    $display("lookup(0.75) = %d", pow2_func(0.75));
    $finish;
end

endmodule
