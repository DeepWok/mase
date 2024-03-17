`timescale 1ns / 1ps


module LLMint #(
    parameter ORIGINAL_PRECISION = 16,
    parameter REDUCED_PRECISION_0 = 8,
    parameter TENSOR_SIZE_DIM = 4,
    parameter HIGH_SLOTS = 2,
    parameter THRESHOLD = 6
) (
    input logic signed [ORIGINAL_PRECISION-1:0] data_in[TENSOR_SIZE_DIM-1:0],
    output logic signed [ORIGINAL_PRECISION-1:0] data_out[TENSOR_SIZE_DIM-1:0]
);

    wire [TENSOR_SIZE_DIM-1:0] low_precision_masked;
    wire [TENSOR_SIZE_DIM-1:0] high_precision_masked;
    wire [TENSOR_SIZE_DIM-1:0] output_linear_low_precision;
    wire [TENSOR_SIZE_DIM-1:0] output_linear_high_precision;

    scatter_threshold#(
        .PRECISION(ORIGINAL_PRECISION),
        .TENSOR_SIZE_DIM(TENSOR_SIZE_DIM),
        .HIGH_SLOTS(HIGH_SLOTS),
        .THRESHOLD(THRESHOLD)
    )scatter(
        .data_in(data_in),
        .o_high_precision(high_precision_masked),
        .o_low_precision(low_precision_masked)
    );

    fixed_linear#(
        .PRECISION(ORIGINAL_PRECISION),
        .REDUCED_PRECISION(REDUCED_PRECISION_0),
        .TENSOR_SIZE_DIM(TENSOR_SIZE_DIM)
    )low_precision_linear(
        .data_in(),
        .data_out_0(output_linear_low_precision)
    );

    fixed_linear#(
        .PRECISION(ORIGINAL_PRECISION),
        .REDUCED_PRECISION(REDUCED_PRECISION_0),
        .TENSOR_SIZE_DIM(TENSOR_SIZE_DIM)
    )high_precision_linear(
        .data_in(),
        .data_out_0(output_linear_high_precision)
    );


    gather#(
        .PRECISION(ORIGINAL_PRECISION),
        .TENSOR_SIZE_DIM(TENSOR_SIZE_DIM),
    )gather(
        .mat_a(),
        .mat_b(),
        .mat_sum(data_out) 
    );
    

endmodule