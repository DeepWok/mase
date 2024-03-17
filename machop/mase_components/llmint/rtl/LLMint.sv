`timescale 1ns / 1ps


module LLMint #(
    parameter ORIGINAL_PRECISION = 16,
    parameter REDUCED_PRECISION_0 = 8,
    parameter TENSOR_SIZE_DIM = 4,
    parameter HIGH_SLOTS = 2,
    parameter THRESHOLD = 6
) (
    input [ORIGINAL_PRECISION-1:0] data_in[TENSOR_SIZE_DIM-1:0],
    output logic [ORIGINAL_PRECISION-1:0] data_out[TENSOR_SIZE_DIM-1:0]
);

    scatter_threshold#(
        .PRECISION(ORIGINAL_PRECISION),
        .TENSOR_SIZE_DIM(TENSOR_SIZE_DIM),
        .HIGH_SLOTS(HIGH_SLOTS),
        .THRESHOLD(THRESHOLD)
    )masker(
        .data_in(data_in),
        .o_high_precision(),
        .o_low_precision() 
    );



    gather#(
        .PRECISION(ORIGINAL_PRECISION),
        .TENSOR_SIZE_DIM(TENSOR_SIZE_DIM),
    )masker(
        .mat_a(),
        .mat_b(),
        .mat_sum(data_out) 
    );
    

endmodule