`timescale 1ns / 1ps


module LLMint #(
    parameter ORIGINAL_PRECISION = 16,
    parameter REDUCED_PRECISION = 8,
    parameter TENSOR_SIZE_DIM = 4,
    parameter WEIGHT_DIM_0 = TENSOR_SIZE_DIM,
    parameter WEIGHT_DIM_1 = TENSOR_SIZE_DIM,
    parameter HIGH_SLOTS = 2,
    parameter THRESHOLD = 6
) (
    input clk,
    input rst,
    input logic signed [ORIGINAL_PRECISION-1:0] data_in[TENSOR_SIZE_DIM-1:0],
    input logic signed [ORIGINAL_PRECISION-1:0] weights[WEIGHT_DIM_0-1:0][WEIGHT_DIM_1-1:0],
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

    function [REDUCED_PRECISION-1:0] quantize_array(input [ORIGINAL_PRECISION-1:0] signal_array[TENSOR_SIZE_DIM-1:0]);
        integer i;
        [REDUCED_PRECISION-1:0] result[TENSOR_SIZE_DIM-1:0];
        for (i = 0; i < TENSOR_SIZE_DIM; i = i + 1) begin
            result[i] = signal_array[i] >>> (ORIGINAL_PRECISION - REDUCED_PRECISION);
        end
        return result;
    endfunction

    function [REDUCED_PRECISION-1:0] quantize_2D_array(input [ORIGINAL_PRECISION-1:0] signal_array[WEIGHT_DIM_0-1:0][WEIGHT_DIM_1-1:0]);
        integer i, j;
        [REDUCED_PRECISION-1:0] result[TENSOR_SIZE_DIM-1:0][TENSOR_SIZE_DIM-1:0];
        for (i = 0; i < TENSOR_SIZE_DIM; i = i + 1) begin
            for (j = 0; j < TENSOR_SIZE_DIM; j = j + 1) begin
                result[i][j] = signal_array[i][j] >>> (ORIGINAL_PRECISION - REDUCED_PRECISION);
            end
        end
        return result;
    endfunction

    assign input_linear_low_precision = quantize_array(low_precision_masked);
    assign quantized_weights = quantize_2D_array(weights);

    fixed_linear#(
        .DATA_IN_0_PRECISION_0(REDUCED_PRECISION),
        .DATA_IN_0_TENSOR_SIZE_DIM_0(TENSOR_SIZE_DIM),
        .DATA_IN_0_TENSOR_SIZE_DIM_1(1),
        .DATA_IN_0_PARALLELISM_DIM_0(TENSOR_SIZE_DIM),
        .DATA_IN_0_PARALLELISM_DIM_1(1),
        .WEIGHT_PRECISION_0(REDUCED_PRECISION),
        .WEIGHT_TENSOR_SIZE_DIM_0(WEIGHT_DIM_0),
        .WEIGHT_TENSOR_SIZE_DIM_1(WEIGHT_DIM_1),
        .WEIGHT_PARALLELISM_DIM_0(WEIGHT_DIM_0),
        .WEIGHT_PARALLELISM_DIM_1(WEIGHT_DIM_1),
        .DATA_OUT_0_TENSOR_SIZE_DIM_0(TENSOR_SIZE_DIM),
        .DATA_OUT_0_TENSOR_SIZE_DIM_1(1),
        .DATA_OUT_0_PARALLELISM_DIM_1(1),
    )low_precision_linear(
        .clk(clk),
        .rst(rst),

        .data_in_0(input_linear_low_precision),
        .data_in_0_valid(1'b1),
        .data_in_0_ready(),

        .weight(quantized_weights),
        .weight_valid(1'b1),
        .weight_ready(),

        .data_out_0(output_linear_low_precision),
        .data_out_0_valid(),
        .data_out_0_ready(1'b1),        
    );

    fixed_linear#(
        .DATA_IN_0_PRECISION_0(ORIGINAL_PRECISION),
        .DATA_IN_0_TENSOR_SIZE_DIM_0(TENSOR_SIZE_DIM),
        .DATA_IN_0_TENSOR_SIZE_DIM_1(1),
        .DATA_IN_0_PARALLELISM_DIM_0(TENSOR_SIZE_DIM),
        .DATA_IN_0_PARALLELISM_DIM_1(1),
        .WEIGHT_PRECISION_0(ORIGINAL_PRECISION),
        .WEIGHT_TENSOR_SIZE_DIM_0(WEIGHT_DIM_0),
        .WEIGHT_TENSOR_SIZE_DIM_1(WEIGHT_DIM_1),
        .WEIGHT_PARALLELISM_DIM_0(WEIGHT_DIM_0),
        .WEIGHT_PARALLELISM_DIM_1(WEIGHT_DIM_1),
        .DATA_OUT_0_TENSOR_SIZE_DIM_0(TENSOR_SIZE_DIM),
        .DATA_OUT_0_TENSOR_SIZE_DIM_1(1),
        .DATA_OUT_0_PARALLELISM_DIM_1(1),
    )high_precision_linear(
        .clk(clk),
        .rst(rst),

        .data_in_0(high_precision_masked),
        .data_in_0_valid(1'b1),
        .data_in_0_ready(),

        .weight(weights),
        .weight_valid(1'b1),
        .weight_ready(),

        .data_out_0(output_linear_high_precision),
        .data_out_0_valid(),
        .data_out_0_ready(1'b1),
    );

    gather#(
        .PRECISION(ORIGINAL_PRECISION),
        .TENSOR_SIZE_DIM(TENSOR_SIZE_DIM),
    )gather(
        .mat_a(output_linear_high_precision),
        .mat_b(output_linear_low_precision),
        .mat_sum(data_out) 
    );
    

endmodule