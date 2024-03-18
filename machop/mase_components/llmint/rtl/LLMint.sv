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
    input logic signed [ORIGINAL_PRECISION-1:0] weights[WEIGHT_DIM_0 * WEIGHT_DIM_1-1:0],
    output logic signed [ORIGINAL_PRECISION-1:0] data_out[TENSOR_SIZE_DIM-1:0]
);

    logic signed [ORIGINAL_PRECISION-1:0] low_precision_masked[TENSOR_SIZE_DIM-1:0];
    logic signed [ORIGINAL_PRECISION-1:0] high_precision_masked[TENSOR_SIZE_DIM-1:0];
    logic signed [2 * REDUCED_PRECISION + $clog2(TENSOR_SIZE_DIM)-1:0] output_linear_low_precision[TENSOR_SIZE_DIM-1:0];
    logic signed [2 * ORIGINAL_PRECISION + $clog2(TENSOR_SIZE_DIM)-1:0] output_linear_high_precision[TENSOR_SIZE_DIM-1:0];

    logic signed [REDUCED_PRECISION-1:0] input_linear_low_precision[TENSOR_SIZE_DIM-1:0];
    logic signed [REDUCED_PRECISION-1:0] quantized_weights[WEIGHT_DIM_0 * WEIGHT_DIM_1-1:0];
    logic signed [ORIGINAL_PRECISION-1:0] high_for_gather[TENSOR_SIZE_DIM-1:0];
    logic signed [ORIGINAL_PRECISION-1:0] low_for_gather[TENSOR_SIZE_DIM-1:0];

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

    // Quantizing the input and weights
    for (genvar i = 0; i < TENSOR_SIZE_DIM; i = i + 1) begin
        assign input_linear_low_precision[i] = low_precision_masked[i] >>> (ORIGINAL_PRECISION - REDUCED_PRECISION);
    end

    for (genvar i = 0; i < WEIGHT_DIM_0 * WEIGHT_DIM_1; i = i + 1) begin
        assign quantized_weights[i] = weights[i] >>> (ORIGINAL_PRECISION - REDUCED_PRECISION);
    end

    /*
    function [REDUCED_PRECISION-1:0] quantize_2D_array(input [ORIGINAL_PRECISION-1:0] signal_array[WEIGHT_DIM_0-1:0][WEIGHT_DIM_1-1:0]);
        integer i, j;
        logic signed [REDUCED_PRECISION-1:0] result [TENSOR_SIZE_DIM-1:0][TENSOR_SIZE_DIM-1:0];
        for (i = 0; i < TENSOR_SIZE_DIM; i = i + 1) begin
            for (j = 0; j < TENSOR_SIZE_DIM; j = j + 1) begin
                result[i][j] = signal_array[i][j] >>> (ORIGINAL_PRECISION - REDUCED_PRECISION);
            end
        end
        return result;
    endfunction*/


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
        .WEIGHT_PARALLELISM_DIM_1(1),
        .DATA_OUT_0_TENSOR_SIZE_DIM_0(TENSOR_SIZE_DIM),
        .DATA_OUT_0_TENSOR_SIZE_DIM_1(1),
        .DATA_OUT_0_PARALLELISM_DIM_1(1)
    )low_precision_linear(
        .clk(clk),
        .rst(rst),

        .data_in_0(input_linear_low_precision),
        .data_in_0_valid(data_in_0_valid_low),

        .weight(quantized_weights),
        .weight_valid(weight_valid_low),

        .data_out_0(output_linear_low_precision),
        .data_out_0_ready(data_out_0_ready_low)       
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
        .WEIGHT_PARALLELISM_DIM_1(1),
        .DATA_OUT_0_TENSOR_SIZE_DIM_0(TENSOR_SIZE_DIM),
        .DATA_OUT_0_TENSOR_SIZE_DIM_1(1),
        .DATA_OUT_0_PARALLELISM_DIM_1(1)
    )high_precision_linear(
        .clk(clk),
        .rst(rst),

        .data_in_0(high_precision_masked),
        .data_in_0_valid(data_in_0_valid_high),

        .weight(weights),
        .weight_valid(weight_valid_high),

        .data_out_0(output_linear_high_precision),
        .data_out_0_ready(data_out_0_ready_high)
    );

    assign data_in_0_valid_low = 1'b1;
    assign weight_valid_low = 1'b1;
    assign data_out_0_ready_low = 1'b1;
    assign data_in_0_valid_high = 1'b1;
    assign weight_valid_high = 1'b1;
    assign data_out_0_ready_high = 1'b1;

    for (genvar i = 0; i < TENSOR_SIZE_DIM; i = i + 1) begin
        assign high_for_gather[i] = output_linear_high_precision[i] >>> (2 * ORIGINAL_PRECISION + $clog2(TENSOR_SIZE_DIM) - ORIGINAL_PRECISION);
    end

    for (genvar i = 0; i < TENSOR_SIZE_DIM; i = i + 1) begin
        assign low_for_gather[i] = output_linear_low_precision[i] >>> (2 * REDUCED_PRECISION + $clog2(TENSOR_SIZE_DIM) - ORIGINAL_PRECISION);
    end

    gather#(
        .TENSOR_SIZE_DIM(TENSOR_SIZE_DIM),
        .PRECISION(ORIGINAL_PRECISION)
    )gather(
        .mat_a(high_for_gather),
        .mat_b(low_for_gather),
        .mat_sum(data_out) 
    );
    

endmodule