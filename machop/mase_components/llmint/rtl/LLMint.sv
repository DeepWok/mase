`timescale 1ns / 1ps


module LLMint #(
    parameter ORIGINAL_PRECISION = 32,
    parameter REDUCED_PRECISION = 8,
    parameter TENSOR_SIZE_DIM = 4,
    parameter WEIGHT_DIM_0 = TENSOR_SIZE_DIM,
    parameter WEIGHT_DIM_1 = TENSOR_SIZE_DIM,
    parameter HIGH_SLOTS = 2,
    parameter THRESHOLD = 6,
    parameter DESIGN = 1

) (
    input clk,
    input rst,
    input logic data_in_valid,
    output logic data_in_ready,
    input logic weight_valid,
    output logic weight_ready,
    input logic q_weight_valid,
    output logic q_weight_ready,
    input logic data_out_ready,
    output logic data_out_valid,
    input logic signed [ORIGINAL_PRECISION-1:0] data_in[TENSOR_SIZE_DIM-1:0],
    // We combine weights and quantized weights into a single array
    input logic signed [ORIGINAL_PRECISION-1:0] weights[WEIGHT_DIM_0 * WEIGHT_DIM_1-1:0],
    input logic signed [REDUCED_PRECISION-1:0] q_weights[WEIGHT_DIM_0 * WEIGHT_DIM_1-1:0],
    output logic signed [ORIGINAL_PRECISION-1:0] data_out[TENSOR_SIZE_DIM-1:0],
    //output logic signed [2 * REDUCED_PRECISION + $clog2(TENSOR_SIZE_DIM)-1:0] data_out[TENSOR_SIZE_DIM-1:0],
    output logic signed [REDUCED_PRECISION-1:0] input_linear_low_precision[TENSOR_SIZE_DIM-1:0],
    output logic signed [ORIGINAL_PRECISION-1:0] high_precision_masked[TENSOR_SIZE_DIM-1:0],
    output logic signed [ORIGINAL_PRECISION-1:0] low_precision_masked[TENSOR_SIZE_DIM-1:0]
);

  //logic  [ORIGINAL_PRECISION-1:0] low_precision_masked[TENSOR_SIZE_DIM-1:0];
  //logic  [ORIGINAL_PRECISION-1:0] high_precision_masked[TENSOR_SIZE_DIM-1:0];
  logic signed [2 * REDUCED_PRECISION + $clog2(
TENSOR_SIZE_DIM
)-1:0] output_linear_low_precision[TENSOR_SIZE_DIM-1:0];
  logic signed [2 * ORIGINAL_PRECISION + $clog2(
TENSOR_SIZE_DIM
)-1:0] output_linear_high_precision[TENSOR_SIZE_DIM-1:0];

  //logic signed [REDUCED_PRECISION-1:0] input_linear_low_precision[TENSOR_SIZE_DIM-1:0];
  logic signed [ORIGINAL_PRECISION-1:0] high_for_gather[TENSOR_SIZE_DIM-1:0];
  //logic signed [2 * REDUCED_PRECISION + $clog2(TENSOR_SIZE_DIM)-1:0] high_for_gather[TENSOR_SIZE_DIM-1:0];
  logic signed [ORIGINAL_PRECISION-1:0] low_for_gather[TENSOR_SIZE_DIM-1:0];


  scatter_threshold #(
      .PRECISION(ORIGINAL_PRECISION),
      .TENSOR_SIZE_DIM(TENSOR_SIZE_DIM),
      .HIGH_SLOTS(HIGH_SLOTS),
      .THRESHOLD(THRESHOLD),
      .DESIGN(DESIGN)
  ) scatter (
      .data_in(data_in),
      .o_high_precision(high_precision_masked),
      .o_low_precision(low_precision_masked)
  );

  // Quantizing the input and weights
  for (genvar i = 0; i < TENSOR_SIZE_DIM; i = i + 1) begin
    assign input_linear_low_precision[i] = low_precision_masked[i];
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

  logic data_in_ready_low_precision;
  logic data_in_ready_high_precision;

  assign data_in_ready = data_in_ready_low_precision & data_in_ready_high_precision;

  logic data_out_valid_low_precision, data_out_valid_high_precision;
  assign data_out_valid = data_out_valid_low_precision & data_out_valid_high_precision;

  logic bias_ready;
  logic bias_valid;
  logic [ORIGINAL_PRECISION-1:0] bias;

  assign bias_ready = 1;
  assign bias_valid = 1;
  assign bias = 0;

  fixed_linear #(
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
  ) low_precision_linear (
      .clk(clk),
      .rst(rst),

      .data_in_0(input_linear_low_precision),
      .data_in_0_valid(data_in_valid),
      .data_in_0_ready(data_in_ready_low_precision),

      .weight(q_weights),
      .weight_valid(q_weight_valid),
      .weight_ready(q_weight_ready),

      .bias(bias),
      .bias_valid(bias_valid),
      .bias_ready(bias_ready),

      .data_out_0(output_linear_low_precision),
      .data_out_0_ready(data_out_ready),
      .data_out_0_valid(data_out_valid_low_precision)

  );

  fixed_linear #(
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
  ) high_precision_linear (
      .clk(clk),
      .rst(rst),

      .data_in_0(high_precision_masked),
      .data_in_0_valid(data_in_valid),
      .data_in_0_ready(data_in_ready_high_precision),

      .weight(weights),
      .weight_valid(weight_valid),
      .weight_ready(weight_ready),

      .bias(bias),
      .bias_valid(bias_valid),
      .bias_ready(bias_ready),

      .data_out_0(output_linear_high_precision),
      .data_out_0_ready(data_out_ready),
      .data_out_0_valid(data_out_valid_high_precision)
  );

  for (genvar i = 0; i < TENSOR_SIZE_DIM; i = i + 1) begin
    assign high_for_gather[i] = output_linear_high_precision[i];
  end

  for (genvar i = 0; i < TENSOR_SIZE_DIM; i = i + 1) begin
    assign low_for_gather[i] = output_linear_low_precision[i];
  end

  gather #(
      .TENSOR_SIZE_DIM(TENSOR_SIZE_DIM),
      .PRECISION(ORIGINAL_PRECISION)
      //.PRECISION(2 * REDUCED_PRECISION + $clog2(TENSOR_SIZE_DIM))
  ) gather (
      .mat_a  (high_for_gather),
      .mat_b  (low_for_gather),
      //.mat_b(output_linear_low_precision),
      .mat_sum(data_out)
  );


endmodule
