/*
Module      : group_norm_2d
Description : This module calculates the generalised group norm.
              https://arxiv.org/abs/1803.08494v3

              This module can be easily trivially specialised into layer norm or
              instance norm by setting the GROUP_CHANNELS param to equal C or 1
              respectively.

              Group norm is independent of batch size, so the input shape is:
              (GROUP, DEPTH_DIM1 * DEPTH_DIM0, COMPUTE_DIM1 * COMPUTE_DIM0)
    assume we flatten layernorm.normalized_shape to, and then calculate it
    so it actually is dim_0 = prod(normalized_shape), x.reshape(-1, dim0), out = norm(dim_0)(x)
    2d means parallelism here
*/

`timescale 1ns / 1ps
module layer_norm_2d #(
    // Dimensions
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 2,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 2,

    // Data widths
    parameter DATA_IN_0_PRECISION_0        = 8,
    parameter DATA_IN_0_PRECISION_1        = 4,
    parameter WEIGHT_PRECISION_0           = 8,
    parameter WEIGHT_PRECISION_1           = 4,
    parameter BIAS_PRECISION_0             = 8,
    parameter BIAS_PRECISION_1             = 4,
    parameter ELEMENTWISE_AFFINE           = 0,
    parameter HAS_BIAS                     = 0,
    parameter ISQRT_IN_PRECISION_0         = 8,
    parameter ISQRT_IN_PRECISION_1         = 8,
    parameter ISQRT_OUT_PRECISION_0        = 8,
    parameter ISQRT_OUT_PRECISION_1        = 4,
    parameter BIAS_TENSOR_SIZE_DIM_0       = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter BIAS_PARALLELISM_DIM_0       = DATA_IN_0_PARALLELISM_DIM_0,
    parameter BIAS_TENSOR_SIZE_DIM_1       = 1,
    parameter BIAS_PARALLELISM_DIM_1       = 1,
    parameter WEIGHT_TENSOR_SIZE_DIM_0     = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter WEIGHT_PARALLELISM_DIM_0     = DATA_IN_0_PARALLELISM_DIM_0,
    parameter WEIGHT_TENSOR_SIZE_DIM_1     = 1,
    parameter WEIGHT_PARALLELISM_DIM_1     = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,
    parameter DATA_OUT_0_PRECISION_0       = 8,
    parameter DATA_OUT_0_PRECISION_1       = 4
) (
    input logic clk,
    input logic rst,

    input  logic [DATA_IN_0_PRECISION_0-1:0] data_in_0 [DATA_IN_0_PARALLELISM_DIM_1*DATA_IN_0_PARALLELISM_DIM_0-1:0],
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    input  logic [WEIGHT_PRECISION_0-1:0] weight      [DATA_IN_0_PARALLELISM_DIM_0 - 1 : 0],
    input  logic                          weight_valid,
    output logic                          weight_ready,

    input  logic [BIAS_PRECISION_0-1:0] bias      [DATA_IN_0_PARALLELISM_DIM_0 - 1 : 0],
    input  logic                        bias_valid,
    output logic                        bias_ready,

    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_IN_0_PARALLELISM_DIM_1*DATA_IN_0_PARALLELISM_DIM_0-1:0],
    output logic data_out_0_valid,
    input logic data_out_0_ready
);
  logic [DATA_IN_0_PARALLELISM_DIM_1 - 1:0] parallel_norm_in_ready, parallel_norm_out_valid;
  logic join_out_valid, join_out_ready;
  logic [DATA_OUT_0_PRECISION_0 - 1:0] norm_out [DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1 - 1:0];
  localparam AFFINE_PRECISION_0 = DATA_OUT_0_PRECISION_0 + WEIGHT_PRECISION_0 + 1;
  localparam AFFINE_PRECISION_1 = DATA_OUT_0_PRECISION_1 + WEIGHT_PRECISION_1;
  logic [AFFINE_PRECISION_0 -1:0] uncast_data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1 - 1:0];
  logic [AFFINE_PRECISION_0 - 1:0] casted_bias[DATA_OUT_0_PARALLELISM_DIM_0-1:0];
  logic [  BIAS_PRECISION_0 - 1:0] bias_buffered  [DATA_IN_0_PARALLELISM_DIM_0 - 1 : 0];
  logic [WEIGHT_PRECISION_0 - 1:0] weight_buffered[DATA_IN_0_PARALLELISM_DIM_0 - 1 : 0];
  logic bias_buffered_valid, bias_buffered_ready, weight_buffered_ready, weight_buffered_valid;
  for (genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_1; i++) begin : parallel_dim_1
    layer_norm_1d #(
        .DATA_IN_0_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0),
        .DATA_IN_0_PARALLELISM_DIM_0(DATA_IN_0_PARALLELISM_DIM_0),
        // Data widths
        .DATA_IN_0_PRECISION_0(DATA_IN_0_PRECISION_0),
        .DATA_IN_0_PRECISION_1(DATA_IN_0_PRECISION_1),
        .ISQRT_IN_PRECISION_0(ISQRT_IN_PRECISION_0),
        .ISQRT_IN_PRECISION_1(ISQRT_IN_PRECISION_1),
        .ISQRT_OUT_PRECISION_0(ISQRT_OUT_PRECISION_0),
        .ISQRT_OUT_PRECISION_1(ISQRT_OUT_PRECISION_1),
        .DATA_OUT_0_TENSOR_SIZE_DIM_0(DATA_OUT_0_TENSOR_SIZE_DIM_0),
        .DATA_OUT_0_PARALLELISM_DIM_0(DATA_OUT_0_PARALLELISM_DIM_0),
        .DATA_OUT_0_PRECISION_0(DATA_OUT_0_PRECISION_0),
        .DATA_OUT_0_PRECISION_1(DATA_OUT_0_PRECISION_1)
    ) layer_norm_inst (
        .clk,
        .rst,
        .data_in_0(data_in_0[i*DATA_IN_0_PARALLELISM_DIM_0 + DATA_IN_0_PARALLELISM_DIM_0 - 1:  i*DATA_IN_0_PARALLELISM_DIM_0]),
        .data_in_0_valid(data_in_0_valid),
        .data_in_0_ready(parallel_norm_in_ready[i]),
        .data_out_0(norm_out[i*DATA_IN_0_PARALLELISM_DIM_0 + DATA_IN_0_PARALLELISM_DIM_0 - 1:  i*DATA_IN_0_PARALLELISM_DIM_0]),
        .data_out_0_valid(parallel_norm_out_valid[i]),
        .data_out_0_ready(join_out_ready)
    );
  end
  assign data_in_0_ready = parallel_norm_in_ready[0];
  assign join_out_valid  = parallel_norm_out_valid[0];
  input_buffer #(
      .DATA_WIDTH (BIAS_PRECISION_0),
      .IN_NUM     (DATA_IN_0_PARALLELISM_DIM_0),
      .REPEAT     (DATA_IN_0_TENSOR_SIZE_DIM_1 / DATA_IN_0_PARALLELISM_DIM_1),
      .BUFFER_SIZE(DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0)
  ) bias_buffer_inst (
      .clk,
      .rst,

      // Input streaming port
      .data_in(bias),
      .data_in_valid(bias_valid),
      .data_in_ready(bias_ready),

      // Output streaming port
      .data_out(bias_buffered),
      .data_out_valid(bias_buffered_valid),
      .data_out_ready(bias_buffered_ready)
  );
  input_buffer #(
      .DATA_WIDTH (WEIGHT_PRECISION_0),
      .IN_NUM     (DATA_IN_0_PARALLELISM_DIM_0),
      .REPEAT     (DATA_IN_0_TENSOR_SIZE_DIM_1 / DATA_IN_0_PARALLELISM_DIM_1),
      .BUFFER_SIZE(DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0)
  ) weight_buffer_inst (
      .clk,
      .rst,

      // Input streaming port
      .data_in(weight),
      .data_in_valid(weight_valid),
      .data_in_ready(weight_ready),

      // Output streaming port
      .data_out(weight_buffered),
      .data_out_valid(weight_buffered_valid),
      .data_out_ready(weight_buffered_ready)
  );
  if (ELEMENTWISE_AFFINE == 1) begin
    logic wd_valid, wd_ready;
    join2 weight_data_join_inst (
        .data_in_valid ({weight_buffered_valid, join_out_valid}),
        .data_in_ready ({weight_buffered_ready, join_out_ready}),
        .data_out_valid(wd_valid),
        .data_out_ready(wd_ready)
    );
    logic [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1 - 1:0]
        parallel_wd_ready, parallel_bias_ready, parallel_data_out_0_valid;
    assign bias_buffered_ready = parallel_bias_ready[0];
    assign wd_ready = parallel_wd_ready[0];
    assign data_out_0_valid = parallel_data_out_0_valid[0];
    for (genvar i = 0; i < DATA_OUT_0_PARALLELISM_DIM_1; i++) begin : affine_parallel_dim1
      for (genvar j = 0; j < DATA_OUT_0_PARALLELISM_DIM_0; j++) begin : affine_parallel_dim0
        localparam int k = i * DATA_IN_0_PARALLELISM_DIM_0 + j;
        if (HAS_BIAS == 1) begin
          join2 wd_bias_join_inst (
              .data_in_valid ({wd_valid, bias_buffered_valid}),
              .data_in_ready ({parallel_wd_ready[k], parallel_bias_ready[k]}),
              .data_out_valid(parallel_data_out_0_valid[k]),
              .data_out_ready(data_out_0_ready)
          );
          if (i==0) begin
            fixed_signed_cast #(
                .IN_WIDTH(BIAS_PRECISION_0),
                .IN_FRAC_WIDTH(BIAS_PRECISION_1),
                .OUT_WIDTH(AFFINE_PRECISION_0),
                .OUT_FRAC_WIDTH(AFFINE_PRECISION_1),
                .SYMMETRIC(0),
                .ROUND_FLOOR(1)
            ) variance_cast_i (
                .in_data (bias_buffered[j]),
                .out_data(casted_bias[j])
            );
          end
          assign uncast_data_out_0[k] = $signed(
              norm_out[k]
          ) * $signed(
              weight_buffered[j]
          ) + $signed(
              casted_bias[j]
          );
        end else begin
          assign parallel_wd_ready[k] = data_out_0_ready;
          assign parallel_data_out_0_valid[k] = wd_valid;
          assign parallel_bias_ready[k] = 1;
          assign uncast_data_out_0[k] = $signed(norm_out[k]) * $signed(weight_buffered[j]);
        end
        fixed_signed_cast #(
            .IN_WIDTH(AFFINE_PRECISION_0),
            .IN_FRAC_WIDTH(AFFINE_PRECISION_1),
            .OUT_WIDTH(DATA_OUT_0_PRECISION_0),
            .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1),
            .SYMMETRIC(0),
            .ROUND_FLOOR(1)
        ) variance_cast_i (
            .in_data (uncast_data_out_0[k]),
            .out_data(data_out_0[k])
        );
      end
    end
  end else begin
    assign join_out_ready = data_out_0_ready;
    assign data_out_0_valid = join_out_valid;
    assign data_out_0 = norm_out;
  end
endmodule
