/*
Module      : fixed_softermax
Description : This module implements softermax.
              https://arxiv.org/abs/2103.09301

              It depends on the "softermax_local_window" and
              "softermax_global_norm" modules.
*/
`timescale 1ns / 1ps
module fixed_softermax #(
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 4,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 10,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,

    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 4,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 10,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1
) (
    input logic clk,
    input logic rst,

    input  logic [DATA_IN_0_PRECISION_0-1:0]  data_in_0 [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_0_valid,
    input logic data_out_0_ready
);

  // * Declarations
  // * =================================================================

  logic [DATA_IN_0_PRECISION_0-1:0] in_data_unflattened [DATA_IN_0_PARALLELISM_DIM_1-1:0] [DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic [DATA_OUT_0_PRECISION_0-1:0] out_data_unflattened [DATA_OUT_0_PARALLELISM_DIM_1-1:0] [DATA_OUT_0_PARALLELISM_DIM_0-1:0];

  logic [DATA_IN_0_PARALLELISM_DIM_1-1:0] in_data_valid;
  logic [DATA_IN_0_PARALLELISM_DIM_1-1:0] in_data_ready;
  logic [DATA_IN_0_PARALLELISM_DIM_1-1:0] out_data_valid;
  logic [DATA_IN_0_PARALLELISM_DIM_1-1:0] out_data_ready;

  // * Instances
  // * =================================================================

  // * Split handshake signals into the rows
  split_n #(
      .N(DATA_IN_0_PARALLELISM_DIM_1)
  ) split_n_i (
      .data_in_valid (data_in_0_valid),
      .data_in_ready (data_in_0_ready),
      .data_out_valid(in_data_valid),
      .data_out_ready(in_data_ready)
  );

  // * Softermax 1d instance for each row
  for (genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_1; i++) begin

    assign in_data_unflattened [i] = data_in_0 [(i + 1) * DATA_IN_0_PARALLELISM_DIM_0 - 1 : i * DATA_IN_0_PARALLELISM_DIM_0];

    fixed_softermax_1d #(
        .TOTAL_DIM     (DATA_IN_0_TENSOR_SIZE_DIM_0),
        .PARALLELISM   (DATA_IN_0_PARALLELISM_DIM_0),
        .IN_WIDTH      (DATA_IN_0_PRECISION_0),
        .IN_FRAC_WIDTH (DATA_IN_0_PRECISION_1),
        .POW2_WIDTH    (DATA_OUT_0_PRECISION_0),
        .OUT_WIDTH     (DATA_OUT_0_PRECISION_0),
        .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1)
    ) fixed_softermax_1d_i (
        .clk,
        .rst,

        .in_data (in_data_unflattened[i]),
        .in_valid(in_data_valid[i]),
        .in_ready(in_data_ready[i]),

        .out_data (out_data_unflattened[i]),
        .out_valid(out_data_valid[i]),
        .out_ready(out_data_ready[i])
    );

    assign data_out_0 [(i + 1) * DATA_IN_0_PARALLELISM_DIM_0 - 1 : i * DATA_IN_0_PARALLELISM_DIM_0] = out_data_unflattened[i];
  end

  // * Join handshake signals from all the rows
  join_n #(
      .NUM_HANDSHAKES(DATA_OUT_0_PARALLELISM_DIM_1)
  ) join_n_i (
      .data_in_valid (out_data_valid),
      .data_in_ready (out_data_ready),
      .data_out_valid(data_out_0_valid),
      .data_out_ready(data_out_0_ready)
  );

endmodule
