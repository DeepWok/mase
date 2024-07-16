`timescale 1ns / 1ps

/*
 * Simple registered adder between two inputs.
 * Currently doesn't support parallelism conversion.
 */

module fixed_adder #(
    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 4,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_2 = 1,

    parameter DATA_IN_1_PRECISION_0 = 16,
    parameter DATA_IN_1_PRECISION_1 = 3,
    parameter DATA_IN_1_TENSOR_SIZE_DIM_0 = 4,
    parameter DATA_IN_1_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_1_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_IN_1_PARALLELISM_DIM_0 = 4,
    parameter DATA_IN_1_PARALLELISM_DIM_1 = 1,
    parameter DATA_IN_1_PARALLELISM_DIM_2 = 1,

    parameter DATA_OUT_0_PRECISION_0 = 16,
    parameter DATA_OUT_0_PRECISION_1 = 3,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_2 = DATA_IN_0_TENSOR_SIZE_DIM_2,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_2 = DATA_IN_0_PARALLELISM_DIM_2
) (
    input clk,
    input rst,

    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0 [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    input logic [DATA_IN_1_PRECISION_0-1:0] data_in_1 [DATA_IN_1_PARALLELISM_DIM_0*DATA_IN_1_PARALLELISM_DIM_1-1:0],
    input logic data_in_1_valid,
    output logic data_in_1_ready,

    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_0_valid,
    input logic data_out_0_ready
);

  localparam MAX_PRECISION_0 = DATA_IN_0_PRECISION_0 > DATA_IN_1_PRECISION_0 ? DATA_IN_0_PRECISION_0 : DATA_IN_1_PRECISION_0;

  localparam SUM_PRECISION_0 = MAX_PRECISION_0 + 1;

  // ! TO DO: check if this is correct
  localparam SUM_PRECISION_1 = DATA_IN_0_PRECISION_1;

  // * Declarations
  // * ---------------------------------------------------------------------------------------------------

  logic joined_input_valid;
  logic joined_input_ready;
  logic [SUM_PRECISION_0-1:0] add_result [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [DATA_OUT_0_PRECISION_0-1:0] cast_out [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0];

  // * Instances
  // * ---------------------------------------------------------------------------------------------------

  // * Wait until both inputs are available
  join2 join_inst (
      .data_in_valid ({data_in_0_valid, data_in_1_valid}),
      .data_in_ready ({data_in_0_ready, data_in_1_ready}),
      .data_out_valid(joined_input_valid),
      .data_out_ready(joined_input_ready)
  );

  // * Cast the sum to the requested output precision
  fixed_cast #(
      .IN_SIZE       (DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1),
      .IN_WIDTH      (SUM_PRECISION_0),
      .IN_FRAC_WIDTH (SUM_PRECISION_1),
      .OUT_WIDTH     (DATA_OUT_0_PRECISION_0),
      .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1)
  ) bias_cast_i (
      .data_in (add_result),
      .data_out(cast_out)
  );

  // * Register the output
  unpacked_register_slice #(
      .DATA_WIDTH(DATA_OUT_0_PRECISION_0),
      .IN_SIZE   (DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1)
  ) register_slice_i (
      .clk(clk),
      .rst(rst),

      .data_in(cast_out),
      .data_in_valid(joined_input_valid),
      .data_in_ready(joined_input_ready),

      .data_out(data_out_0),
      .data_out_valid(data_out_0_valid),
      .data_out_ready(data_out_0_ready)
  );

  // * Logic
  // * ---------------------------------------------------------------------------------------------------

  // * Do the sum
  for (genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1; i++) begin
    assign add_result[i] = data_in_0[i] + data_in_1[i];
  end

endmodule
