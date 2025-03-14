`timescale 1ns / 1ps

/*
 *
 * The fixed_linear module implements torch.nn.functional.linear, which
 * computes Y = X @ W^T + b
 *
 * Weight tensor is assumed to have shape (out_features, in_features)
 * Data tensor is assumed to have shape (batch_size, in_features)
 * Bias tensor is assumed to have shape (out_features)
 *
 * If WEIGHTS_PRE_TRANSPOSED is set to 0, the module will transpose the incoming
 * weight matrix. Otherwise, it will assume that the incoming weight matrix is
 * already transposed.
 *
 */

module fixed_linear #(
    /* verilator lint_off UNUSEDPARAM */
    parameter HAS_BIAS = 1,
    parameter WEIGHTS_PRE_TRANSPOSED = 0,

    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 20,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 20,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 4,  // must equal WEIGHT_PARALLELISM_DIM_1
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 4,
    localparam IN_0_DEPTH_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0,
    localparam IN_0_DEPTH_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1 / DATA_IN_0_PARALLELISM_DIM_1,

    parameter WEIGHT_PRECISION_0 = 16,
    parameter WEIGHT_PRECISION_1 = 3,
    parameter WEIGHT_TENSOR_SIZE_DIM_0 = 20,
    parameter WEIGHT_TENSOR_SIZE_DIM_1 = 20,
    parameter WEIGHT_PARALLELISM_DIM_0 = 4,
    parameter WEIGHT_PARALLELISM_DIM_1 = 4,

    // Inferred precision of the output data
    // if the data out precision will be replaced by the setting
    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
        DATA_IN_0_TENSOR_SIZE_DIM_0
    ) + 1,
    parameter DATA_OUT_0_PRECISION_1 = DATA_IN_0_PRECISION_1 + WEIGHT_PRECISION_1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = (WEIGHTS_PRE_TRANSPOSED == 0)? WEIGHT_TENSOR_SIZE_DIM_1:WEIGHT_TENSOR_SIZE_DIM_0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = (WEIGHTS_PRE_TRANSPOSED == 0)? WEIGHT_PARALLELISM_DIM_1:WEIGHT_PARALLELISM_DIM_0,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,

    parameter BIAS_PRECISION_0 = 16,
    parameter BIAS_PRECISION_1 = 3,
    parameter BIAS_TENSOR_SIZE_DIM_0 = DATA_OUT_0_TENSOR_SIZE_DIM_0,
    parameter BIAS_TENSOR_SIZE_DIM_1 = 1,
    parameter BIAS_PARALLELISM_DIM_0 = DATA_OUT_0_PARALLELISM_DIM_0,
    parameter BIAS_PARALLELISM_DIM_1 = 1,

    localparam BIAS_DEPTH_DIM_0 = BIAS_TENSOR_SIZE_DIM_0 / BIAS_PARALLELISM_DIM_0,
    localparam BIAS_DEPTH_DIM_1 = BIAS_TENSOR_SIZE_DIM_1 / BIAS_PARALLELISM_DIM_1

) (
    input clk,
    input rst,

    // input port for data_inivations
    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0 [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    // input port for weight
    input logic [WEIGHT_PRECISION_0-1:0] weight [WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1-1:0],
    input logic weight_valid,
    output logic weight_ready,

    /* verilator lint_off UNUSEDSIGNAL */
    input logic [BIAS_PRECISION_0-1:0] bias[BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1 -1:0],
    input logic bias_valid,
    /* verilator lint_on UNUSEDSIGNAL */
    output logic bias_ready,

    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_0_valid,
    input logic data_out_0_ready
);
  localparam MMOUT_PRECISION_0 = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
      DATA_IN_0_TENSOR_SIZE_DIM_0
  );
  localparam MMOUT_PRECISION_1 = DATA_IN_0_PRECISION_1 + WEIGHT_PRECISION_1;

  // The TENSOR_SIZE and PARALLELISM parameters for the weights are set by emit verilog according to the real
  // tensor values. Here we account for the change when the weights are pre-transposed
  localparam REAL_WEIGHT_TENSOR_SIZE_DIM_0 = (WEIGHTS_PRE_TRANSPOSED == 0) ? WEIGHT_TENSOR_SIZE_DIM_1 : WEIGHT_TENSOR_SIZE_DIM_0;
  localparam REAL_WEIGHT_TENSOR_SIZE_DIM_1 = (WEIGHTS_PRE_TRANSPOSED == 0) ? WEIGHT_TENSOR_SIZE_DIM_0 : WEIGHT_TENSOR_SIZE_DIM_1;
  localparam REAL_WEIGHT_PARALLELISM_DIM_0 = (WEIGHTS_PRE_TRANSPOSED == 0) ? WEIGHT_PARALLELISM_DIM_1 : WEIGHT_PARALLELISM_DIM_0;
  localparam REAL_WEIGHT_PARALLELISM_DIM_1 = (WEIGHTS_PRE_TRANSPOSED == 0) ? WEIGHT_PARALLELISM_DIM_0 : WEIGHT_PARALLELISM_DIM_1;

  // * Declarations
  // * ---------------------------------------------------------------------------------------------------

  logic [WEIGHT_PRECISION_0-1:0] weight_transposed [WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1-1:0];
  logic weight_transposed_valid;
  logic weight_transposed_ready;

  logic [MMOUT_PRECISION_0-1:0] matmul_out [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0];
  logic matmul_out_valid;
  logic matmul_out_ready;

  logic [BIAS_PRECISION_0-1:0] bias_buffered[BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1 -1:0];
  logic bias_buffered_valid, bias_buffered_ready;

  logic [MMOUT_PRECISION_0-1:0] bias_casted[BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1 -1:0];
  logic [MMOUT_PRECISION_0:0] add_bias_in [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0];
  logic [DATA_OUT_0_PRECISION_0 - 1:0] add_bias_in_casted [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0];
  logic add_bias_in_valid;
  logic add_bias_in_ready;

  // * Instances
  // * ---------------------------------------------------------------------------------------------------

  if (WEIGHTS_PRE_TRANSPOSED == 0) begin
    matrix_stream_transpose #(
        .TOTAL_DIM0  (WEIGHT_TENSOR_SIZE_DIM_0),
        .TOTAL_DIM1  (WEIGHT_TENSOR_SIZE_DIM_1),
        .COMPUTE_DIM0(WEIGHT_PARALLELISM_DIM_0),
        .COMPUTE_DIM1(WEIGHT_PARALLELISM_DIM_1),
        .DATA_WIDTH  (WEIGHT_PRECISION_0)
    ) weight_matrix_transpose_i (
        .clk,
        .rst,

        .in_data (weight),
        .in_valid(weight_valid),
        .in_ready(weight_ready),

        .out_data (weight_transposed),
        .out_valid(weight_transposed_valid),
        .out_ready(weight_transposed_ready)
    );
  end
  matmul #(
      // Total dimensions
      .A_TOTAL_DIM0(DATA_IN_0_TENSOR_SIZE_DIM_0),
      .A_TOTAL_DIM1(DATA_IN_0_TENSOR_SIZE_DIM_1),
      .B_TOTAL_DIM0(REAL_WEIGHT_TENSOR_SIZE_DIM_0),
      .B_TOTAL_DIM1(REAL_WEIGHT_TENSOR_SIZE_DIM_1),

      .A_COMPUTE_DIM0(DATA_IN_0_PARALLELISM_DIM_0),
      .A_COMPUTE_DIM1(DATA_IN_0_PARALLELISM_DIM_1),
      .B_COMPUTE_DIM0(REAL_WEIGHT_PARALLELISM_DIM_0),
      .B_COMPUTE_DIM1(REAL_WEIGHT_PARALLELISM_DIM_1),

      .A_WIDTH     (DATA_IN_0_PRECISION_0),
      .A_FRAC_WIDTH(DATA_IN_0_PRECISION_1),
      .B_WIDTH     (WEIGHT_PRECISION_0),
      .B_FRAC_WIDTH(WEIGHT_PRECISION_1),

      .OUT_WIDTH     (MMOUT_PRECISION_0),
      .OUT_FRAC_WIDTH(MMOUT_PRECISION_1),
      .OUT_SYMMETRIC (0)
  ) matmul_i (
      .clk,
      .rst,

      .a_data (data_in_0),
      .a_valid(data_in_0_valid),
      .a_ready(data_in_0_ready),

      .b_data (weight_transposed),
      .b_valid(weight_transposed_valid),
      .b_ready(weight_transposed_ready),

      .out_data (matmul_out),
      .out_valid(matmul_out_valid),
      .out_ready(matmul_out_ready)
  );

  // Bias output
  if (HAS_BIAS == 1) begin

    join2 join2_matmul_bias_i (
        .data_in_valid ({matmul_out_valid, bias_buffered_valid}),
        .data_in_ready ({matmul_out_ready, bias_buffered_ready}),
        .data_out_valid(add_bias_in_valid),
        .data_out_ready(add_bias_in_ready)
    );

    input_buffer #(
        .DATA_WIDTH (BIAS_PRECISION_0),
        .IN_NUM     (BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1),
        .REPEAT     (IN_0_DEPTH_DIM_1),
        .BUFFER_SIZE(BIAS_DEPTH_DIM_0)
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
    fixed_rounding #(
        .IN_SIZE       (DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1),
        .IN_WIDTH      (MMOUT_PRECISION_0 + 1),
        .IN_FRAC_WIDTH (MMOUT_PRECISION_1),
        .OUT_WIDTH     (DATA_OUT_0_PRECISION_0),
        .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1)
    ) output_cast (
        .data_in (add_bias_in),
        .data_out(add_bias_in_casted)
    );
    unpacked_register_slice #(
        .DATA_WIDTH(DATA_OUT_0_PRECISION_0),
        .IN_SIZE   (DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1)
    ) register_slice_i (
        .clk(clk),
        .rst(rst),

        .data_in(add_bias_in_casted),
        .data_in_valid(add_bias_in_valid),
        .data_in_ready(add_bias_in_ready),

        .data_out(data_out_0),
        .data_out_valid(data_out_0_valid),
        .data_out_ready(data_out_0_ready)
    );
  end

  // * Logic
  // * ---------------------------------------------------------------------------------------------------

  if (WEIGHTS_PRE_TRANSPOSED == 1) begin
    always_comb begin
      weight_transposed_valid = weight_valid;
      weight_ready = weight_transposed_ready;
    end

    for (genvar i = 0; i < WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1; i++) begin
      assign weight_transposed[i] = weight[i];
    end
  end

  // * Add bias
  if (HAS_BIAS == 1) begin
    fixed_cast #(
        .IN_SIZE       (BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1),
        .IN_WIDTH      (BIAS_PRECISION_0),
        .IN_FRAC_WIDTH (BIAS_PRECISION_1),
        .OUT_WIDTH     (MMOUT_PRECISION_0),
        .OUT_FRAC_WIDTH(MMOUT_PRECISION_1)
    ) bias_cast_i (
        .data_in (bias_buffered),
        .data_out(bias_casted)
    );

    for (genvar i_0 = 0; i_0 < DATA_OUT_0_PARALLELISM_DIM_0; i_0++) begin
      for (genvar i_1 = 0; i_1 < DATA_OUT_0_PARALLELISM_DIM_1; i_1++) begin
        assign add_bias_in[i_1*DATA_OUT_0_PARALLELISM_DIM_0+i_0] = $signed(
            matmul_out[i_1*DATA_OUT_0_PARALLELISM_DIM_0+i_0]
        ) + $signed(
            bias_casted[i_0]
        );
      end
    end

  end else begin
    fixed_rounding #(
        .IN_SIZE       (DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1),
        .IN_WIDTH      (MMOUT_PRECISION_0),
        .IN_FRAC_WIDTH (MMOUT_PRECISION_1),
        .OUT_WIDTH     (DATA_OUT_0_PRECISION_0),
        .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1)
    ) output_cast (
        .data_in (matmul_out),
        .data_out(data_out_0)
    );
    assign data_out_0_valid = matmul_out_valid;
    assign matmul_out_ready = data_out_0_ready;
  end

endmodule
