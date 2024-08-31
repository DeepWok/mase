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

module fixed_linear_programmable #(
    /* verilator lint_off UNUSEDPARAM */
    parameter HAS_BIAS = 1,
    parameter WEIGHTS_PRE_TRANSPOSED = 0,

    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter DATA_IN_0_MAX_TENSOR_SIZE_DIM_0 = 20,
    parameter DATA_IN_0_MAX_TENSOR_SIZE_DIM_1 = 20,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 4,  // must equal WEIGHT_PARALLELISM_DIM_1
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_2 = 1,
    localparam DATA_IN_0_MAX_DEPTH_DIM_0 = DATA_IN_0_MAX_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0,
    localparam DATA_IN_0_MAX_DEPTH_DIM_1 = DATA_IN_0_MAX_TENSOR_SIZE_DIM_1 / DATA_IN_0_PARALLELISM_DIM_1,

    parameter WEIGHT_PRECISION_0 = 16,
    parameter WEIGHT_PRECISION_1 = 3,
    parameter WEIGHT_MAX_TENSOR_SIZE_DIM_0 = 20,
    parameter WEIGHT_MAX_TENSOR_SIZE_DIM_1 = 20,
    parameter WEIGHT_PARALLELISM_DIM_0 = 4,
    parameter WEIGHT_PARALLELISM_DIM_1 = 4,
    localparam WEIGHT_MAX_DEPTH_DIM0 = WEIGHT_MAX_TENSOR_SIZE_DIM_0 / WEIGHT_PARALLELISM_DIM_0,
    localparam WEIGHT_MAX_DEPTH_DIM1 = WEIGHT_MAX_TENSOR_SIZE_DIM_1 / WEIGHT_PARALLELISM_DIM_1,

    // Inferred precision of the output data
    parameter DATA_OUT_0_PRECISION_0 = 16,
    parameter DATA_OUT_0_PRECISION_1 = 3,
    parameter DATA_OUT_0_MAX_TENSOR_SIZE_DIM_0 = WEIGHT_MAX_TENSOR_SIZE_DIM_0,
    parameter DATA_OUT_0_MAX_TENSOR_SIZE_DIM_1 = DATA_IN_0_MAX_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_MAX_TENSOR_SIZE_DIM_2 = DATA_IN_0_MAX_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = WEIGHT_PARALLELISM_DIM_0,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_2 = DATA_IN_0_PARALLELISM_DIM_1,

    parameter BIAS_PRECISION_0 = 16,
    parameter BIAS_PRECISION_1 = 3,
    parameter BIAS_MAX_TENSOR_SIZE_DIM_0 = DATA_OUT_0_MAX_TENSOR_SIZE_DIM_0,
    parameter BIAS_MAX_TENSOR_SIZE_DIM_1 = 1,
    parameter BIAS_PARALLELISM_DIM_0 = 4,
    parameter BIAS_PARALLELISM_DIM_1 = 1,
    localparam BIAS_MAX_DEPTH_DIM_0 = BIAS_MAX_TENSOR_SIZE_DIM_0 / BIAS_PARALLELISM_DIM_0,
    localparam BIAS_MAX_DEPTH_DIM_1 = BIAS_MAX_TENSOR_SIZE_DIM_1 / BIAS_PARALLELISM_DIM_1,
    localparam DATA_IN_0_MAX_DEPTH_DIM1_WIDTH = $clog2(DATA_IN_0_MAX_DEPTH_DIM_1),
    localparam WEIGHT_MAX_DEPTH_DIM0_WIDTH = $clog2(WEIGHT_MAX_DEPTH_DIM0),
    localparam WEIGHT_MAX_DEPTH_DIM1_WIDTH = $clog2(WEIGHT_MAX_DEPTH_DIM1),
    localparam WEIGHT_MAX_DEPTH_MULT_WIDTH = $clog2(WEIGHT_MAX_DEPTH_DIM1) + $clog2(WEIGHT_MAX_DEPTH_DIM0),
    localparam WEIGHT_MAX_TENSOR_SIZE_DIM_0_WIDTH = $clog2(WEIGHT_MAX_TENSOR_SIZE_DIM_0)

) (
    input clk,
    input rst,

    input logic [DATA_IN_0_MAX_DEPTH_DIM1_WIDTH:0]     data_in_0_depth_dim1,
    input logic [WEIGHT_MAX_TENSOR_SIZE_DIM_0_WIDTH:0] weight_tensor_size_dim0,
    input logic [WEIGHT_MAX_DEPTH_DIM0_WIDTH:0]        weight_depth_dim0,
    input logic [WEIGHT_MAX_DEPTH_DIM1_WIDTH:0]        weight_depth_dim1,
    input logic [WEIGHT_MAX_DEPTH_MULT_WIDTH:0]        weight_depth_mult,

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


  // * Declarations
  // * ---------------------------------------------------------------------------------------------------

  logic [DATA_OUT_0_PRECISION_0-1:0] matmul_out [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0];
  logic matmul_out_valid;
  logic matmul_out_ready;

  logic [DATA_OUT_0_PRECISION_0-1:0] bias_buffered [BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1 -1:0];
  logic bias_buffered_valid, bias_buffered_ready;

  logic [DATA_OUT_0_PRECISION_0-1:0] bias_casted [BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1 -1:0];
  logic [DATA_OUT_0_PRECISION_0-1:0] add_bias_in [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0];
  logic add_bias_in_valid;
  logic add_bias_in_ready;

  // * Instances
  // * ---------------------------------------------------------------------------------------------------


  matmul_programmable #(
      // Total dimensions
      .A_MAX_DIM0(DATA_IN_0_MAX_TENSOR_SIZE_DIM_0),
      .A_MAX_DIM1(DATA_IN_0_MAX_TENSOR_SIZE_DIM_1),
      .B_MAX_DIM0(WEIGHT_MAX_TENSOR_SIZE_DIM_0),
      .B_MAX_DIM1(WEIGHT_MAX_TENSOR_SIZE_DIM_1),

      .A_COMPUTE_DIM0(DATA_IN_0_PARALLELISM_DIM_0),
      .A_COMPUTE_DIM1(DATA_IN_0_PARALLELISM_DIM_1),
      .B_COMPUTE_DIM0(WEIGHT_PARALLELISM_DIM_0),
      .B_COMPUTE_DIM1(WEIGHT_PARALLELISM_DIM_1),

      .A_WIDTH     (DATA_IN_0_PRECISION_0),
      .A_FRAC_WIDTH(DATA_IN_0_PRECISION_1),
      .B_WIDTH     (WEIGHT_PRECISION_0),
      .B_FRAC_WIDTH(WEIGHT_PRECISION_1),

      .OUT_WIDTH     (DATA_OUT_0_PRECISION_0),
      .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1),
      .OUT_SYMMETRIC (0)
  ) matmul_i (
      .clk,
      .rst,

      .a_depth_dim1(data_in_0_depth_dim1),
      .b_depth_dim0(weight_depth_dim0),
      .b_depth_dim1(weight_depth_dim1),
      .b_depth_mult(weight_depth_mult),

      .a_data (data_in_0),
      .a_valid(data_in_0_valid),
      .a_ready(data_in_0_ready),

      .b_data (weight),
      .b_valid(weight_valid),
      .b_ready(weight_ready),

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

    unpacked_repeat_circular_buffer_programmable #(
        .DATA_WIDTH (BIAS_PRECISION_0),
        .IN_NUM     (BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1),
        .MAX_REPEAT     (DATA_IN_0_MAX_DEPTH_DIM_1),
        .MAX_SIZE       (WEIGHT_MAX_TENSOR_SIZE_DIM_0)
    ) bias_buffer_inst (
        .clk,
        .rst,

        .repeat_n(data_in_0_depth_dim1),
        .size_n(weight_tensor_size_dim0),
        
        // Input streaming port
        .in_data    (bias),
        .in_valid   (bias_valid),
        .in_ready   (bias_ready),
    
        // Output streaming port
        .out_data   (bias_buffered),
        .out_valid  (bias_buffered_valid),
        .out_ready  (bias_buffered_ready)
    );

    unpacked_register_slice #(
        .DATA_WIDTH(DATA_OUT_0_PRECISION_0),
        .IN_SIZE   (DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1)
    ) register_slice_i (
        .clk(clk),
        .rst(rst),

        .data_in (add_bias_in),
        .data_in_valid(add_bias_in_valid),
        .data_in_ready(add_bias_in_ready),

        .data_out (data_out_0),
        .data_out_valid(data_out_0_valid),
        .data_out_ready(data_out_0_ready)
    );
  end



  // * Add bias
  if (HAS_BIAS == 1) begin
    fixed_cast #(
        .IN_SIZE       (BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1),
        .IN_WIDTH      (BIAS_PRECISION_0),
        .IN_FRAC_WIDTH (BIAS_PRECISION_1),
        .OUT_WIDTH     (DATA_OUT_0_PRECISION_0),
        .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1)
    ) bias_cast_i (
        .data_in (bias_buffered),
        .data_out(bias_casted)
    );

    for (genvar i_0 = 0; i_0 < DATA_OUT_0_PARALLELISM_DIM_0 ; i_0++) begin
      for (genvar i_1 = 0; i_1 < DATA_OUT_0_PARALLELISM_DIM_1 ; i_1++) begin
        assign add_bias_in [i_1 * DATA_OUT_0_PARALLELISM_DIM_0 + i_0] = $signed(matmul_out[i_1 * DATA_OUT_0_PARALLELISM_DIM_0 + i_0])  + $signed(bias_casted[i_0]);
      end
    end

  end else begin
    assign data_out_0 = matmul_out;
    assign data_out_0_valid = matmul_out_valid;
    assign matmul_out_ready = data_out_0_ready;
  end

endmodule
