/*
Module      : matmul
Description : This module does a matrix multiplcation between matrices X & Y.

              Try to use port B as the weights if you can optimise the bitwidth
              more aggressively than the input data. This is because as the B
              matrix is buffered, while the data in port A is streamed.

              Please refer to matrix multiplication documentation in
              "docs/hardware" for the parameter naming conventions and
              algorithm/order of matrix multiplications.

              Inputs are streamed row-wise (along dim0 first) in 2D sub-blocks
              which have width and height equal to the compute dimensions.

              Python equivalent:
              z = np.matmul(A, B)
*/

`timescale 1ns / 1ps

/* verilator lint_off UNUSEDPARAM */
module matmul #(
    // Total dimensions
    parameter A_TOTAL_DIM0 = 4,
    parameter A_TOTAL_DIM1 = 4,
    parameter B_TOTAL_DIM0 = 4,
    parameter B_TOTAL_DIM1 = 4,  // must equal A_TOTAL_DIM0

    // Compute dimensions
    parameter A_COMPUTE_DIM0 = 2,
    parameter A_COMPUTE_DIM1 = 2,
    parameter B_COMPUTE_DIM0 = 2,
    parameter B_COMPUTE_DIM1 = 2,  // must equal A_COMPUTE_DIM0

    // Input fixed point widths
    parameter A_WIDTH      = 8,
    parameter A_FRAC_WIDTH = 1,
    parameter B_WIDTH      = 8,
    parameter B_FRAC_WIDTH = 1,

    // Output fixed point widths
    parameter OUT_WIDTH      = 16,
    parameter OUT_FRAC_WIDTH = 2,

    // Output casting/rounding
    parameter OUT_SYMMETRIC = 0,

    // Derived Dimensions (Constants)
    localparam C_TOTAL_DIM0   = B_TOTAL_DIM0,
    localparam C_TOTAL_DIM1   = A_TOTAL_DIM1,
    localparam C_COMPUTE_DIM0 = B_COMPUTE_DIM0,
    localparam C_COMPUTE_DIM1 = A_COMPUTE_DIM1,

    // Derived Depth (Constants)
    localparam A_DEPTH_DIM0 = A_TOTAL_DIM0 / A_COMPUTE_DIM0,
    localparam A_DEPTH_DIM1 = A_TOTAL_DIM1 / A_COMPUTE_DIM1,
    localparam B_DEPTH_DIM0 = B_TOTAL_DIM0 / B_COMPUTE_DIM0,
    localparam B_DEPTH_DIM1 = B_TOTAL_DIM1 / B_COMPUTE_DIM1,
    localparam C_DEPTH_DIM0 = C_TOTAL_DIM0 / C_COMPUTE_DIM0,
    localparam C_DEPTH_DIM1 = C_TOTAL_DIM1 / C_COMPUTE_DIM1
) (
    input logic clk,
    input logic rst,

    // Matix A - row-major order
    input  logic [A_WIDTH-1:0] a_data [A_COMPUTE_DIM0*A_COMPUTE_DIM1-1:0],
    input  logic               a_valid,
    output logic               a_ready,

    // Matix B - row-major order
    input  logic [B_WIDTH-1:0] b_data [B_COMPUTE_DIM0*B_COMPUTE_DIM1-1:0],
    input  logic               b_valid,
    output logic               b_ready,

    // Matrix C - row-major order
    output logic [OUT_WIDTH-1:0] out_data [C_COMPUTE_DIM0*C_COMPUTE_DIM1-1:0],
    output logic                 out_valid,
    input  logic                 out_ready
);

  initial begin
    // Check dimension constraint not violated
    assert (A_TOTAL_DIM0 == B_TOTAL_DIM1)
    else $fatal("A_TOTAL_DIM0 must equal B_TOTAL_DIM1!");
    assert (A_COMPUTE_DIM0 == B_COMPUTE_DIM1)
    else $fatal("A_COMPUTE_DIM0 must equal B_COMPUTE_DIM1!");

    // Check compute vs. total divisibility
    assert (A_TOTAL_DIM0 % A_COMPUTE_DIM0 == 0)
    else $fatal("A_DIM0 compute is not divisible!");
    assert (A_TOTAL_DIM1 % A_COMPUTE_DIM1 == 0)
    else $fatal("A_DIM1 compute is not divisible!");
    assert (B_TOTAL_DIM0 % B_COMPUTE_DIM0 == 0)
    else $fatal("B_DIM0 compute is not divisible!");
    assert (B_TOTAL_DIM1 % B_COMPUTE_DIM1 == 0)
    else $fatal("B_DIM1 compute is not divisible!");
  end

  // -----
  // Params
  // -----

  localparam A_FLAT_WIDTH = A_WIDTH * A_COMPUTE_DIM0 * A_COMPUTE_DIM1;
  localparam B_FLAT_WIDTH = B_WIDTH * B_COMPUTE_DIM0 * B_COMPUTE_DIM1;

  localparam SM_OUT_WIDTH = A_WIDTH + B_WIDTH + $clog2(A_COMPUTE_DIM0);
  localparam SM_OUT_FRAC_WIDTH = A_FRAC_WIDTH + B_FRAC_WIDTH;

  localparam MAT_ACC_PTR_WIDTH = C_DEPTH_DIM0 == 1 ? 1 : $clog2(C_DEPTH_DIM0);
  localparam MAT_ACC_OUT_WIDTH = $clog2(B_DEPTH_DIM1) + SM_OUT_WIDTH;

  // -----
  // Wires
  // -----

  // Buffer unflatten out
  logic a_buffer_out_valid, a_buffer_out_ready;
  logic [A_WIDTH-1:0] a_buffer_out_data[A_COMPUTE_DIM0*A_COMPUTE_DIM1-1:0];

  // Repeat each submatrix in Matrix A stream B_DEPTH_DIM0 times
  // Only if (B_DEPTH_DIM0 > 1)
  logic [A_FLAT_WIDTH-1:0] a_data_flat;
  logic [A_FLAT_WIDTH-1:0] a_buffer_out_data_flat;

  // We need to buffer the B matrix
  // TODO: unless A_DEPTH_DIM1 == 1

  logic [B_FLAT_WIDTH-1:0] b_data_flat;

  // Buffer outputs
  logic [B_FLAT_WIDTH-1:0] b_buffer_out_data_flat;
  logic b_buffer_out_valid, b_buffer_out_ready;

  // Matrix unflatten output
  logic [B_WIDTH-1:0] b_buffer_out_data[B_COMPUTE_DIM0*B_COMPUTE_DIM1-1:0];

  logic [SM_OUT_WIDTH-1:0] sm_out_data[C_COMPUTE_DIM0*C_COMPUTE_DIM1];
  logic sm_out_valid, sm_out_ready;

  logic [C_DEPTH_DIM0-1:0] acc_in_valid;
  logic [C_DEPTH_DIM0-1:0] acc_in_ready;
  logic [C_DEPTH_DIM0-1:0] acc_out_valid;
  logic [C_DEPTH_DIM0-1:0] acc_out_ready;
  logic [MAT_ACC_OUT_WIDTH-1:0] acc_out_data[C_DEPTH_DIM0-1:0][C_COMPUTE_DIM0*C_COMPUTE_DIM1-1:0];

  logic [MAT_ACC_OUT_WIDTH-1:0] cast_in_data[C_COMPUTE_DIM0*C_COMPUTE_DIM1-1:0];


  // -----
  // State
  // -----

  struct {
    // Points to which matrix accumulator should store the simple_matmul output
    logic [MAT_ACC_PTR_WIDTH-1:0] matrix_acc_ptr;
    // Points at which output accumulator should be connected to the out stream
    logic [MAT_ACC_PTR_WIDTH-1:0] output_acc_ptr;
  }
      self, next_self;


  // -----
  // Logic
  // -----

  generate

    // B matrix Buffers

    if (B_DEPTH_DIM0 > 1) begin

      matrix_flatten #(
          .DATA_WIDTH(A_WIDTH),
          .DIM0      (A_COMPUTE_DIM0),
          .DIM1      (A_COMPUTE_DIM1)
      ) weight_buffer_flatten_a (
          .data_in (a_data),
          .data_out(a_data_flat)
      );

      single_element_repeat #(
          .DATA_WIDTH(A_FLAT_WIDTH),
          // Repeat for number of rows in matrix A
          .REPEAT    (B_DEPTH_DIM0)
      ) input_stream_buffer (
          .clk      (clk),
          .rst      (rst),
          .in_data  (a_data_flat),
          .in_valid (a_valid),
          .in_ready (a_ready),
          .out_data (a_buffer_out_data_flat),
          .out_valid(a_buffer_out_valid),
          .out_ready(a_buffer_out_ready)
      );

      matrix_unflatten #(
          .DATA_WIDTH(A_WIDTH),
          .DIM0      (A_COMPUTE_DIM0),
          .DIM1      (A_COMPUTE_DIM1)
      ) weight_buffer_unflatten_a (
          .data_in (a_buffer_out_data_flat),
          .data_out(a_buffer_out_data)
      );

    end else begin
      assign a_buffer_out_data = a_data;
      assign a_buffer_out_valid = a_valid;
      assign a_ready = a_buffer_out_ready;
    end

    // A matrix Buffers

    if (A_DEPTH_DIM1 > 1) begin
      matrix_flatten #(
          .DATA_WIDTH(B_WIDTH),
          .DIM0      (B_COMPUTE_DIM0),
          .DIM1      (B_COMPUTE_DIM1)
      ) weight_buffer_flatten_b (
          .data_in (b_data),
          .data_out(b_data_flat)
      );

      repeat_circular_buffer #(
          .DATA_WIDTH(B_FLAT_WIDTH),
          // Repeat for number of rows in matrix A
          .REPEAT    (A_DEPTH_DIM1),
          .SIZE      (B_DEPTH_DIM0 * B_DEPTH_DIM1)
      ) weight_buffer (
          .clk      (clk),
          .rst      (rst),
          .in_data  (b_data_flat),
          .in_valid (b_valid),
          .in_ready (b_ready),
          .out_data (b_buffer_out_data_flat),
          .out_valid(b_buffer_out_valid),
          .out_ready(b_buffer_out_ready)
      );

      matrix_unflatten #(
          .DATA_WIDTH(B_WIDTH),
          .DIM0      (B_COMPUTE_DIM0),
          .DIM1      (B_COMPUTE_DIM1)
      ) weight_buffer_unflatten_b (
          .data_in (b_buffer_out_data_flat),
          .data_out(b_buffer_out_data)
      );
    end else begin
      assign b_buffer_out_data = b_data;
      assign b_buffer_out_valid = b_valid;
      assign b_ready = b_buffer_out_ready;
    end
  endgenerate

  // Feed input A & buffered input B into simple matrix mult

  // Simple matrix multiply block's accumulator width
  // We do not round at simple_matmul level as we want to keep high precision
  // and round ourselves after the output accumulation in this matmul module.

  simple_matmul #(
      .N              (A_COMPUTE_DIM1),
      .M              (A_COMPUTE_DIM0),    // == B_COMPUTE_DIM1
      .K              (B_COMPUTE_DIM0),
      .X_WIDTH        (A_WIDTH),
      .X_FRAC_WIDTH   (A_FRAC_WIDTH),
      .Y_WIDTH        (B_WIDTH),
      .Y_FRAC_WIDTH   (B_FRAC_WIDTH),
      .OUTPUT_ROUNDING(0),
      .OUT_WIDTH      (SM_OUT_WIDTH),
      .OUT_FRAC_WIDTH (SM_OUT_FRAC_WIDTH)
  ) simple_matmul_inst (
      .clk      (clk),
      .rst      (rst),
      .x_data   (a_buffer_out_data),
      .x_valid  (a_buffer_out_valid),
      .x_ready  (a_buffer_out_ready),
      .y_data   (b_buffer_out_data),
      .y_valid  (b_buffer_out_valid),
      .y_ready  (b_buffer_out_ready),
      .out_data (sm_out_data),
      .out_valid(sm_out_valid),
      .out_ready(sm_out_ready)
  );

  // Direct the result of the simple matmul to the correct matrix_accumulator

  for (genvar i = 0; i < C_DEPTH_DIM0; i++) begin : accumulators
    matrix_accumulator #(
        .IN_DEPTH(B_DEPTH_DIM1),
        .IN_WIDTH(SM_OUT_WIDTH),
        .DIM0    (C_COMPUTE_DIM0),
        .DIM1    (C_COMPUTE_DIM1)
    ) matrix_acc_inst (
        .clk      (clk),
        .rst      (rst),
        .in_data  (sm_out_data),
        .in_valid (acc_in_valid[i]),
        .in_ready (acc_in_ready[i]),
        .out_data (acc_out_data[i]),
        .out_valid(acc_out_valid[i]),
        .out_ready(acc_out_ready[i])
    );
  end

  for (genvar i = 0; i < C_DEPTH_DIM0; i++) begin
    // Change which accumulator the output of simple_matmul goes to
    assign acc_in_valid[i]  = self.matrix_acc_ptr == i ? sm_out_valid : 0;

    // Select which accumulator can output on out stream
    assign acc_out_ready[i] = self.output_acc_ptr == i ? out_ready : 0;
  end

  assign sm_out_ready = acc_in_ready[self.matrix_acc_ptr];

  for (genvar i = 0; i < C_COMPUTE_DIM0 * C_COMPUTE_DIM1; i++) begin
    fixed_signed_cast #(
        .IN_WIDTH      (MAT_ACC_OUT_WIDTH),
        .IN_FRAC_WIDTH (SM_OUT_FRAC_WIDTH),
        .OUT_WIDTH     (OUT_WIDTH),
        .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),
        .SYMMETRIC     (OUT_SYMMETRIC),
        .ROUND_FLOOR   (1)
    ) output_cast (
        .in_data (cast_in_data[i]),
        .out_data(out_data[i])
    );
  end

  // Logic to handle accumulator selection & output selection.
  always_comb begin
    next_self = self;

    for (int i = 0; i < C_COMPUTE_DIM0 * C_COMPUTE_DIM1; i++) begin
      cast_in_data[i] = acc_out_data[self.output_acc_ptr][i];
    end
    out_valid = acc_out_valid[self.output_acc_ptr];

    // Change accumulator pointer
    if (sm_out_valid && sm_out_ready) begin
      if (self.matrix_acc_ptr == C_DEPTH_DIM0 - 1) begin
        next_self.matrix_acc_ptr = 0;
      end else begin
        next_self.matrix_acc_ptr += 1;
      end
    end

    // Change output pointer
    if (|acc_out_ready && |acc_out_valid) begin
      if (self.output_acc_ptr == C_DEPTH_DIM0 - 1) begin
        next_self.output_acc_ptr = 0;
      end else begin
        next_self.output_acc_ptr += 1;
      end
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      self <= '{default: 0};
    end else begin
      self <= next_self;
    end
  end

endmodule
