/*
Module      : fixed_gqa_head

Description : Implements an attention head which is used in group query
              attention (GQA). It has no K, V matrix multiplications as this is
              done outside of the heads due to the shared K, V weight matrices.

              The module has parameterised intermediate fixed-point widths and
              Q has a linear layer where the embedding dimension of the input
              can reduced from EMBEDDING_DIM to HEAD_DIM for this head.

              !! The K, V matrix multiply and K transpose is done outside of
              this module. !!

              Dimensions of each input/output port is in the comments below.

Dataflow    : 1. Q get projected from EMBEDDING_DIM to HEAD_DIM
              2. QK^T matrix multiplication
              3. Softermax on attention scores: softermax(QK^T)
              4. Final matrix multiply to get attention: softermax(QK^T) * V

Assumptions : 1. All activation inputs share same total and compute dimensions.
              2. All weight inputs share same total and compute dimensions.
              3. Activations and weights share same compute dimensions.
              4. The K input is transposed already.
*/

`timescale 1ns / 1ps

module fixed_gqa_head #(
    // Dimensions
    parameter TOTAL_EMBEDDING_DIM   = 32,
    parameter TOTAL_HEAD_DIM        = 16,
    parameter TOTAL_SEQUENCE_DIM    = 16,
    parameter COMPUTE_EMBEDDING_DIM = 4,
    parameter COMPUTE_HEAD_DIM      = 4,
    parameter COMPUTE_SEQUENCE_DIM  = 4,

    // Input Port Widths
    parameter Q_ACT_WIDTH         = 8,
    parameter Q_ACT_FRAC_WIDTH    = 2,
    parameter Q_WEIGHT_WIDTH      = 8,
    parameter Q_WEIGHT_FRAC_WIDTH = 2,

    parameter K_ACT_WIDTH      = 8,
    parameter K_ACT_FRAC_WIDTH = 2,

    parameter V_ACT_WIDTH      = 8,
    parameter V_ACT_FRAC_WIDTH = 2,

    // Output Port Widths
    parameter OUT_ACT_WIDTH      = 8,
    parameter OUT_ACT_FRAC_WIDTH = 2,

    // Intermediate widths
    // Output widths for query activation & weight multiplication
    parameter Q_OUT_WIDTH              = 16,
    parameter Q_OUT_FRAC_WIDTH         = 8,
    // Output width for QK^T matrix multiplication
    parameter QK_OUT_WIDTH             = 16,
    parameter QK_OUT_FRAC_WIDTH        = 8,
    // Widths for Softermax module
    parameter SOFTERMAX_POW2_WIDTH     = 16,
    parameter SOFTERMAX_OUT_WIDTH      = 16,
    parameter SOFTERMAX_OUT_FRAC_WIDTH = 15
) (
    input logic clk,
    input logic rst,

    // Query Activation & Weight Matrices
    // The multiplication between q_act and q_weight will reduce the embedding dimension
    // from COMPUTE_EMBEDDING_DIM down to the COMPUTE_HEAD_DIM for this head.

    // Query Activation Input (dims = seq_dim x embedding_dim)
    input  logic [Q_ACT_WIDTH-1:0] q_act_data [COMPUTE_SEQUENCE_DIM*COMPUTE_EMBEDDING_DIM-1:0],
    input  logic                   q_act_valid,
    output logic                   q_act_ready,

    // Query Weights for this Head (dims = embedding_dim x head_dim)
    input  logic [Q_WEIGHT_WIDTH-1:0] q_weight_data [COMPUTE_EMBEDDING_DIM*COMPUTE_HEAD_DIM-1:0],
    input  logic                      q_weight_valid,
    output logic                      q_weight_ready,

    // Key & Value Matmul has been done outside of this module so they are already
    // in the specified head embedding dim.

    // Pre-Calculated & Transposed Key Activation Matrix (dims = head_dim x seq_dim)
    input logic [K_ACT_WIDTH-1:0] k_transposed_act_data[COMPUTE_HEAD_DIM*COMPUTE_SEQUENCE_DIM-1:0],
    input logic k_transposed_act_valid,
    output logic k_transposed_act_ready,

    // Pre-Calculated Value Activation Matrix (dims = seq_dim x head_dim)
    input  logic [V_ACT_WIDTH-1:0] v_act_data [COMPUTE_SEQUENCE_DIM*COMPUTE_HEAD_DIM-1:0],
    input  logic                   v_act_valid,
    output logic                   v_act_ready,

    // Output Activation Matrix (dims = seq_dim x head_dim)
    output logic [OUT_ACT_WIDTH-1:0] out_act_data [COMPUTE_SEQUENCE_DIM*COMPUTE_HEAD_DIM-1:0],
    output logic                     out_act_valid,
    input  logic                     out_act_ready
);

  // -----
  // Params
  // -----

  localparam EMBEDDING_DEPTH = TOTAL_EMBEDDING_DIM / COMPUTE_EMBEDDING_DIM;
  localparam HEAD_DEPTH = TOTAL_HEAD_DIM / COMPUTE_HEAD_DIM;
  localparam SEQUENCE_DEPTH = TOTAL_SEQUENCE_DIM / COMPUTE_SEQUENCE_DIM;

  initial begin
    // Check divisibility
    assert (EMBEDDING_DEPTH * COMPUTE_EMBEDDING_DIM == TOTAL_EMBEDDING_DIM);
    assert (HEAD_DEPTH * COMPUTE_HEAD_DIM == TOTAL_HEAD_DIM);
    assert (SEQUENCE_DEPTH * COMPUTE_SEQUENCE_DIM == TOTAL_SEQUENCE_DIM);
  end


  // -----
  // Wires
  // -----

  // Output of q_act x q_weight (dims = seq_dim x head_dim)
  logic [Q_OUT_WIDTH-1:0] q_out_data[COMPUTE_SEQUENCE_DIM*COMPUTE_HEAD_DIM-1:0];
  logic q_out_valid, q_out_ready;

  // Output of q_out x k_transposed_act (dims = seq_dim x seq_dim)
  logic [QK_OUT_WIDTH-1:0] qk_out_data[COMPUTE_SEQUENCE_DIM*COMPUTE_SEQUENCE_DIM-1:0];
  logic qk_out_valid, qk_out_ready;

  // Output of softermax(q_out x k_transposed_act) (dims = seq_dim x seq_dim)
  logic [SOFTERMAX_OUT_WIDTH-1:0] softermax_out_data[COMPUTE_SEQUENCE_DIM*COMPUTE_SEQUENCE_DIM-1:0];
  logic [SOFTERMAX_OUT_WIDTH:0] softermax_unsigned_out_data [COMPUTE_SEQUENCE_DIM*COMPUTE_SEQUENCE_DIM-1:0];
  logic softermax_out_valid, softermax_out_ready;

  // -----
  // Modules
  // -----

  matmul #(
      // Activations
      .A_TOTAL_DIM0  (TOTAL_EMBEDDING_DIM),
      .A_TOTAL_DIM1  (TOTAL_SEQUENCE_DIM),
      .A_COMPUTE_DIM0(COMPUTE_EMBEDDING_DIM),
      .A_COMPUTE_DIM1(COMPUTE_SEQUENCE_DIM),
      .A_WIDTH       (Q_ACT_WIDTH),
      .A_FRAC_WIDTH  (Q_ACT_FRAC_WIDTH),
      // Weights
      .B_TOTAL_DIM0  (TOTAL_HEAD_DIM),
      .B_TOTAL_DIM1  (TOTAL_EMBEDDING_DIM),
      .B_COMPUTE_DIM0(COMPUTE_HEAD_DIM),
      .B_COMPUTE_DIM1(COMPUTE_EMBEDDING_DIM),
      .B_WIDTH       (Q_WEIGHT_WIDTH),
      .B_FRAC_WIDTH  (Q_WEIGHT_FRAC_WIDTH),
      // Output
      .OUT_WIDTH     (Q_OUT_WIDTH),
      .OUT_FRAC_WIDTH(Q_OUT_FRAC_WIDTH),
      .OUT_SYMMETRIC (0)
  ) q_matmul (
      .clk      (clk),
      .rst      (rst),
      .a_data   (q_act_data),
      .a_valid  (q_act_valid),
      .a_ready  (q_act_ready),
      .b_data   (q_weight_data),
      .b_valid  (q_weight_valid),
      .b_ready  (q_weight_ready),
      .out_data (q_out_data),
      .out_valid(q_out_valid),
      .out_ready(q_out_ready)
  );

  // TODO: Fix buffering problem
  // Ideally, we want to buffer port A instead of port B because the critical path
  // is on the second port anyway due to the transpose. This means that we need to
  // insert a large fifo on the Q path to latency/cycle match the K path to
  // prevent throughput issues and deadlocks.
  matmul #(
      // Port A: q_out
      .A_TOTAL_DIM0  (TOTAL_HEAD_DIM),
      .A_TOTAL_DIM1  (TOTAL_SEQUENCE_DIM),
      .A_COMPUTE_DIM0(COMPUTE_HEAD_DIM),
      .A_COMPUTE_DIM1(COMPUTE_SEQUENCE_DIM),
      .A_WIDTH       (Q_OUT_WIDTH),
      .A_FRAC_WIDTH  (Q_OUT_FRAC_WIDTH),
      // Port B: k_transpose
      .B_TOTAL_DIM0  (TOTAL_SEQUENCE_DIM),
      .B_TOTAL_DIM1  (TOTAL_HEAD_DIM),
      .B_COMPUTE_DIM0(COMPUTE_SEQUENCE_DIM),
      .B_COMPUTE_DIM1(COMPUTE_HEAD_DIM),
      .B_WIDTH       (K_ACT_WIDTH),
      .B_FRAC_WIDTH  (K_ACT_FRAC_WIDTH),
      // Output
      .OUT_WIDTH     (QK_OUT_WIDTH),
      .OUT_FRAC_WIDTH(QK_OUT_FRAC_WIDTH),
      .OUT_SYMMETRIC (0)
  ) qk_matmul (
      .clk      (clk),
      .rst      (rst),
      .a_data   (q_out_data),
      .a_valid  (q_out_valid),
      .a_ready  (q_out_ready),
      .b_data   (k_transposed_act_data),
      .b_valid  (k_transposed_act_valid),
      .b_ready  (k_transposed_act_ready),
      .out_data (qk_out_data),
      .out_valid(qk_out_valid),
      .out_ready(qk_out_ready)
  );

  fixed_softermax_2d #(
      .TOTAL_DIM0    (TOTAL_SEQUENCE_DIM),
      .TOTAL_DIM1    (TOTAL_SEQUENCE_DIM),
      .COMPUTE_DIM0  (COMPUTE_SEQUENCE_DIM),
      .COMPUTE_DIM1  (COMPUTE_SEQUENCE_DIM),
      .IN_WIDTH      (QK_OUT_WIDTH),
      .IN_FRAC_WIDTH (QK_OUT_FRAC_WIDTH),
      .POW2_WIDTH    (SOFTERMAX_POW2_WIDTH),
      .OUT_WIDTH     (SOFTERMAX_OUT_WIDTH),
      .OUT_FRAC_WIDTH(SOFTERMAX_OUT_FRAC_WIDTH)
  ) qk_softermax (
      .clk      (clk),
      .rst      (rst),
      .in_data  (qk_out_data),
      .in_valid (qk_out_valid),
      .in_ready (qk_out_ready),
      .out_data (softermax_out_data),
      .out_valid(softermax_out_valid),
      .out_ready(softermax_out_ready)
  );

  // Unsigned pad 0 to softmax result
  for (
      genvar i = 0; i < COMPUTE_SEQUENCE_DIM * COMPUTE_SEQUENCE_DIM; i++
  ) begin : gen_softermax_unsigned
    assign softermax_unsigned_out_data[i] = {1'b0, softermax_out_data[i]};
  end

  matmul #(
      // Port A: softermax attention scores
      .A_TOTAL_DIM0  (TOTAL_SEQUENCE_DIM),
      .A_TOTAL_DIM1  (TOTAL_SEQUENCE_DIM),
      .A_COMPUTE_DIM0(COMPUTE_SEQUENCE_DIM),
      .A_COMPUTE_DIM1(COMPUTE_SEQUENCE_DIM),
      .A_WIDTH       (SOFTERMAX_OUT_WIDTH + 1),   // Added 1 bit for unsigned
      .A_FRAC_WIDTH  (SOFTERMAX_OUT_FRAC_WIDTH),
      // Port B: value matrix
      .B_TOTAL_DIM0  (TOTAL_HEAD_DIM),
      .B_TOTAL_DIM1  (TOTAL_SEQUENCE_DIM),
      .B_COMPUTE_DIM0(COMPUTE_HEAD_DIM),
      .B_COMPUTE_DIM1(COMPUTE_SEQUENCE_DIM),
      .B_WIDTH       (V_ACT_WIDTH),
      .B_FRAC_WIDTH  (V_ACT_FRAC_WIDTH),
      // Output
      .OUT_WIDTH     (OUT_ACT_WIDTH),
      .OUT_FRAC_WIDTH(OUT_ACT_FRAC_WIDTH),
      .OUT_SYMMETRIC (0)
  ) attn_matmul (
      .clk      (clk),
      .rst      (rst),
      .a_data   (softermax_unsigned_out_data),
      .a_valid  (softermax_out_valid),
      .a_ready  (softermax_out_ready),
      .b_data   (v_act_data),
      .b_valid  (v_act_valid),
      .b_ready  (v_act_ready),
      .out_data (out_act_data),
      .out_valid(out_act_valid),
      .out_ready(out_act_ready)
  );

endmodule
