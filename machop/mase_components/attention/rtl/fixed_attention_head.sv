/*
Module      : fixed_attention_head

Description : Implements attention with no input buffering and parameterised
              intermediate fixed-point widths. All of the inputs Q, K, and V
              have a linear layer where the embedding dimension of the input can
              reduced from EMBEDDING_DIM to HEAD_DIM for this particular head.

Assumptions : 1. All activation inputs share same total and compute dimensions.
              2. All weight inputs share same total and compute dimensions.
              3. Activations and weights share same compute dimensions.

Dimensions  : q_act_data: (TOTAL_SEQUENCE_DIM x TOTAL_EMBEDDING_DIM)
              q_weight_data: (TOTAL_EMBEDDING_DIM x TOTAL_HEAD_DIM)
*/

`timescale 1ns/1ps

module fixed_attention_head #(
    // Dimensions
    parameter TOTAL_EMBEDDING_DIM       = 32,
    parameter TOTAL_HEAD_DIM            = 16,
    parameter TOTAL_SEQUENCE_DIM        = 16,
    parameter COMPUTE_EMBEDDING_DIM     = 4,
    parameter COMPUTE_HEAD_DIM          = 4,
    parameter COMPUTE_SEQUENCE_DIM      = 4,

    // Input Port Widths
    parameter Q_ACT_WIDTH               = 8,
    parameter Q_ACT_FRAC_WIDTH          = 2,
    parameter Q_WEIGHT_WIDTH            = 8,
    parameter Q_WEIGHT_FRAC_WIDTH       = 2,

    parameter K_ACT_WIDTH               = 8,
    parameter K_ACT_FRAC_WIDTH          = 2,
    parameter K_WEIGHT_WIDTH            = 8,
    parameter K_WEIGHT_FRAC_WIDTH       = 2,

    parameter V_ACT_WIDTH               = 8,
    parameter V_ACT_FRAC_WIDTH          = 2,
    parameter V_WEIGHT_WIDTH            = 8,
    parameter V_WEIGHT_FRAC_WIDTH       = 2,

    // Output Port Widths
    parameter OUT_ACT_WIDTH             = 8,
    parameter OUT_ACT_FRAC_WIDTH        = 2,

    // Intermediate widths
    parameter Q_OUT_WIDTH               = 16,
    parameter Q_OUT_FRAC_WIDTH          = 4,
    parameter K_OUT_WIDTH               = 16,
    parameter K_OUT_FRAC_WIDTH          = 4,
    parameter V_OUT_WIDTH               = 16,
    parameter V_OUT_FRAC_WIDTH          = 4,
    parameter QK_OUT_WIDTH              = 16,
    parameter QK_OUT_FRAC_WIDTH         = 4,
    parameter SOFTERMAX_POW2_WIDTH      = 16
    parameter SOFTERMAX_OUT_WIDTH       = 16,
    parameter SOFTERMAX_OUT_FRAC_WIDTH  = 4,

    // Derived Params
    localparam ACTIVATION_PARALLELISM   = COMPUTE_SEQUENCE_DIM * COMPUTE_EMBEDDING_DIM,
    localparam WEIGHT_PARALLELISM       = COMPUTE_EMBEDDING_DIM * COMPUTE_HEAD_DIM
) (
    input  logic                      clk,
    input  logic                      rst,

    // Query Activation & Weight Matrices
    input  logic [Q_ACT_WIDTH-1:0]    q_act_data [ACTIVATION_PARALLELISM-1:0],
    input  logic                      q_act_valid,
    output logic                      q_act_ready,

    input  logic [Q_WEIGHT_WIDTH-1:0] q_weight_data [WEIGHT_PARALLELISM-1:0],
    input  logic                      q_weight_valid,
    output logic                      q_weight_ready,

    // Key Activation & Weight Matrices
    input  logic [K_ACT_WIDTH-1:0]    k_act_data [ACTIVATION_PARALLELISM-1:0],
    input  logic                      k_act_valid,
    output logic                      k_act_ready,

    input  logic [K_WEIGHT_WIDTH-1:0] k_weight_data [WEIGHT_PARALLELISM-1:0],
    input  logic                      k_weight_valid,
    output logic                      k_weight_ready,

    // Value Activation & Weight Matrices
    input  logic [V_ACT_WIDTH-1:0]    v_act_data [ACTIVATION_PARALLELISM-1:0],
    input  logic                      v_act_valid,
    output logic                      v_act_ready,

    input  logic [V_WEIGHT_WIDTH-1:0] v_weight_data [WEIGHT_PARALLELISM-1:0],
    input  logic                      v_weight_valid,
    output logic                      v_weight_ready,

    // Output Activation Matrix
    output logic [OUT_ACT_WIDTH-1:0]  out_act_data [ACTIVATION_PARALLELISM-1:0],
    output logic                      out_act_valid,
    input  logic                      out_act_ready
);

// -----
// Params
// -----

// localparam QKV_OUT_PARALLELISM = COMPUTE_SEQUENCE_DIM*COMPUTE_HEAD_DIM;

// -----
// Wires
// -----

logic [Q_OUT_WIDTH-1:0] q_out_data [COMPUTE_SEQUENCE_DIM*COMPUTE_HEAD_DIM-1:0];
logic q_out_valid, q_out_ready;

logic [K_OUT_WIDTH-1:0] k_out_data [COMPUTE_SEQUENCE_DIM*COMPUTE_HEAD_DIM-1:0];
logic k_out_valid, k_out_ready;

logic [V_OUT_WIDTH-1:0] v_out_data [COMPUTE_SEQUENCE_DIM*COMPUTE_HEAD_DIM-1:0];
logic v_out_valid, v_out_ready;

logic [K_OUT_WIDTH-1:0] k_transpose_data [COMPUTE_SEQUENCE_DIM*COMPUTE_HEAD_DIM-1:0];
logic k_transpose_valid, k_transpose_ready;

logic [QK_OUT_WIDTH-1:0] qk_out_data [COMPUTE_SEQUENCE_DIM*COMPUTE_SEQUENCE_DIM-1:0];
logic qk_out_valid, qk_out_ready;

logic [SOFTERMAX_OUT_WIDTH-1:0] softermax_out_data [COMPUTE_SEQUENCE_DIM*COMPUTE_SEQUENCE_DIM-1:0];
logic softermax_out_valid, softermax_out_ready;


// -----
// Modules
// -----

matmul #(
    // Activations
    .A_TOTAL_DIM0    (TOTAL_EMBEDDING_DIM),
    .A_TOTAL_DIM1    (TOTAL_SEQUENCE_DIM),
    .A_COMPUTE_DIM0  (COMPUTE_EMBEDDING_DIM),
    .A_COMPUTE_DIM1  (COMPUTE_SEQUENCE_DIM),
    .A_WIDTH         (Q_ACT_WIDTH),
    .A_FRAC_WIDTH    (Q_ACT_FRAC_WIDTH),
    // Weights
    .B_TOTAL_DIM0    (TOTAL_HEAD_DIM),
    .B_TOTAL_DIM1    (TOTAL_EMBEDDING_DIM),
    .B_COMPUTE_DIM0  (COMPUTE_HEAD_DIM),
    .B_COMPUTE_DIM1  (COMPUTE_EMBEDDING_DIM),
    .B_WIDTH         (Q_WEIGHT_WIDTH),
    .B_FRAC_WIDTH    (Q_WEIGHT_FRAC_WIDTH),
    // Output
    .OUT_WIDTH       (Q_OUT_WIDTH),
    .OUT_FRAC_WIDTH  (Q_OUT_FRAC_WIDTH),
    .OUT_SYMMETRIC   (0)
) q_matmul (
    .clk             (clk),
    .rst             (rst),
    .a_data          (q_act_data),
    .a_valid         (q_act_valid),
    .a_ready         (q_act_ready),
    .b_data          (q_weight_data),
    .b_valid         (q_weight_valid),
    .b_ready         (q_weight_ready),
    .out_data        (q_out_data),
    .out_valid       (q_out_valid),
    .out_ready       (q_out_ready)
);

matmul #(
    // Activations
    .A_TOTAL_DIM0    (TOTAL_EMBEDDING_DIM),
    .A_TOTAL_DIM1    (TOTAL_SEQUENCE_DIM),
    .A_COMPUTE_DIM0  (COMPUTE_EMBEDDING_DIM),
    .A_COMPUTE_DIM1  (COMPUTE_SEQUENCE_DIM),
    .A_WIDTH         (K_ACT_WIDTH),
    .A_FRAC_WIDTH    (K_ACT_FRAC_WIDTH),
    // Weights
    .B_TOTAL_DIM0    (TOTAL_HEAD_DIM),
    .B_TOTAL_DIM1    (TOTAL_EMBEDDING_DIM),
    .B_COMPUTE_DIM0  (COMPUTE_HEAD_DIM),
    .B_COMPUTE_DIM1  (COMPUTE_EMBEDDING_DIM),
    .B_WIDTH         (K_WEIGHT_WIDTH),
    .B_FRAC_WIDTH    (K_WEIGHT_FRAC_WIDTH),
    // Output
    .OUT_WIDTH       (K_OUT_WIDTH),
    .OUT_FRAC_WIDTH  (K_OUT_FRAC_WIDTH),
    .OUT_SYMMETRIC   (0)
) k_matmul (
    .clk             (clk),
    .rst             (rst),
    .a_data          (k_act_data),
    .a_valid         (k_act_valid),
    .a_ready         (k_act_ready),
    .b_data          (k_weight_data),
    .b_valid         (k_weight_valid),
    .b_ready         (k_weight_ready),
    .out_data        (k_out_data),
    .out_valid       (k_out_valid),
    .out_ready       (k_out_ready)
);

matmul #(
    // Activations
    .A_TOTAL_DIM0    (TOTAL_EMBEDDING_DIM),
    .A_TOTAL_DIM1    (TOTAL_SEQUENCE_DIM),
    .A_COMPUTE_DIM0  (COMPUTE_EMBEDDING_DIM),
    .A_COMPUTE_DIM1  (COMPUTE_SEQUENCE_DIM),
    .A_WIDTH         (V_ACT_WIDTH),
    .A_FRAC_WIDTH    (V_ACT_FRAC_WIDTH),
    // Weights
    .B_TOTAL_DIM0    (TOTAL_HEAD_DIM),
    .B_TOTAL_DIM1    (TOTAL_EMBEDDING_DIM),
    .B_COMPUTE_DIM0  (COMPUTE_HEAD_DIM),
    .B_COMPUTE_DIM1  (COMPUTE_EMBEDDING_DIM),
    .B_WIDTH         (V_WEIGHT_WIDTH),
    .B_FRAC_WIDTH    (V_WEIGHT_FRAC_WIDTH),
    // Output
    .OUT_WIDTH       (V_OUT_WIDTH),
    .OUT_FRAC_WIDTH  (V_OUT_FRAC_WIDTH),
    .OUT_SYMMETRIC   (0)
) v_matmul (
    .clk             (clk),
    .rst             (rst),
    .a_data          (v_act_data),
    .a_valid         (v_act_valid),
    .a_ready         (v_act_ready),
    .b_data          (v_weight_data),
    .b_valid         (v_weight_valid),
    .b_ready         (v_weight_ready),
    .out_data        (v_out_data),
    .out_valid       (v_out_valid),
    .out_ready       (v_out_ready)
);

matrix_stream_transpose #(
    .TOTAL_DIM0      (TOTAL_HEAD_DIM),
    .TOTAL_DIM1      (TOTAL_SEQUENCE_DIM),
    .COMPUTE_DIM0    (COMPUTE_HEAD_DIM),
    .COMPUTE_DIM1    (COMPUTE_SEQUENCE_DIM),
    .DATA_WIDTH      (K_OUT_WIDTH)
) k_transpose (
    .clk             (clk),
    .rst             (rst),
    .in_data         (k_out_data),
    .in_valid        (k_out_valid),
    .in_ready        (k_out_ready),
    .out_data        (k_transpose_data),
    .out_valid       (k_transpose_valid),
    .out_ready       (k_transpose_ready)
);

// TODO: Fix buffering problem
// Ideally, we want to buffer port A instead of port B because the critical path
// is on the second port anyway due to the transpose. This means that we need to
// insert a large fifo on the Q path to latency/cycle match the K path to
// prevent throughput issues and deadlocks.
matmul #(
    // Port A: q_out
    .A_TOTAL_DIM0    (TOTAL_HEAD_DIM),
    .A_TOTAL_DIM1    (TOTAL_SEQUENCE_DIM),
    .A_COMPUTE_DIM0  (COMPUTE_HEAD_DIM),
    .A_COMPUTE_DIM1  (COMPUTE_SEQUENCE_DIM),
    .A_WIDTH         (Q_OUT_WIDTH),
    .A_FRAC_WIDTH    (Q_OUT_FRAC_WIDTH),
    // Port B: k_transpose
    .B_TOTAL_DIM0    (TOTAL_SEQUENCE_DIM),
    .B_TOTAL_DIM1    (TOTAL_HEAD_DIM),
    .B_COMPUTE_DIM0  (COMPUTE_SEQUENCE_DIM),
    .B_COMPUTE_DIM1  (COMPUTE_HEAD_DIM),
    .B_WIDTH         (K_OUT_WIDTH),
    .B_FRAC_WIDTH    (K_OUT_FRAC_WIDTH),
    // Output
    .OUT_WIDTH       (QK_OUT_WIDTH),
    .OUT_FRAC_WIDTH  (QK_OUT_FRAC_WIDTH),
    .OUT_SYMMETRIC   (0)
) qk_matmul (
    .clk             (clk),
    .rst             (rst),
    .a_data          (q_out_data),
    .a_valid         (q_out_valid),
    .a_ready         (q_out_ready),
    .b_data          (k_transpose_data),
    .b_valid         (k_transpose_valid),
    .b_ready         (k_transpose_ready),
    .out_data        (qk_out_data),
    .out_valid       (qk_out_valid),
    .out_ready       (qk_out_ready)
);

fixed_softermax_2d #(
    .TOTAL_DIM0      (TOTAL_SEQUENCE_DIM),
    .TOTAL_DIM1      (TOTAL_SEQUENCE_DIM),
    .COMPUTE_DIM0    (COMPUTE_SEQUENCE_DIM),
    .COMPUTE_DIM1    (COMPUTE_SEQUENCE_DIM),
    .IN_WIDTH        (QK_OUT_WIDTH),
    .IN_FRAC_WIDTH   (QK_OUT_FRAC_WIDTH),
    .POW2_WIDTH      (SOFTERMAX_POW2_WIDTH),
    .OUT_WIDTH       (SOFTERMAX_OUT_WIDTH),
    .OUT_FRAC_WIDTH  (SOFTERMAX_OUT_FRAC_WIDTH)
) qk_softermax (
    .clk             (clk),
    .rst             (rst),
    .in_data         (qk_out_data),
    .in_valid        (qk_out_valid),
    .in_ready        (qk_out_ready),
    .out_data        (softermax_out_data),
    .out_valid       (softermax_out_valid),
    .out_ready       (softermax_out_ready)
);

matmul #(
    // Port A: softermax attention scores
    .A_TOTAL_DIM0    (TOTAL_SEQUENCE_DIM),
    .A_TOTAL_DIM1    (TOTAL_SEQUENCE_DIM),
    .A_COMPUTE_DIM0  (COMPUTE_SEQUENCE_DIM),
    .A_COMPUTE_DIM1  (COMPUTE_SEQUENCE_DIM),
    .A_WIDTH         (SOFTERMAX_OUT_WIDTH),
    .A_FRAC_WIDTH    (SOFTERMAX_OUT_FRAC_WIDTH),
    // Port B: value matrix
    .B_TOTAL_DIM0    (TOTAL_HEAD_DIM),
    .B_TOTAL_DIM1    (TOTAL_SEQUENCE_DIM),
    .B_COMPUTE_DIM0  (COMPUTE_HEAD_DIM),
    .B_COMPUTE_DIM1  (COMPUTE_SEQUENCE_DIM),
    .B_WIDTH         (V_OUT_WIDTH),
    .B_FRAC_WIDTH    (V_OUT_FRAC_WIDTH),
    // Output
    .OUT_WIDTH       (OUT_ACT_WIDTH),
    .OUT_FRAC_WIDTH  (OUT_ACT_FRAC_WIDTH),
    .OUT_SYMMETRIC   (0)
) attn_matmul (
    .clk             (clk),
    .rst             (rst),
    .a_data          (softermax_out_data),
    .a_valid         (softermax_out_valid),
    .a_ready         (softermax_out_ready),
    .b_data          (v_out_data),
    .b_valid         (v_out_valid),
    .b_ready         (v_out_ready),
    .out_data        (out_act_data),
    .out_valid       (out_act_valid),
    .out_ready       (out_act_ready)
);

endmodule
