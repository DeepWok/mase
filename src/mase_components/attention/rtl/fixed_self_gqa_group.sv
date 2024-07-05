/*
Module      : fixed_self_gqa_group
Description : Implements a single group in grouped query self-attention (GQA).
*/

`timescale 1ns / 1ps

module fixed_self_gqa_group #(
    // GQA Parameters
    parameter GROUP_SIZE = 4,

    // Dimensions
    parameter TOTAL_EMBEDDING_DIM   = 32,
    parameter TOTAL_SEQUENCE_DIM    = 16,
    parameter COMPUTE_EMBEDDING_DIM = 4,
    parameter COMPUTE_SEQUENCE_DIM  = 4,

    // Input Port Widths
    parameter ACT_WIDTH           = 8,
    parameter ACT_FRAC_WIDTH      = 2,
    parameter Q_WEIGHT_WIDTH      = 8,
    parameter Q_WEIGHT_FRAC_WIDTH = 2,
    parameter K_WEIGHT_WIDTH      = 8,
    parameter K_WEIGHT_FRAC_WIDTH = 2,
    parameter V_WEIGHT_WIDTH      = 8,
    parameter V_WEIGHT_FRAC_WIDTH = 2,

    // Output Port Widths
    parameter OUT_ACT_WIDTH      = 8,
    parameter OUT_ACT_FRAC_WIDTH = 2,

    // Intermediate widths
    parameter Q_OUT_WIDTH              = 16,
    parameter Q_OUT_FRAC_WIDTH         = 4,
    parameter K_OUT_WIDTH              = 16,
    parameter K_OUT_FRAC_WIDTH         = 4,
    parameter V_OUT_WIDTH              = 16,
    parameter V_OUT_FRAC_WIDTH         = 4,
    parameter QK_OUT_WIDTH             = 16,
    parameter QK_OUT_FRAC_WIDTH        = 4,
    parameter SOFTERMAX_POW2_WIDTH     = 16,
    parameter SOFTERMAX_OUT_WIDTH      = 16,
    parameter SOFTERMAX_OUT_FRAC_WIDTH = 4,

    localparam TOTAL_HEAD_DIM   = TOTAL_EMBEDDING_DIM / GROUP_SIZE,
    localparam COMPUTE_HEAD_DIM = COMPUTE_EMBEDDING_DIM / GROUP_SIZE
) (
    input logic clk,
    input logic rst,

    // Input activations
    input  logic [ACT_WIDTH-1:0] act_data [COMPUTE_SEQUENCE_DIM*COMPUTE_EMBEDDING_DIM-1:0],
    input  logic                 act_valid,
    output logic                 act_ready,

    // GROUP_SIZE Channels of Query Weights
    input  logic [Q_WEIGHT_WIDTH-1:0] q_weight_data [GROUP_SIZE-1:0] [COMPUTE_EMBEDDING_DIM*COMPUTE_HEAD_DIM-1:0],
    input logic q_weight_valid,
    output logic q_weight_ready,

    // Single Channel Key Weights
    input  logic [K_WEIGHT_WIDTH-1:0] k_weight_data [COMPUTE_EMBEDDING_DIM*COMPUTE_HEAD_DIM-1:0],
    input  logic                      k_weight_valid,
    output logic                      k_weight_ready,

    // Single Channel Value Weights
    input  logic [V_WEIGHT_WIDTH-1:0] v_weight_data [COMPUTE_EMBEDDING_DIM*COMPUTE_HEAD_DIM-1:0],
    input  logic                      v_weight_valid,
    output logic                      v_weight_ready,

    // Output Activation Matrix
    output logic [OUT_ACT_WIDTH-1:0]  out_act_data [GROUP_SIZE-1:0] [COMPUTE_SEQUENCE_DIM*COMPUTE_HEAD_DIM-1:0],
    output logic out_act_valid,
    input logic out_act_ready
);


  initial begin
    // Check divisibility
    assert (TOTAL_HEAD_DIM * GROUP_SIZE == TOTAL_EMBEDDING_DIM);
    assert (COMPUTE_HEAD_DIM * GROUP_SIZE == COMPUTE_EMBEDDING_DIM);
  end

  // -----
  // Wires
  // -----

  logic [K_OUT_WIDTH-1:0] k_out_data[COMPUTE_SEQUENCE_DIM*COMPUTE_HEAD_DIM-1:0];
  logic k_out_valid, k_out_ready;

  logic [V_OUT_WIDTH-1:0] v_out_data[COMPUTE_SEQUENCE_DIM*COMPUTE_HEAD_DIM-1:0];
  logic v_out_valid, v_out_ready;

  logic [K_OUT_WIDTH-1:0] k_transpose_data[COMPUTE_SEQUENCE_DIM*COMPUTE_HEAD_DIM-1:0];
  logic k_transpose_valid;
  logic k_transpose_ready[GROUP_SIZE-1:0];

  logic [OUT_ACT_WIDTH-1:0] head_act_data [GROUP_SIZE-1:0] [COMPUTE_SEQUENCE_DIM*COMPUTE_HEAD_DIM-1:0];
  logic head_act_valid[GROUP_SIZE-1:0];
  logic head_act_ready;


  // -----
  // Modules
  // -----

  matmul #(
      // Activations
      .A_TOTAL_DIM0  (TOTAL_EMBEDDING_DIM),
      .A_TOTAL_DIM1  (TOTAL_SEQUENCE_DIM),
      .A_COMPUTE_DIM0(COMPUTE_EMBEDDING_DIM),
      .A_COMPUTE_DIM1(COMPUTE_SEQUENCE_DIM),
      .A_WIDTH       (ACT_WIDTH),
      .A_FRAC_WIDTH  (ACT_FRAC_WIDTH),
      // Weights
      .B_TOTAL_DIM0  (TOTAL_HEAD_DIM),
      .B_TOTAL_DIM1  (TOTAL_EMBEDDING_DIM),
      .B_COMPUTE_DIM0(COMPUTE_HEAD_DIM),
      .B_COMPUTE_DIM1(COMPUTE_EMBEDDING_DIM),
      .B_WIDTH       (K_WEIGHT_WIDTH),
      .B_FRAC_WIDTH  (K_WEIGHT_FRAC_WIDTH),
      // Output
      .OUT_WIDTH     (K_OUT_WIDTH),
      .OUT_FRAC_WIDTH(K_OUT_FRAC_WIDTH),
      .OUT_SYMMETRIC (0)
  ) k_matmul (
      .clk      (clk),
      .rst      (rst),
      .a_data   (act_data),
      .a_valid  (act_valid),
      .a_ready  (act_ready),
      .b_data   (k_weight_data),
      .b_valid  (k_weight_valid),
      .b_ready  (k_weight_ready),
      .out_data (k_out_data),
      .out_valid(k_out_valid),
      .out_ready(k_out_ready)
  );

  matmul #(
      // Activations
      .A_TOTAL_DIM0  (TOTAL_EMBEDDING_DIM),
      .A_TOTAL_DIM1  (TOTAL_SEQUENCE_DIM),
      .A_COMPUTE_DIM0(COMPUTE_EMBEDDING_DIM),
      .A_COMPUTE_DIM1(COMPUTE_SEQUENCE_DIM),
      .A_WIDTH       (ACT_WIDTH),
      .A_FRAC_WIDTH  (ACT_FRAC_WIDTH),
      // Weights
      .B_TOTAL_DIM0  (TOTAL_HEAD_DIM),
      .B_TOTAL_DIM1  (TOTAL_EMBEDDING_DIM),
      .B_COMPUTE_DIM0(COMPUTE_HEAD_DIM),
      .B_COMPUTE_DIM1(COMPUTE_EMBEDDING_DIM),
      .B_WIDTH       (V_WEIGHT_WIDTH),
      .B_FRAC_WIDTH  (V_WEIGHT_FRAC_WIDTH),
      // Output
      .OUT_WIDTH     (V_OUT_WIDTH),
      .OUT_FRAC_WIDTH(V_OUT_FRAC_WIDTH),
      .OUT_SYMMETRIC (0)
  ) v_matmul (
      .clk      (clk),
      .rst      (rst),
      .a_data   (act_data),
      .a_valid  (act_valid),
      .a_ready  (act_ready),
      .b_data   (v_weight_data),
      .b_valid  (v_weight_valid),
      .b_ready  (v_weight_ready),
      .out_data (v_out_data),
      .out_valid(v_out_valid),
      .out_ready(v_out_ready)
  );

  matrix_stream_transpose #(
      .TOTAL_DIM0  (TOTAL_HEAD_DIM),
      .TOTAL_DIM1  (TOTAL_SEQUENCE_DIM),
      .COMPUTE_DIM0(COMPUTE_HEAD_DIM),
      .COMPUTE_DIM1(COMPUTE_SEQUENCE_DIM),
      .DATA_WIDTH  (K_OUT_WIDTH)
  ) k_transpose (
      .clk      (clk),
      .rst      (rst),
      .in_data  (k_out_data),
      .in_valid (k_out_valid),
      .in_ready (k_out_ready),
      .out_data (k_transpose_data),
      .out_valid(k_transpose_valid),
      .out_ready(k_transpose_ready[0])
  );


  for (genvar head = 0; head < GROUP_SIZE; head++) begin : gqa_heads

    fixed_gqa_head #(
        // Dimensions
        .TOTAL_EMBEDDING_DIM     (TOTAL_EMBEDDING_DIM),
        .TOTAL_HEAD_DIM          (TOTAL_HEAD_DIM),
        .TOTAL_SEQUENCE_DIM      (TOTAL_SEQUENCE_DIM),
        .COMPUTE_EMBEDDING_DIM   (COMPUTE_EMBEDDING_DIM),
        .COMPUTE_HEAD_DIM        (COMPUTE_HEAD_DIM),
        .COMPUTE_SEQUENCE_DIM    (COMPUTE_SEQUENCE_DIM),
        // Q Activation & Weight Widths
        .Q_ACT_WIDTH             (ACT_WIDTH),
        .Q_ACT_FRAC_WIDTH        (ACT_FRAC_WIDTH),
        .Q_WEIGHT_WIDTH          (Q_WEIGHT_WIDTH),
        .Q_WEIGHT_FRAC_WIDTH     (Q_WEIGHT_FRAC_WIDTH),
        // K Activation Width
        .K_ACT_WIDTH             (K_OUT_WIDTH),
        .K_ACT_FRAC_WIDTH        (K_OUT_FRAC_WIDTH),
        // V Activation Width
        .V_ACT_WIDTH             (V_OUT_WIDTH),
        .V_ACT_FRAC_WIDTH        (V_OUT_FRAC_WIDTH),
        // Output Activation Width
        .OUT_ACT_WIDTH           (OUT_ACT_WIDTH),
        .OUT_ACT_FRAC_WIDTH      (OUT_ACT_FRAC_WIDTH),
        // Intermediate Q Matrix Mult Widths
        .Q_OUT_WIDTH             (Q_OUT_WIDTH),
        .Q_OUT_FRAC_WIDTH        (Q_OUT_FRAC_WIDTH),
        // Intermediate QK^T Matrix Mult Widths
        .QK_OUT_WIDTH            (QK_OUT_WIDTH),
        .QK_OUT_FRAC_WIDTH       (QK_OUT_FRAC_WIDTH),
        // Intermediate Softermax Widths
        .SOFTERMAX_POW2_WIDTH    (SOFTERMAX_POW2_WIDTH),
        .SOFTERMAX_OUT_WIDTH     (SOFTERMAX_OUT_WIDTH),
        .SOFTERMAX_OUT_FRAC_WIDTH(SOFTERMAX_OUT_FRAC_WIDTH)
    ) gqa_head_inst (
        .clk                   (clk),
        .rst                   (rst),
        // Q Activation & Weights in
        .q_act_data            (act_data),
        .q_act_valid           (act_valid),
        .q_act_ready           (act_ready),
        .q_weight_data         (q_weight_data[head]),
        .q_weight_valid        (q_weight_valid),
        .q_weight_ready        (q_weight_ready),
        // Shared K^T Data
        .k_transposed_act_data (k_transpose_data),
        .k_transposed_act_valid(k_transpose_valid),
        .k_transposed_act_ready(k_transpose_ready[head]),
        // Shared V
        .v_act_data            (v_out_data),
        .v_act_valid           (v_out_valid),
        .v_act_ready           (v_out_ready),
        .out_act_data          (head_act_data[head]),
        .out_act_valid         (head_act_valid[head]),
        .out_act_ready         (head_act_ready)
    );

  end

  assign out_act_data   = head_act_data;
  assign out_act_valid  = head_act_valid[0];
  assign head_act_ready = out_act_ready;


endmodule
