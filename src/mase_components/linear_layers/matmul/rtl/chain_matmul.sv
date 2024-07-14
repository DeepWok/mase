/*
Module      : test_chain_matmul
Description : This module should not be instantiated!
              It is used to test the repeated matrix multiplication.

              This module does (A * B) * C = D

              Dimensions are (nm * mk) * kz = nz

              Note:
              - Every matmul will use 2x2 compute window to keep it simple
              - All matrices will use 8-bit widths
*/

`timescale 1ns / 1ps

module chain_matmul #(
    parameter N = 4,
    parameter M = 4,
    parameter K = 2,
    parameter Z = 4,

    // Constants
    localparam COMPUTE_DIM0 = 2,
    localparam COMPUTE_DIM1 = 2,
    localparam IN_WIDTH = 4,
    localparam IN_FRAC_WIDTH = 1,
    localparam INT_WIDTH = 8,
    localparam INT_FRAC_WIDTH = 1,
    localparam OUT_WIDTH = 12,
    localparam OUT_FRAC_WIDTH = 1,
    localparam SYMMETRIC = 0
) (
    input logic clk,
    input logic rst,

    // Matix A
    input  logic [IN_WIDTH-1:0] a_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    input  logic                a_valid,
    output logic                a_ready,

    // Matix B
    input  logic [IN_WIDTH-1:0] b_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    input  logic                b_valid,
    output logic                b_ready,

    // Matrix C
    input  logic [IN_WIDTH-1:0] c_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    input  logic                c_valid,
    output logic                c_ready,

    // Matrix D - Output
    output logic [OUT_WIDTH-1:0] d_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    output logic                 d_valid,
    input  logic                 d_ready
);


  // Intermediate n-by-k matrix from A * B
  logic [INT_WIDTH-1:0] int_data[COMPUTE_DIM0*COMPUTE_DIM1-1:0];
  logic int_valid, int_ready;

  matmul #(
      // Dimensions
      .A_TOTAL_DIM0  (M),
      .A_TOTAL_DIM1  (N),
      .B_TOTAL_DIM0  (K),
      .B_TOTAL_DIM1  (M),
      .A_COMPUTE_DIM0(2),
      .A_COMPUTE_DIM1(2),
      .B_COMPUTE_DIM0(2),
      .B_COMPUTE_DIM1(2),
      // Input fixed point widths
      .A_WIDTH       (IN_WIDTH),
      .A_FRAC_WIDTH  (IN_FRAC_WIDTH),
      .B_WIDTH       (IN_WIDTH),
      .B_FRAC_WIDTH  (IN_FRAC_WIDTH),
      // Output fixed point widths
      .OUT_WIDTH     (INT_WIDTH),
      .OUT_FRAC_WIDTH(INT_FRAC_WIDTH),
      .OUT_SYMMETRIC (SYMMETRIC)
  ) matmul_0 (
      .clk      (clk),
      .rst      (rst),
      .a_data   (a_data),
      .a_valid  (a_valid),
      .a_ready  (a_ready),
      .b_data   (b_data),
      .b_valid  (b_valid),
      .b_ready  (b_ready),
      .out_data (int_data),
      .out_valid(int_valid),
      .out_ready(int_ready)
  );


  matmul #(
      .A_TOTAL_DIM0  (K),
      .A_TOTAL_DIM1  (N),
      .B_TOTAL_DIM0  (Z),
      .B_TOTAL_DIM1  (K),
      .A_COMPUTE_DIM0(2),
      .A_COMPUTE_DIM1(2),
      .B_COMPUTE_DIM0(2),
      .B_COMPUTE_DIM1(2),
      // Input fixed point widths
      .A_WIDTH       (INT_WIDTH),
      .A_FRAC_WIDTH  (INT_FRAC_WIDTH),
      .B_WIDTH       (IN_WIDTH),
      .B_FRAC_WIDTH  (IN_FRAC_WIDTH),
      // Output fixed point widths
      .OUT_WIDTH     (OUT_WIDTH),
      .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),
      .OUT_SYMMETRIC (SYMMETRIC)
  ) matmul_1 (
      .clk      (clk),
      .rst      (rst),
      .a_data   (int_data),
      .a_valid  (int_valid),
      .a_ready  (int_ready),
      .b_data   (c_data),
      .b_valid  (c_valid),
      .b_ready  (c_ready),
      .out_data (d_data),
      .out_valid(d_valid),
      .out_ready(d_ready)
  );


endmodule
