/*
Module      : simple_matmul
Description : This module does a matrix multiplcation between matrices X & Y.

              The dimensions for the matrix multiplcation are:
              n x m * m x k

              or in MASE naming convention
              a_dim1 x a_dim0 * b_dim1 x bdim_0

              Python equivalent:
              out = np.matmul(X, Y)
*/

`timescale 1ns / 1ps

module simple_matmul #(
    // Dimensions
    parameter N               = 2,
    parameter M               = 2,
    parameter K               = 2,
    // Input fixed point widths
    parameter X_WIDTH         = 8,
    parameter X_FRAC_WIDTH    = 1,
    parameter Y_WIDTH         = 8,
    parameter Y_FRAC_WIDTH    = 1,
    // Output fixed point widths
    // if OUTPUT_ROUNDING == 0:
    // then out_width & out_frac_width must match accumulator widths
    parameter OUTPUT_ROUNDING = 1,
    parameter OUT_WIDTH       = 16,
    parameter OUT_FRAC_WIDTH  = 2
) (
    input logic clk,
    input logic rst,

    // Input matrix X, row-wise ordering
    input  logic [X_WIDTH-1:0] x_data [N*M-1:0],
    input  logic               x_valid,
    output logic               x_ready,

    // Input matrix Y, column-wise ordering
    input  logic [Y_WIDTH-1:0] y_data [M*K-1:0],
    input  logic               y_valid,
    output logic               y_ready,

    // Output matrix
    output logic [OUT_WIDTH-1:0] out_data [N*K-1:0],
    output logic                 out_valid,
    input  logic                 out_ready
);

  // -----
  // Params
  // -----

  // Accumulator widths in linear layer
  localparam ACC_WIDTH = X_WIDTH + Y_WIDTH + $clog2(M);
  localparam ACC_FRAC_WIDTH = X_FRAC_WIDTH + Y_FRAC_WIDTH;

  initial begin
    if (OUTPUT_ROUNDING == 0) begin
      assert (ACC_WIDTH == OUT_WIDTH)
      else $fatal("OUT_WIDTH must be %d if OUTPUT_ROUNDING == 0", ACC_WIDTH);
      assert (ACC_FRAC_WIDTH == OUT_FRAC_WIDTH)
      else $fatal("OUT_FRAC_WIDTH must be %d if OUTPUT_ROUNDING == 0", ACC_FRAC_WIDTH);
    end
  end


  // -----
  // Wires
  // -----

  logic [Y_WIDTH-1:0] y_data_transpose[K*M-1:0];
  logic dot_product_ready;
  logic inputs_valid, inputs_ready;

  logic [N*K-1:0] dot_product_valid;
  logic [N*K-1:0] sync_ready;
  logic [ACC_WIDTH-1:0] dot_product_data_out[N*K-1:0];
  logic [OUT_WIDTH-1:0] rounded_dot_product[N*K-1:0];


  // -----
  // Logic
  // -----

  // Need to synchronise x & y inputs
  assign inputs_ready = sync_ready[0];
  join2 sync_handshake (
      .data_in_valid ({x_valid, y_valid}),
      .data_in_ready ({x_ready, y_ready}),
      .data_out_valid(inputs_valid),
      .data_out_ready(inputs_ready)
  );

  // Transpose y to make column assignment easier, this module is just a rewire
  // so it shouldn't contribute anything to comb path.
  transpose #(
      .WIDTH(Y_WIDTH),
      .DIM0 (K),
      .DIM1 (M)
  ) y_transpose (
      .in_data (y_data),
      .out_data(y_data_transpose)
  );

  // Instantiate N-by-K number of dot products
  for (genvar i = 0; i < N; i++) begin : multi_row
    for (genvar j = 0; j < K; j++) begin : multi_col

      fixed_dot_product #(
          .IN_WIDTH    (X_WIDTH),
          .IN_SIZE     (M),
          .WEIGHT_WIDTH(Y_WIDTH)
      ) dot_product_inst (
          .clk           (clk),
          .rst           (rst),
          .data_in       (x_data[((i+1)*M)-1 : i*M]),
          .data_in_valid (inputs_valid),
          .data_in_ready (sync_ready[i*K+j]),
          .weight        (y_data_transpose[((j+1)*M)-1 : j*M]),
          .weight_valid  (inputs_valid),
          /* verilator lint_off PINCONNECTEMPTY */
          // This pin is the same as data_in_ready pin
          .weight_ready  (),
          /* verilator lint_on PINCONNECTEMPTY */
          .data_out      (dot_product_data_out[i*K+j]),
          .data_out_valid(dot_product_valid[i*K+j]),
          .data_out_ready(dot_product_ready)
      );

      if (OUTPUT_ROUNDING) begin : rounding
        // Rounded output
        fixed_round #(
            .IN_WIDTH      (ACC_WIDTH),
            .IN_FRAC_WIDTH (ACC_FRAC_WIDTH),
            .OUT_WIDTH     (OUT_WIDTH),
            .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH)
        ) round_inst (
            .data_in (dot_product_data_out[i*K+j]),
            .data_out(rounded_dot_product[i*K+j])
        );
        assign out_data[i*K+j] = rounded_dot_product[i*K+j];
      end else begin : no_rounding
        assign out_data[i*K+j] = dot_product_data_out[i*K+j];
      end

    end
  end

  assign out_valid = dot_product_valid[0];
  assign dot_product_ready = out_ready;

endmodule
