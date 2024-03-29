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
    parameter OUT_FRAC_WIDTH  = 0
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

  // Need to synchronise backpressure/valid signals
  // logic sync_valid, sync_ready;
  // join2 #() sync_handshake (
  //     .data_in_valid ({x_valid, y_valid}),
  //     .data_in_ready ({x_ready, y_ready}),
  //     .data_out_valid(sync_valid),
  //     .data_out_ready(sync_ready)
  // );


  logic [N*K-1:0] dot_product_ready;
  logic [N*K-1:0] dot_product_valid;
  assign dot_product_ready = {(N * K) {out_ready}};

  generate
    for (genvar i = 0; i < N; i++) begin : multi_row
      for (genvar j = 0; j < K; j++) begin : multi_col

        // Slice a single row of x
        logic [X_WIDTH-1:0] row_x[M-1:0];
        assign row_x = x_data[(i+1)*M-1 : i*M];

        // Slice a column of y
        logic [Y_WIDTH-1:0] col_y[M-1:0];
        for (genvar m = 0; m < M; m++) begin : col_assign
          assign col_y[m] = y_data[m*K+j];
        end

        // Input ready signal
        logic sync_ready;

        // Linear output
        logic [ACC_WIDTH-1:0] dot_product_data_out;

        fixed_dot_product #(
            .IN_WIDTH    (X_WIDTH),
            .IN_SIZE     (M),
            .WEIGHT_WIDTH(Y_WIDTH)
        ) linear_inst (
            .clk           (clk),
            .rst           (rst),
            .data_in       (row_x),
            .data_in_valid (sync_valid),
            .data_in_ready (sync_ready),
            .weight        (col_y),
            .weight_valid  (sync_valid),
            /* verilator lint_off PINCONNECTEMPTY */
            // This pin is the same as data_in_ready pin
            .weight_ready  (),
            /* verilator lint_on PINCONNECTEMPTY */
            .data_out      (dot_product_data_out),
            .data_out_valid(dot_product_valid[i*K+j]),
            .data_out_ready(dot_product_ready[i*K+j])
        );

        if (OUTPUT_ROUNDING) begin : rounding
          // Rounded output
          logic [OUT_WIDTH-1:0] rounded_dot_product;
          fixed_round #(
              .IN_WIDTH      (ACC_WIDTH),
              .IN_FRAC_WIDTH (ACC_FRAC_WIDTH),
              .OUT_WIDTH     (OUT_WIDTH),
              .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH)
          ) round_inst (
              .data_in (dot_product_data_out),
              .data_out(rounded_dot_product)
          );
          assign out_data[i*K+j] = rounded_dot_product;
        end else begin : no_rounding
          assign out_data[i*K+j] = dot_product_data_out;
        end

      end
    end
  endgenerate

  // Need to synchronise backpressure/valid signals
  logic sync_valid, join_sync_ready;
  assign join_sync_ready = multi_row[0].multi_col[0].sync_ready;

  join2 #() sync_handshake (
      .data_in_valid ({x_valid, y_valid}),
      .data_in_ready ({x_ready, y_ready}),
      .data_out_valid(sync_valid),
      .data_out_ready(join_sync_ready)
  );

  assign out_valid = &dot_product_valid;

endmodule
