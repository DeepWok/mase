/*
Module      : matrix_stream_transpose
Description : This module handles transposing a matrix which is being streamed
              one compute chunk at a time.

              Note: If you need a simpler combinatorial transpose for a
              non-streaming architecture, see transpose.sv
*/

`timescale 1ns / 1ps

// TODO: fix throughput problems

module matrix_stream_transpose #(
    // Total dimensions
    parameter TOTAL_DIM0 = 4,
    parameter TOTAL_DIM1 = 4,

    // Compute dimensions
    parameter COMPUTE_DIM0 = 2,
    parameter COMPUTE_DIM1 = 2,

    // Other params
    parameter DATA_WIDTH = 8
) (
    input logic clk,
    input logic rst,

    // In Matrix
    input  logic [DATA_WIDTH-1:0] in_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    input  logic                  in_valid,
    output logic                  in_ready,

    // Out Matrix
    output logic [DATA_WIDTH-1:0] out_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    output logic                  out_valid,
    input  logic                  out_ready
);

  initial begin
    // Check compute vs. total divisibility
    assert (TOTAL_DIM0 % COMPUTE_DIM0 == 0)
    else $fatal("DIM0 compute is not divisible!");
    assert (TOTAL_DIM1 % COMPUTE_DIM1 == 0)
    else $fatal("DIM1 compute is not divisible!");
  end

  // -----
  // Parameters
  // -----
  // let max(a, b) = (a > b) ? a : b;

  localparam IN_DEPTH_DIM0 = TOTAL_DIM0 / COMPUTE_DIM0;
  localparam IN_DEPTH_DIM1 = TOTAL_DIM1 / COMPUTE_DIM1;
  localparam OUT_DEPTH_DIM0 = IN_DEPTH_DIM1;
  localparam OUT_DEPTH_DIM1 = IN_DEPTH_DIM0;
  localparam IN_ROW_COUNTER_WIDTH = $clog2(IN_DEPTH_DIM1) > 1 ? $clog2(IN_DEPTH_DIM1) : 1;
  localparam IN_COL_COUNTER_WIDTH = $clog2(IN_DEPTH_DIM0) > 1 ? $clog2(IN_DEPTH_DIM0) : 1;
  localparam OUT_ROW_COUNTER_WIDTH = $clog2(OUT_DEPTH_DIM1) > 1 ? $clog2(OUT_DEPTH_DIM1) : 1;
  localparam OUT_COL_COUNTER_WIDTH = $clog2(OUT_DEPTH_DIM0) > 1 ? $clog2(OUT_DEPTH_DIM0) : 1;

  localparam FIFO_DEPTH = IN_DEPTH_DIM1;
  localparam FIFO_DATA_WIDTH = DATA_WIDTH * COMPUTE_DIM0 * COMPUTE_DIM1;

  // -----
  // State
  // -----

  struct {
    // Current row & col that the window is at for the input
    logic [IN_ROW_COUNTER_WIDTH-1:0]  in_row_count;
    logic [IN_COL_COUNTER_WIDTH-1:0]  in_col_count;
    // Current row & col that the window is at for the output
    logic [OUT_ROW_COUNTER_WIDTH-1:0] out_row_count;
    logic [OUT_COL_COUNTER_WIDTH-1:0] out_col_count;
  }
      self, next_self;

  // -----
  // Wires
  // -----

  logic [FIFO_DATA_WIDTH-1:0] in_data_flat;
  logic [FIFO_DATA_WIDTH-1:0] fifo_in_data[IN_DEPTH_DIM0-1:0];
  logic fifo_in_valid[IN_DEPTH_DIM0-1:0];
  logic fifo_in_ready[IN_DEPTH_DIM0-1:0];
  logic [FIFO_DATA_WIDTH-1:0] fifo_out_data_flat[IN_DEPTH_DIM0-1:0];
  logic fifo_out_valid[IN_DEPTH_DIM0-1:0];
  logic fifo_out_ready[IN_DEPTH_DIM0-1:0];

  logic fifo_data_readys[IN_DEPTH_DIM0-1:0];

  logic [FIFO_DATA_WIDTH-1:0] fifo_out_data_flat_mux_in[IN_DEPTH_DIM0-1:0];
  logic [FIFO_DATA_WIDTH-1:0] fifo_out_data_flat_mux_out;
  logic fifo_out_valids[IN_DEPTH_DIM0-1:0];

  logic [DATA_WIDTH-1:0] transpose_data_in[COMPUTE_DIM0*COMPUTE_DIM1-1:0];


  // FIFOs
  // We want to generate IN_DEPTH_DIM0 FIFOs to buffer the input chunks.
  // Each FIFO will need to be IN_DEPTH_DIM1 elements deep and each element will
  // be flattened to be size (DATA_WIDTH * COMPUTE_DIM0 * COMPUTE_DIM1)

  matrix_flatten #(
      .DATA_WIDTH(DATA_WIDTH),
      .DIM0      (COMPUTE_DIM0),
      .DIM1      (COMPUTE_DIM1)
  ) flatten_inst (
      .data_in (in_data),
      .data_out(in_data_flat)
  );

  for (genvar i = 0; i < IN_DEPTH_DIM0; i++) begin : fifos
    fifo #(
        .DEPTH     (FIFO_DEPTH),
        .DATA_WIDTH(FIFO_DATA_WIDTH)
    ) fifo_inst (
        .clk      (clk),
        .rst      (rst),
        .in_data  (fifo_in_data[i]),
        .in_valid (fifo_in_valid[i]),
        .in_ready (fifo_in_ready[i]),
        .out_data (fifo_out_data_flat[i]),
        .out_valid(fifo_out_valid[i]),
        .out_ready(fifo_out_ready[i]),
        .empty    (),
        .full     ()
    );
  end

  // Connect up wires to write to all of the fifos using in_col_count as index
  // The valid and ready signals will be used to select which one is written to
  for (genvar i = 0; i < IN_DEPTH_DIM0; i++) begin
    assign fifo_in_data[i] = in_data_flat;
    assign fifo_in_valid[i] = (self.in_col_count == i) ? in_valid : 0;
    assign fifo_data_readys[i] = fifo_in_ready[i];
  end

  generate
    if (IN_DEPTH_DIM0 > 1) begin
      mux #(
          .NUM_INPUTS(IN_DEPTH_DIM0),
          .DATA_WIDTH(1)
      ) in_ready_mux (
          .data_in (fifo_data_readys),
          .select  (self.in_col_count),
          .data_out(in_ready)
      );
    end else begin
      assign in_ready = fifo_data_readys[0];
    end
  endgenerate

  // Connect up wires to read from all of the fifos using out_row_count to index
  // into the column fifos which buffer the matrix

  for (genvar i = 0; i < IN_DEPTH_DIM0; i++) begin
    assign fifo_out_data_flat_mux_in[i] = fifo_out_data_flat[i];
    assign fifo_out_valids[i] = fifo_out_valid[i];
    assign fifo_out_ready[i] = (self.out_row_count == i) ? out_ready : 0;
  end

  generate
    if (IN_DEPTH_DIM0 > 1) begin
      mux #(
          .NUM_INPUTS(IN_DEPTH_DIM0),
          .DATA_WIDTH(FIFO_DATA_WIDTH)
      ) fifo_data_out_mux (
          .data_in (fifo_out_data_flat_mux_in),
          .select  (self.out_row_count),
          .data_out(fifo_out_data_flat_mux_out)
      );
      mux #(
          .NUM_INPUTS(IN_DEPTH_DIM0),
          .DATA_WIDTH(1)
      ) fifo_data_valid_mux (
          .data_in (fifo_out_valids),
          .select  (self.out_row_count),
          .data_out(out_valid)
      );
    end else begin
      assign fifo_out_data_flat_mux_out = fifo_out_data_flat_mux_in[0];
      assign out_valid = fifo_out_valids[0];
    end
  endgenerate

  // Unflatten FIFO data
  matrix_unflatten #(
      .DATA_WIDTH(DATA_WIDTH),
      .DIM0      (COMPUTE_DIM0),
      .DIM1      (COMPUTE_DIM1)
  ) pre_transpose_flatten_inst (
      .data_in (fifo_out_data_flat_mux_out),
      .data_out(transpose_data_in)
  );

  // Combinatorial transpose module
  transpose #(
      .WIDTH(DATA_WIDTH),
      .DIM0 (COMPUTE_DIM0),
      .DIM1 (COMPUTE_DIM1)
  ) transpose_inst (
      .in_data (transpose_data_in),
      .out_data(out_data)
  );

  always_comb begin
    next_self = self;

    // Increment input side counters
    if (in_valid && in_ready) begin
      if (self.in_row_count == IN_DEPTH_DIM1 - 1 && self.in_col_count == IN_DEPTH_DIM0 - 1) begin
        // End of matrix
        next_self.in_row_count = 0;
        next_self.in_col_count = 0;
      end else if (self.in_col_count == IN_DEPTH_DIM0 - 1) begin
        // End of row
        next_self.in_row_count = self.in_row_count + 1;
        next_self.in_col_count = 0;
      end else begin
        // Increment col counter
        next_self.in_col_count = self.in_col_count + 1;
      end
    end

    // Increment output side counters
    if (out_valid && out_ready) begin
      if (self.out_row_count == OUT_DEPTH_DIM1 - 1 &&
            self.out_col_count == OUT_DEPTH_DIM0 - 1) begin
        // End of matrix
        next_self.out_row_count = 0;
        next_self.out_col_count = 0;
      end else if (self.out_col_count == OUT_DEPTH_DIM0 - 1) begin
        // End of row
        next_self.out_row_count = self.out_row_count + 1;
        next_self.out_col_count = 0;
      end else begin
        // Increment col counter
        next_self.out_col_count = self.out_col_count + 1;
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
