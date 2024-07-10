/*
Module      : softermax_local_window
Description : This module implements the first section of the softermax compute
              pipeline which calculates the local maximum and power of 2.

              Refer to the top half of Fig. 4.a) in the Softermax Paper.
              https://arxiv.org/abs/2103.09301
*/

`timescale 1ns / 1ps

module softermax_local_window #(
    // Input shape independent of TOTAL_DIM since this module only
    // operates on local windows
    parameter PARALLELISM = 4,

    // Widths
    parameter IN_WIDTH = 8,
    parameter IN_FRAC_WIDTH = 4,
    parameter OUT_WIDTH = 8,
    parameter OUT_FRAC_WIDTH = 7,

    // Derived params
    localparam MAX_WIDTH = IN_WIDTH - IN_FRAC_WIDTH
) (
    input logic clk,
    input logic rst,

    // Input streaming interface
    input  logic [IN_WIDTH-1:0] in_data [PARALLELISM-1:0],
    input  logic                in_valid,
    output logic                in_ready,

    // Output streaming interface with pow of 2 values & max
    output logic [OUT_WIDTH-1:0] out_values[PARALLELISM-1:0],
    output logic [MAX_WIDTH-1:0] out_max,
    output logic                 out_valid,
    input  logic                 out_ready
);

  // -----
  // Parameters
  // -----

  localparam MAX_TREE_DEPTH = $clog2(PARALLELISM);

  localparam SUBTRACT_WIDTH = IN_WIDTH + 1;
  localparam SUBTRACT_FRAC_WIDTH = IN_FRAC_WIDTH;

  initial begin
    assert (PARALLELISM > 1);
    assert (IN_WIDTH > IN_FRAC_WIDTH);
    assert (OUT_WIDTH > OUT_FRAC_WIDTH);
  end

  // -----
  // Wires
  // -----

  logic [MAX_WIDTH-1:0] rounded_int_in_data[PARALLELISM-1:0];
  logic [MAX_WIDTH-1:0] rounded_int_out_data[PARALLELISM-1:0];
  logic rounded_int_in_valid;
  logic rounded_int_in_ready[PARALLELISM-1:0];
  logic rounded_int_out_valid[PARALLELISM-1:0];
  logic rounded_int_out_ready;

  // logic [MAX_WIDTH-1:0] max_tree_in_data [PARALLELISM-1:0];
  logic [MAX_WIDTH-1:0] max_tree_out_data;
  // logic max_tree_in_valid, max_tree_in_ready;
  logic max_tree_out_valid, max_tree_out_ready;

  logic [IN_WIDTH-1:0] input_fifo_in_data [PARALLELISM-1:0];
  logic [IN_WIDTH-1:0] input_fifo_out_data[PARALLELISM-1:0];
  logic input_fifo_in_valid, input_fifo_in_ready;
  logic input_fifo_out_valid, input_fifo_out_ready;

  logic [MAX_WIDTH-1:0] max_fifo_in_data, max_fifo_out_data;
  logic max_fifo_in_valid, max_fifo_in_ready;
  logic max_fifo_out_valid, max_fifo_out_ready;

  logic [MAX_WIDTH-1:0] sub_max_in_data;
  logic [MAX_WIDTH-1:0] sub_max_out_data;
  logic sub_max_in_valid, sub_max_in_ready;
  logic sub_max_out_valid, sub_max_out_ready;

  logic [SUBTRACT_WIDTH-1:0] subtract_in_data[PARALLELISM-1:0];
  logic [SUBTRACT_WIDTH-1:0] subtract_out_data[PARALLELISM-1:0];
  logic subtract_in_valid;
  logic subtract_in_ready[PARALLELISM-1:0];
  logic subtract_out_valid[PARALLELISM-1:0];
  logic subtract_out_ready[PARALLELISM-1:0];

  logic [OUT_WIDTH-1:0] lpw_out_data[PARALLELISM-1:0];
  logic lpw_out_valid[PARALLELISM-1:0];
  logic lpw_out_ready;

  // -----
  // Modules
  // -----

  split2 input_split (
      .data_in_valid (in_valid),
      .data_in_ready (in_ready),
      .data_out_valid({rounded_int_in_valid, input_fifo_in_valid}),
      .data_out_ready({rounded_int_in_ready[0], input_fifo_in_ready})
  );

  for (genvar i = 0; i < PARALLELISM; i++) begin : rounding
    fixed_signed_cast #(
        .IN_WIDTH(IN_WIDTH),
        .IN_FRAC_WIDTH(IN_FRAC_WIDTH),
        .OUT_WIDTH(MAX_WIDTH),
        .OUT_FRAC_WIDTH(0),  // No fraction
        .SYMMETRIC(0),
        .ROUND_FLOOR(1)
    ) rounding_inst (
        .in_data (in_data[i]),
        .out_data(rounded_int_in_data[i])
    );

    // Required even though cast is only doing slice since theres another split
    skid_buffer #(
        .DATA_WIDTH(MAX_WIDTH)
    ) cast_buff (
        .clk(clk),
        .rst(rst),
        .data_in(rounded_int_in_data[i]),
        .data_in_valid(rounded_int_in_valid),
        .data_in_ready(rounded_int_in_ready[i]),
        .data_out(rounded_int_out_data[i]),
        .data_out_valid(rounded_int_out_valid[i]),
        .data_out_ready(rounded_int_out_ready)
    );
  end

  assign input_fifo_in_data = in_data;

  comparator_tree #(
      .SIZE(PARALLELISM),
      .DATA_WIDTH(MAX_WIDTH),
      .MAX1_MIN0(1),  // MAX
      .SIGNED(1)
  ) max_tree (
      .clk(clk),
      .rst(rst),
      .in_data(rounded_int_out_data),
      .in_valid(rounded_int_out_valid[0]),
      .in_ready(rounded_int_out_ready),
      .out_data(max_tree_out_data),
      .out_valid(max_tree_out_valid),
      .out_ready(max_tree_out_ready)
  );

  matrix_fifo #(
      .DATA_WIDTH(IN_WIDTH),
      .DIM0(PARALLELISM),
      .DIM1(1),
      .FIFO_SIZE(16)
  ) input_buffer (
      .clk(clk),
      .rst(rst),
      .in_data(input_fifo_in_data),
      .in_valid(input_fifo_in_valid),
      .in_ready(input_fifo_in_ready),
      .out_data(input_fifo_out_data),
      .out_valid(input_fifo_out_valid),
      .out_ready(input_fifo_out_ready)
  );

  assign max_fifo_in_data = max_tree_out_data;

  fifo #(
      .DATA_WIDTH(MAX_WIDTH),
      // Should be enough. Subtract takes 1 cycle and LPW is max 3 cycles
      .DEPTH(8)
  ) max_buffer (
      .clk(clk),
      .rst(rst),
      .in_data(max_fifo_in_data),
      .in_valid(max_fifo_in_valid),
      .in_ready(max_fifo_in_ready),
      .out_data(max_fifo_out_data),
      .out_valid(max_fifo_out_valid),
      .out_ready(max_fifo_out_ready),
      .empty(),
      .full()
  );

  split2 max_tree_split (
      .data_in_valid (max_tree_out_valid),
      .data_in_ready (max_tree_out_ready),
      .data_out_valid({sub_max_in_valid, max_fifo_in_valid}),
      .data_out_ready({sub_max_in_ready, max_fifo_in_ready})
  );

  assign sub_max_in_data = max_tree_out_data;

  skid_buffer #(
      .DATA_WIDTH(MAX_WIDTH)
  ) max_intermediate_buff (
      .clk(clk),
      .rst(rst),
      .data_in(sub_max_in_data),
      .data_in_valid(sub_max_in_valid),
      .data_in_ready(sub_max_in_ready),
      .data_out(sub_max_out_data),
      .data_out_valid(sub_max_out_valid),
      .data_out_ready(sub_max_out_ready)
  );

  join2 subtract_join (
      .data_in_valid ({sub_max_out_valid, input_fifo_out_valid}),
      .data_in_ready ({sub_max_out_ready, input_fifo_out_ready}),
      .data_out_valid(subtract_in_valid),
      .data_out_ready(subtract_in_ready[0])
  );

  // Batched subtract & power of 2
  for (genvar i = 0; i < PARALLELISM; i++) begin : subtract_pow2
    // Need to pad the maxInt with fixed-point zeros in fraction
    assign subtract_in_data[i] = $signed(
        input_fifo_out_data[i]
    ) - $signed(
        {sub_max_out_data, {IN_FRAC_WIDTH{1'b0}}}
    );

    skid_buffer #(
        .DATA_WIDTH(SUBTRACT_WIDTH)
    ) sub_reg (
        .clk(clk),
        .rst(rst),
        .data_in(subtract_in_data[i]),
        .data_in_valid(subtract_in_valid),
        .data_in_ready(subtract_in_ready[i]),
        .data_out(subtract_out_data[i]),
        .data_out_valid(subtract_out_valid[i]),
        .data_out_ready(subtract_out_ready[i])
    );

    softermax_lpw_pow2 #(
        .IN_WIDTH(SUBTRACT_WIDTH),
        .IN_FRAC_WIDTH(SUBTRACT_FRAC_WIDTH),
        .OUT_WIDTH(OUT_WIDTH),
        .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH)
    ) lpw_pow2 (
        .clk(clk),
        .rst(rst),
        .in_data(subtract_out_data[i]),
        .in_valid(subtract_out_valid[i]),
        .in_ready(subtract_out_ready[i]),
        .out_data(lpw_out_data[i]),
        .out_valid(lpw_out_valid[i]),
        .out_ready(lpw_out_ready)
    );

  end

  // Final synchronize
  join2 output_sync (
      .data_in_valid ({max_fifo_out_valid, lpw_out_valid[0]}),
      .data_in_ready ({max_fifo_out_ready, lpw_out_ready}),
      .data_out_valid(out_valid),
      .data_out_ready(out_ready)
  );

  assign out_values = lpw_out_data;
  assign out_max = max_fifo_out_data;

endmodule
