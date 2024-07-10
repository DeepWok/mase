/*
Module      : comparator_tree
Description : This module implements a maximum number comparator tree.

              Can do signed/unsigned max/min reductions.
*/

`timescale 1ns / 1ps

module comparator_tree #(
    parameter SIZE = 8,  // Only supports powers of 2
    parameter DATA_WIDTH = 8,
    parameter MAX1_MIN0 = 1,  // MAX = 1, MIN = 0
    parameter SIGNED = 0
) (
    input logic clk,
    input logic rst,

    input  logic [DATA_WIDTH-1:0] in_data [SIZE-1:0],
    input  logic                  in_valid,
    output logic                  in_ready,

    output logic [DATA_WIDTH-1:0] out_data,
    output logic                  out_valid,
    input  logic                  out_ready
);

  localparam LEVELS = $clog2(SIZE);

  initial begin
    assert (2 ** LEVELS == SIZE);  // Only support power of 2
  end

  for (genvar level = 0; level <= LEVELS; level++) begin : vars
    logic [DATA_WIDTH-1:0] data[(2**(LEVELS-level))-1:0];
    logic valid;
    logic ready;
  end


  for (genvar level = 0; level < LEVELS; level++) begin : element_handshake
    logic [2 ** (LEVELS - level - 1) - 1 : 0] element_input_valid;
    logic [2 ** (LEVELS - level - 1) - 1 : 0] element_input_ready;

    logic [2 ** (LEVELS - level - 1) - 1 : 0] element_output_valid;
    logic [2 ** (LEVELS - level - 1) - 1 : 0] element_output_ready;
  end : element_handshake

  for (genvar i = 0; i < LEVELS; i++) begin : level

    for (genvar c = 0; c < 2 ** (LEVELS - i - 1); c++) begin : comparator

      logic [DATA_WIDTH-1:0] left, right, result;
      assign left  = vars[i].data[2*c];
      assign right = vars[i].data[2*c+1];

      if (MAX1_MIN0) begin
        if (SIGNED) begin
          assign result = $signed(left) > $signed(right) ? left : right;
        end else begin
          assign result = left > right ? left : right;
        end
      end else begin
        if (SIGNED) begin
          assign result = $signed(left) < $signed(right) ? left : right;
        end else begin
          assign result = left < right ? left : right;
        end
      end

      skid_buffer #(
          .DATA_WIDTH(DATA_WIDTH)
      ) max_reg (
          .clk(clk),
          .rst(rst),
          .data_in(result),
          .data_in_valid(element_handshake[i].element_input_valid[c]),
          .data_in_ready(element_handshake[i].element_input_ready[c]),
          .data_out(vars[i+1].data[c]),
          .data_out_valid(element_handshake[i].element_output_valid[c]),
          .data_out_ready(element_handshake[i].element_output_ready[c])
      );
    end

    // Join handshake signals from each skid buffer into a single 
    // handshake interface to drive the next level
    split_n #(
        .N(2 ** (LEVELS - i - 1))
    ) handshake_split (
        .data_in_valid (vars[i].valid),
        .data_in_ready (vars[i].ready),
        .data_out_valid(element_handshake[i].element_input_valid),
        .data_out_ready(element_handshake[i].element_input_ready)
    );

    join_n #(
        .NUM_HANDSHAKES(2 ** (LEVELS - i - 1))
    ) handshake_join (
        .data_in_valid (element_handshake[i].element_output_valid),
        .data_in_ready (element_handshake[i].element_output_ready),
        .data_out_valid(vars[i+1].valid),
        .data_out_ready(vars[i+1].ready)
    );
  end

  // Connect up first and last layer wires
  assign vars[0].data = in_data;
  assign vars[0].valid = in_valid;
  assign in_ready = vars[0].ready;

  assign out_data = vars[LEVELS].data[0];
  assign out_valid = vars[LEVELS].valid;
  assign vars[LEVELS].ready = out_ready;

endmodule
