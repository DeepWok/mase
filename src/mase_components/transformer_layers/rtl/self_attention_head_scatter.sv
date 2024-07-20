`timescale 1ns / 1ps
module self_attention_head_scatter #(
    parameter NUM_HEADS = 12,

    // * Queries, keys and values are assumed to have the same
    // * precision, dimensions and parallelism
    parameter IN_DATA_TENSOR_SIZE_DIM_0 = 64,
    parameter IN_DATA_TENSOR_SIZE_DIM_1 = 32,
    parameter IN_DATA_PARALLELISM_DIM_0 = 4,
    parameter IN_DATA_PARALLELISM_DIM_1 = 4
) (
    input logic clk,
    input logic rst,

    input  logic query_valid,
    output logic query_ready,

    input  logic key_valid,
    output logic key_ready,

    input  logic value_valid,
    output logic value_ready,

    output logic [NUM_HEADS-1:0] split_query_valid,
    input  logic [NUM_HEADS-1:0] split_query_ready,

    output logic [NUM_HEADS-1:0] split_key_valid,
    input  logic [NUM_HEADS-1:0] split_key_ready,

    output logic [NUM_HEADS-1:0] split_value_valid,
    input  logic [NUM_HEADS-1:0] split_value_ready
);

  parameter IN_DATA_DEPTH = IN_DATA_TENSOR_SIZE_DIM_0 / IN_DATA_PARALLELISM_DIM_0;
  parameter BLOCKS_PER_HEAD = IN_DATA_DEPTH / NUM_HEADS;

  // Block counters
  logic [$clog2(BLOCKS_PER_HEAD):0] query_block_cnt;
  logic [$clog2(BLOCKS_PER_HEAD):0] key_block_cnt;
  logic [$clog2(BLOCKS_PER_HEAD):0] value_block_cnt;

  // Head counters
  logic [$clog2(NUM_HEADS):0] query_head_cnt;
  logic [$clog2(NUM_HEADS):0] key_head_cnt;
  logic [$clog2(NUM_HEADS):0] value_head_cnt;

  // * Increment block and head counters

  always_ff @(posedge clk) begin
    if (rst) begin
      query_block_cnt <= '0;
      key_block_cnt <= '0;
      value_block_cnt <= '0;

      query_head_cnt <= '0;
      key_head_cnt <= '0;
      value_head_cnt <= '0;
    end else begin
      // Increment query counter
      if (query_valid && query_ready) begin
        query_block_cnt <= (query_block_cnt == BLOCKS_PER_HEAD - 1) ? '0 : query_block_cnt + 1'b1;

        if (query_block_cnt == BLOCKS_PER_HEAD - 1) begin
          query_head_cnt <= (query_head_cnt == NUM_HEADS - 1) ? '0 : query_head_cnt + 1'b1;
        end
      end

      // Increment key counter
      if (key_valid && key_ready) begin
        key_block_cnt <= (key_block_cnt == BLOCKS_PER_HEAD - 1) ? '0 : key_block_cnt + 1'b1;

        if (key_block_cnt == BLOCKS_PER_HEAD - 1) begin
          key_head_cnt <= (key_head_cnt == NUM_HEADS - 1) ? '0 : key_head_cnt + 1'b1;
        end
      end

      // Increment query counter
      if (value_valid && value_ready) begin
        value_block_cnt <= (value_block_cnt == BLOCKS_PER_HEAD - 1) ? '0 : value_block_cnt + 1'b1;

        if (value_block_cnt == BLOCKS_PER_HEAD - 1) begin
          value_head_cnt <= (value_head_cnt == NUM_HEADS - 1) ? '0 : value_head_cnt + 1'b1;
        end
      end

    end
  end

  // * Drive split QKV handshake interface

  for (genvar head = 0; head < NUM_HEADS; head++) begin
    always_comb begin
      split_query_valid[head] = query_valid && (query_head_cnt == head);
      split_key_valid[head]   = key_valid && (key_head_cnt == head);
      split_value_valid[head] = value_valid && (value_head_cnt == head);
    end
  end

  always_comb begin
    query_ready = split_query_ready[query_head_cnt];
    key_ready   = split_key_ready[key_head_cnt];
    value_ready = split_value_ready[value_head_cnt];
  end



endmodule
