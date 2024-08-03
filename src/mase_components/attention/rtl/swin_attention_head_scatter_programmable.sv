`timescale 1ns / 1ps
module swin_attention_head_scatter_programmable #(
    parameter NUM_HEADS = 12,

    // * Queries, keys and values are assumed to have the same
    // * precision, dimensions and parallelism
    parameter IN_DATA_MAX_TENSOR_SIZE_DIM_0 = 64,
    parameter IN_DATA_MAX_TENSOR_SIZE_DIM_1 = 32,
    parameter IN_DATA_PARALLELISM_DIM_0 = 4,
    parameter IN_DATA_PARALLELISM_DIM_1 = 4,
    localparam IN_DATA_MAX_DEPTH_DIM_0 = IN_DATA_MAX_TENSOR_SIZE_DIM_0/IN_DATA_PARALLELISM_DIM_0,
    localparam IN_DATA_MAX_BLOCK_PER_HEAD = IN_DATA_MAX_DEPTH_DIM_0/NUM_HEADS,
    localparam IN_DATA_MAX_BLOCK_PER_HEAD_WIDTH = $clog2(IN_DATA_MAX_BLOCK_PER_HEAD)
) (
    input logic clk,
    input logic rst,

    input logic [IN_DATA_MAX_BLOCK_PER_HEAD_WIDTH:0] block_per_head,

    input logic query_con_valid,
    output logic query_con_ready,

    input logic query_pos_valid,
    output logic query_pos_ready,

    input  logic key_valid,
    output logic key_ready,

    input  logic value_valid,
    output logic value_ready,

    input  logic pos_embed_valid,
    output logic pos_embed_ready,

    output logic [NUM_HEADS-1:0] split_query_con_valid,
    input  logic [NUM_HEADS-1:0] split_query_con_ready,

    output logic [NUM_HEADS-1:0] split_query_pos_valid,
    input  logic [NUM_HEADS-1:0] split_query_pos_ready,

    output logic [NUM_HEADS-1:0] split_key_valid,
    input  logic [NUM_HEADS-1:0] split_key_ready,

    output logic [NUM_HEADS-1:0] split_value_valid,
    input  logic [NUM_HEADS-1:0] split_value_ready,

    output logic [NUM_HEADS-1:0] split_pos_embed_valid,
    input  logic [NUM_HEADS-1:0] split_pos_embed_ready
);


  // Block counters
  logic [IN_DATA_MAX_BLOCK_PER_HEAD_WIDTH:0] pos_embed_block_cnt;
  logic [IN_DATA_MAX_BLOCK_PER_HEAD_WIDTH:0] query_con_block_cnt;
  logic [IN_DATA_MAX_BLOCK_PER_HEAD_WIDTH:0] query_pos_block_cnt;
  logic [IN_DATA_MAX_BLOCK_PER_HEAD_WIDTH:0] key_block_cnt;
  logic [IN_DATA_MAX_BLOCK_PER_HEAD_WIDTH:0] value_block_cnt;

  // Head counters
  logic [IN_DATA_MAX_BLOCK_PER_HEAD_WIDTH:0] pos_embed_head_cnt;
  logic [IN_DATA_MAX_BLOCK_PER_HEAD_WIDTH:0] query_con_head_cnt;
  logic [IN_DATA_MAX_BLOCK_PER_HEAD_WIDTH:0] query_pos_head_cnt;
  logic [$clog2(NUM_HEADS):0] key_head_cnt;
  logic [$clog2(NUM_HEADS):0] value_head_cnt;

  // * Increment block and head counters

  always_ff @(posedge clk) begin
    if (rst) begin
      pos_embed_block_cnt <= '0;
      query_con_block_cnt <= '0;
      query_pos_block_cnt <= '0;
      key_block_cnt <= '0;
      value_block_cnt <= '0;

      pos_embed_head_cnt <= '0;
      query_con_head_cnt <= '0;
      query_pos_head_cnt <= '0;
      key_head_cnt <= '0;
      value_head_cnt <= '0;
    end else begin



      // Increment query content counter
      if (pos_embed_valid && pos_embed_ready) begin
        pos_embed_block_cnt <= (pos_embed_block_cnt == block_per_head - 1) ? '0 : pos_embed_block_cnt + 1'b1;

        if (pos_embed_block_cnt == block_per_head - 1) begin
          pos_embed_head_cnt <= (pos_embed_head_cnt == NUM_HEADS - 1) ? '0 : pos_embed_head_cnt + 1'b1;
        end
      end

      // Increment query content counter
      if (query_con_valid && query_con_ready) begin
        query_con_block_cnt <= (query_con_block_cnt == block_per_head - 1) ? '0 : query_con_block_cnt + 1'b1;

        if (query_con_block_cnt == block_per_head - 1) begin
          query_con_head_cnt <= (query_con_head_cnt == NUM_HEADS - 1) ? '0 : query_con_head_cnt + 1'b1;
        end
      end

      // Increment query content counter
      if (query_pos_valid && query_pos_ready) begin
        query_pos_block_cnt <= (query_pos_block_cnt == block_per_head - 1) ? '0 : query_pos_block_cnt + 1'b1;

        if (query_pos_block_cnt == block_per_head - 1) begin
          query_pos_head_cnt <= (query_pos_head_cnt == NUM_HEADS - 1) ? '0 : query_pos_head_cnt + 1'b1;
        end
      end

      // Increment key counter
      if (key_valid && key_ready) begin
        key_block_cnt <= (key_block_cnt == block_per_head - 1) ? '0 : key_block_cnt + 1'b1;

        if (key_block_cnt == block_per_head - 1) begin
          key_head_cnt <= (key_head_cnt == NUM_HEADS - 1) ? '0 : key_head_cnt + 1'b1;
        end
      end

      // Increment query counter
      if (value_valid && value_ready) begin
        value_block_cnt <= (value_block_cnt == block_per_head - 1) ? '0 : value_block_cnt + 1'b1;

        if (value_block_cnt == block_per_head - 1) begin
          value_head_cnt <= (value_head_cnt == NUM_HEADS - 1) ? '0 : value_head_cnt + 1'b1;
        end
      end

    end
  end

  // * Drive split QKV handshake interface

  for (genvar head = 0; head < NUM_HEADS; head++) begin
    always_comb begin
      split_pos_embed_valid[head] = pos_embed_valid && (pos_embed_head_cnt == head);
      split_query_con_valid[head] = query_con_valid && (query_con_head_cnt == head);
      split_query_pos_valid[head] = query_pos_valid && (query_pos_head_cnt == head);
      split_key_valid[head]   = key_valid && (key_head_cnt == head);
      split_value_valid[head] = value_valid && (value_head_cnt == head);
    end
  end

  always_comb begin
    pos_embed_ready = split_pos_embed_ready[pos_embed_head_cnt];
    query_con_ready = split_query_con_ready[query_con_head_cnt];
    query_pos_ready = split_query_pos_ready[query_pos_head_cnt];
    key_ready   = split_key_ready[key_head_cnt];
    value_ready = split_value_ready[value_head_cnt];
  end



endmodule
