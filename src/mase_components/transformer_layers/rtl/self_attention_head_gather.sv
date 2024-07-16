`timescale 1ns / 1ps
module self_attention_head_gather #(
    parameter NUM_HEADS = 12,

    // * Queries, keys and values are assumed to have the same
    // * precision, dimensions and parallelism
    parameter IN_DATA_TENSOR_SIZE_DIM_0 = 64,
    parameter IN_DATA_TENSOR_SIZE_DIM_1 = 32,
    parameter IN_DATA_PARALLELISM_DIM_0 = 4,
    parameter IN_DATA_PARALLELISM_DIM_1 = 4,
    parameter IN_DATA_PRECISION_0 = 16,
    parameter IN_DATA_PRECISION_1 = 3

) (
    input logic clk,
    input logic rst,

    input  logic [IN_DATA_PRECISION_0-1:0] split_head_out [NUM_HEADS-1:0] [IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_1-1:0],
    input logic [NUM_HEADS-1:0] split_head_out_valid,
    output logic [NUM_HEADS-1:0] split_head_out_ready,

    output logic [IN_DATA_PRECISION_0-1:0] updated_tokens [IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_1-1:0],
    output logic updated_tokens_valid,
    input logic updated_tokens_ready
);

  parameter IN_DATA_DEPTH = IN_DATA_TENSOR_SIZE_DIM_0 / IN_DATA_PARALLELISM_DIM_0;
  parameter BLOCKS_PER_HEAD = IN_DATA_DEPTH / NUM_HEADS;

  // Block counters
  logic [NUM_HEADS-1:0][$clog2(BLOCKS_PER_HEAD):0] block_counter;
  logic [NUM_HEADS-1:0] heads_flushed;
  logic [$clog2(NUM_HEADS)-1:0] head_flushing_idx;

  // * Count the number of blocks received for each head
  // * Create head_done mask e.g. 00000111111 (heads that have finished flushing contents)
  // * Invert head_done mask e.g. 11111000000 (heads that haven't yet flushed contents)
  // * Find first index gives the select signal to drive the output interface

  for (genvar head = 0; head < NUM_HEADS; head++) begin
    always_ff @(posedge clk) begin
      if (rst) begin
        block_counter[head] <= '0;

        // * Increment block counter when accepting a block for a given head
        // * But saturate at BLOCKS_PER_HEAD
      end else if (split_head_out_valid[head] & split_head_out_ready[head]) begin
        block_counter [head] <= (block_counter == BLOCKS_PER_HEAD - 1) ? BLOCKS_PER_HEAD : block_counter[head] + 1'b1;

        // * Reset counter when all heads done
      end else if (heads_flushed == '1) begin
        block_counter[head] <= '0;
      end
    end

    // * Create mask of heads with block count saturated at BLOCKS_PER_HEAD
    // * (i.e. finished heads)
    assign heads_flushed[head] = (block_counter[head] == BLOCKS_PER_HEAD);
  end

  // * Find index of first (least significant) head that hasn't yet
  // * finished dumping all its blocks
  find_first_arbiter #(
      .NUM_REQUESTERS(NUM_HEADS)
  ) ff_arb_i (
      .request  (~heads_flushed),
      .grant_oh (),
      .grant_bin(head_flushing_idx)
  );

  // * Drive output handshake interface

  assign updated_tokens = split_head_out[head_flushing_idx];
  assign updated_tokens_valid = split_head_out_valid[head_flushing_idx];
  for (genvar head = 0; head < NUM_HEADS; head++) begin
    always_comb begin
      split_head_out_ready[head] = updated_tokens_ready && (head_flushing_idx == head);
    end
  end

endmodule
