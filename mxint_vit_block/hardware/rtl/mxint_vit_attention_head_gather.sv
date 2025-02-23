`timescale 1ns / 1ps
module mxint_vit_attention_head_gather #(
    parameter NUM_HEADS = 12,

    parameter IN_DATA_TENSOR_SIZE_DIM_0 = 64,
    parameter IN_DATA_TENSOR_SIZE_DIM_1 = 32,
    parameter IN_DATA_PARALLELISM_DIM_0 = 4,
    parameter IN_DATA_PARALLELISM_DIM_1 = 4,
    parameter MAN_WIDTH = 16,
    parameter EXP_WIDTH = 3

) (
    input logic clk,
    input logic rst,

    input  logic [MAN_WIDTH-1:0] msplit_head_out [NUM_HEADS-1:0] [IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_1-1:0],
    input  logic [EXP_WIDTH-1:0] esplit_head_out [NUM_HEADS-1:0],
    input logic [NUM_HEADS-1:0] split_head_out_valid,
    output logic [NUM_HEADS-1:0] split_head_out_ready,

    output logic [MAN_WIDTH-1:0] mupdated_tokens [IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_1-1:0],
    output logic [EXP_WIDTH-1:0] eupdated_tokens,
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
  for (genvar head = 0; head < NUM_HEADS; head++) begin
    always_ff @(posedge clk) begin
      if (rst) begin
        block_counter[head] <= '0;
      end else if (split_head_out_valid[head] & split_head_out_ready[head]) begin
        if (block_counter[head] != BLOCKS_PER_HEAD) begin
          block_counter[head] <= block_counter[head] + 1'b1;
        end else begin
          block_counter[head] <= 1'b1;
        end
      end else if (heads_flushed == '1) begin
        block_counter[head] <= '0;
      end
    end

    assign heads_flushed[head] = (block_counter[head] == BLOCKS_PER_HEAD);
  end

  // * Find index of first head that hasn't finished
  find_first_arbiter #(
      .NUM_REQUESTERS(NUM_HEADS)
  ) ff_arb_i (
      .request  (~heads_flushed),
      .grant_oh (),
      .grant_bin(head_flushing_idx)
  );

  // * Drive output interfaces with mantissa and exponent
  assign mupdated_tokens = msplit_head_out[head_flushing_idx];
  assign eupdated_tokens = esplit_head_out[head_flushing_idx];
  assign updated_tokens_valid = split_head_out_valid[head_flushing_idx];
  
  for (genvar head = 0; head < NUM_HEADS; head++) begin
    always_comb begin
      split_head_out_ready[head] = updated_tokens_ready && (head_flushing_idx == head);
    end
  end

endmodule
