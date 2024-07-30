`timescale 1ns / 1ps
module self_attention_head_single_scatter #(
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

    input  logic in_valid,
    output logic in_ready,

    output logic [NUM_HEADS-1:0] out_valid,
    input  logic [NUM_HEADS-1:0] out_ready
);

  parameter IN_DATA_DEPTH = IN_DATA_TENSOR_SIZE_DIM_0 / IN_DATA_PARALLELISM_DIM_0;
  parameter BLOCKS_PER_HEAD = IN_DATA_DEPTH / NUM_HEADS;

  initial begin
    // Check divisibility
    assert (IN_DATA_DEPTH * IN_DATA_PARALLELISM_DIM_0 == IN_DATA_TENSOR_SIZE_DIM_0);
    assert (BLOCKS_PER_HEAD * NUM_HEADS == IN_DATA_DEPTH);
  end

  // Block counters
  logic [$clog2(BLOCKS_PER_HEAD):0] block_cnt;

  // Head counters
  logic [$clog2(NUM_HEADS):0] head_cnt;

  // * Increment block and head counters

  always_ff @(posedge clk) begin
    if (rst) begin
      block_cnt <= '0;
      head_cnt  <= '0;
    end else begin
      // Increment query counter
      if (in_valid && in_ready) begin
        block_cnt <= (block_cnt == BLOCKS_PER_HEAD - 1) ? '0 : block_cnt + 1'b1;

        if (block_cnt == BLOCKS_PER_HEAD - 1) begin
          head_cnt <= (head_cnt == NUM_HEADS - 1) ? '0 : head_cnt + 1'b1;
        end
      end
    end
  end

  // * Drive split handshake interface
  for (genvar head = 0; head < NUM_HEADS; head++) begin
    always_comb begin
      out_valid[head] = in_valid && (head_cnt == head);
    end
  end
  always_comb begin
    in_ready = out_ready[head_cnt];
  end

endmodule
