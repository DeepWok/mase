`timescale 1ns / 1ps

module gqa_head_scatter_control #(
    parameter NUM_HEADS  = 12,
    parameter GROUP_SIZE = 4,

    // Example:
    // NUM_HEADS=12 with GROUP_SIZE=4 means that there are 3 groups (NUM_GROUPS)
    // of 4 heads that share the same weights for K & V

    // * Queries, keys and values are assumed to have the same
    // * precision, dimensions and parallelism
    parameter IN_DATA_TENSOR_SIZE_DIM_0 = 64,
    parameter IN_DATA_TENSOR_SIZE_DIM_1 = 32,
    parameter IN_DATA_PARALLELISM_DIM_0 = 4,
    parameter IN_DATA_PARALLELISM_DIM_1 = 4,

    // Widths
    parameter IN_DATA_PRECISION_0 = 16
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

  // -----
  // Params
  // -----

  localparam NUM_GROUPS = NUM_HEADS / GROUP_SIZE;

  localparam GROUPED_TENSOR_SIZE_DIM_0 = IN_DATA_TENSOR_SIZE_DIM_0 / GROUP_SIZE;
  localparam GROUPED_TENSOR_SIZE_DIM_1 = IN_DATA_TENSOR_SIZE_DIM_1;

  localparam HEAD_TENSOR_SIZE_DIM_0 = IN_DATA_TENSOR_SIZE_DIM_0 / NUM_HEADS;

  localparam GROUPED_DEPTH_DIM_0 = HEAD_TENSOR_SIZE_DIM_0 / IN_DATA_PARALLELISM_DIM_0;
  localparam GROUPED_DEPTH_DIM_1 = GROUPED_TENSOR_SIZE_DIM_1 / IN_DATA_PARALLELISM_DIM_1;

  // Number of packets per head
  localparam HEAD_NUM_PACKETS = GROUPED_DEPTH_DIM_0 * GROUPED_DEPTH_DIM_1;
  localparam NUM_PACKETS_CTR_WIDTH = HEAD_NUM_PACKETS == 1 ? 1 : $clog2(HEAD_NUM_PACKETS);

  // Group counter
  localparam GROUP_CTR_WIDTH = NUM_GROUPS == 1 ? 1 : $clog2(NUM_GROUPS);

  initial begin
    // Divisibility checks
    assert (NUM_GROUPS * GROUP_SIZE == NUM_HEADS);
    assert (GROUP_SIZE * GROUPED_TENSOR_SIZE_DIM_0 == IN_DATA_TENSOR_SIZE_DIM_0);
    assert (NUM_HEADS * HEAD_TENSOR_SIZE_DIM_0 == IN_DATA_TENSOR_SIZE_DIM_0);
    assert (IN_DATA_PARALLELISM_DIM_0 * GROUPED_DEPTH_DIM_0 == HEAD_TENSOR_SIZE_DIM_0);
    assert (IN_DATA_PARALLELISM_DIM_1 * GROUPED_DEPTH_DIM_1 == GROUPED_TENSOR_SIZE_DIM_1);
  end

  // -----
  // Wires
  // -----

  logic [NUM_GROUPS-1:0] grouped_key_valid;
  logic [NUM_GROUPS-1:0] grouped_key_ready;

  logic [NUM_GROUPS-1:0] grouped_value_valid;
  logic [NUM_GROUPS-1:0] grouped_value_ready;

  // -----
  // State
  // -----

  typedef struct packed {
    logic [NUM_PACKETS_CTR_WIDTH-1:0] k_block_count;
    logic [GROUP_CTR_WIDTH-1:0] k_group_count;
  } self_t;

  self_t self, next_self;

  // -----
  // Modules
  // -----

  // Instantiate Q scatter

  self_attention_head_single_scatter #(
      .NUM_HEADS                (NUM_HEADS),
      .IN_DATA_TENSOR_SIZE_DIM_0(IN_DATA_TENSOR_SIZE_DIM_0),
      .IN_DATA_TENSOR_SIZE_DIM_1(IN_DATA_TENSOR_SIZE_DIM_1),
      .IN_DATA_PARALLELISM_DIM_0(IN_DATA_PARALLELISM_DIM_0),
      .IN_DATA_PARALLELISM_DIM_1(IN_DATA_PARALLELISM_DIM_1)
  ) q_scatter (
      .clk      (clk),
      .rst      (rst),
      .in_valid (query_valid),
      .in_ready (query_ready),
      .out_valid(split_query_valid),
      .out_ready(split_query_ready)
  );

  // K Scatter
  // The key is already transposed and in correct format so all we do is count
  // input signal

  // Handshake signals
  for (genvar i = 0; i < NUM_GROUPS; i++) begin
    assign grouped_key_valid[i] = (self.k_group_count == i) && key_valid;
  end
  assign key_ready = grouped_key_ready[self.k_group_count];

  // Counter logic
  always_comb begin
    next_self = self;

    if (key_valid && key_ready) begin
      if ((self.k_group_count == NUM_GROUPS-1) &&
            (self.k_block_count == HEAD_NUM_PACKETS-1)) begin
        next_self.k_block_count = 0;
        next_self.k_group_count = 0;
      end else if (self.k_block_count == HEAD_NUM_PACKETS - 1) begin
        next_self.k_block_count = 0;
        next_self.k_group_count = self.k_group_count + 1;
      end else begin
        next_self.k_block_count = self.k_block_count + 1;
      end
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      self <= '{default: '0};
    end else begin
      self <= next_self;
    end
  end

  // Instantiate V scatter

  // We only instantiate NUM_GROUPS of outputs for K & V since we will duplicate
  // each output GROUP_SIZE times.

  self_attention_head_single_scatter #(
      .NUM_HEADS                (NUM_GROUPS),
      .IN_DATA_TENSOR_SIZE_DIM_0(GROUPED_TENSOR_SIZE_DIM_0),
      .IN_DATA_TENSOR_SIZE_DIM_1(GROUPED_TENSOR_SIZE_DIM_1),
      .IN_DATA_PARALLELISM_DIM_0(IN_DATA_PARALLELISM_DIM_0),
      .IN_DATA_PARALLELISM_DIM_1(IN_DATA_PARALLELISM_DIM_1)
  ) v_scatter (
      .clk      (clk),
      .rst      (rst),
      .in_valid (value_valid),
      .in_ready (value_ready),
      .out_valid(grouped_value_valid),
      .out_ready(grouped_value_ready)
  );


  // Pipeline split single group handshake signal across entire group
  // Not just simple duplication since heads operate at different times.
  // Note that data is also buffered in separate downstream fifos in the main module.

  for (genvar i = 0; i < NUM_GROUPS; i++) begin : pipeline_splits
    split_n #(
        .N(GROUP_SIZE)
    ) split_key_inst (
        .data_in_valid (grouped_key_valid[i]),
        .data_in_ready (grouped_key_ready[i]),
        .data_out_valid(split_key_valid[(i+1)*GROUP_SIZE-1 : i*GROUP_SIZE]),
        .data_out_ready(split_key_ready[(i+1)*GROUP_SIZE-1 : i*GROUP_SIZE])
    );
    split_n #(
        .N(GROUP_SIZE)
    ) split_value_inst (
        .data_in_valid (grouped_value_valid[i]),
        .data_in_ready (grouped_value_ready[i]),
        .data_out_valid(split_value_valid[(i+1)*GROUP_SIZE-1 : i*GROUP_SIZE]),
        .data_out_ready(split_value_ready[(i+1)*GROUP_SIZE-1 : i*GROUP_SIZE])
    );
  end


endmodule
