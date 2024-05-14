`timescale 1ns / 1ps

module gqa_head_scatter_control #(
    parameter NUM_HEADS = 12,
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

initial begin
    // Divisibility checks
    assert (NUM_GROUPS * GROUP_SIZE == NUM_HEADS);
end

// -----
// Wires
// -----

logic [NUM_GROUPS-1:0] grouped_key_valid;
logic [NUM_GROUPS-1:0] grouped_key_ready;

logic [NUM_GROUPS-1:0] grouped_value_valid;
logic [NUM_GROUPS-1:0] grouped_value_ready;

// -----
// Modules
// -----

// Instantiate Q scatter

self_attention_head_single_scatter #(
    .NUM_HEADS                  (NUM_HEADS),
    .IN_DATA_TENSOR_SIZE_DIM_0  (IN_DATA_TENSOR_SIZE_DIM_0),
    .IN_DATA_TENSOR_SIZE_DIM_1  (IN_DATA_TENSOR_SIZE_DIM_1),
    .IN_DATA_PARALLELISM_DIM_0  (IN_DATA_PARALLELISM_DIM_0),
    .IN_DATA_PARALLELISM_DIM_1  (IN_DATA_PARALLELISM_DIM_1)
) q_scatter (
    .clk                        (clk),
    .rst                        (rst),
    .in_valid                   (query_valid),
    .in_ready                   (query_ready),
    .out_valid                  (split_query_valid),
    .out_ready                  (split_query_ready)
);

// Instantiate Q & V scatters

// We only instantiate NUM_GROUPS of outputs for K & V since we will duplicate
// each output GROUP_SIZE times.

self_attention_head_single_scatter #(
    .NUM_HEADS                  (NUM_GROUPS),
    .IN_DATA_TENSOR_SIZE_DIM_0  (GROUPED_TENSOR_SIZE_DIM_0),
    .IN_DATA_TENSOR_SIZE_DIM_1  (GROUPED_TENSOR_SIZE_DIM_1),
    .IN_DATA_PARALLELISM_DIM_0  (IN_DATA_PARALLELISM_DIM_0),
    .IN_DATA_PARALLELISM_DIM_1  (IN_DATA_PARALLELISM_DIM_1)
) k_scatter (
    .clk                        (clk),
    .rst                        (rst),
    .in_valid                   (key_valid),
    .in_ready                   (key_ready),
    .out_valid                  (grouped_key_valid),
    .out_ready                  (grouped_key_ready)
);

self_attention_head_single_scatter #(
    .NUM_HEADS                  (NUM_GROUPS),
    .IN_DATA_TENSOR_SIZE_DIM_0  (GROUPED_TENSOR_SIZE_DIM_0),
    .IN_DATA_TENSOR_SIZE_DIM_1  (GROUPED_TENSOR_SIZE_DIM_1),
    .IN_DATA_PARALLELISM_DIM_0  (IN_DATA_PARALLELISM_DIM_0),
    .IN_DATA_PARALLELISM_DIM_1  (IN_DATA_PARALLELISM_DIM_1)
) v_scatter (
    .clk                        (clk),
    .rst                        (rst),
    .in_valid                   (value_valid),
    .in_ready                   (value_ready),
    .out_valid                  (grouped_value_valid),
    .out_ready                  (grouped_value_ready)
);


// Pipeline split single group handshake signal across entire group
// Not just simple duplication since heads operate at different times.
// Note that data is also buffered in separate downstream fifos in the main module.

for (genvar i = 0; i < NUM_GROUPS; i++) begin : pipeline_splits

    split_n #(
        .N               (GROUP_SIZE)
    ) split_key_inst (
        .data_in_valid   (grouped_key_valid[i]),
        .data_in_ready   (grouped_key_ready[i]),
        .data_out_valid  (split_key_valid[(i+1)*GROUP_SIZE-1 : i*GROUP_SIZE]),
        .data_out_ready  (split_key_ready[(i+1)*GROUP_SIZE-1 : i*GROUP_SIZE])
    );

    split_n #(
        .N               (GROUP_SIZE)
    ) split_value_inst (
        .data_in_valid   (grouped_value_valid[i]),
        .data_in_ready   (grouped_value_ready[i]),
        .data_out_valid  (split_value_valid[(i+1)*GROUP_SIZE-1 : i*GROUP_SIZE]),
        .data_out_ready  (split_value_ready[(i+1)*GROUP_SIZE-1 : i*GROUP_SIZE])
    );

end


endmodule
