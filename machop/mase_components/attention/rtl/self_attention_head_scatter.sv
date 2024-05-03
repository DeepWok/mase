module self_attention_head_scatter #(
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

    input logic [IN_DATA_PRECISION_0-1:0] query [IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_1-1:0],
    input logic query_valid,
    output logic query_ready,

    input logic [IN_DATA_PRECISION_0-1:0] key [IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_1-1:0],
    input logic key_valid,
    output logic key_ready,

    input logic [IN_DATA_PRECISION_0-1:0] value [IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_1-1:0],
    input logic value_valid,
    output logic value_ready,

    output logic [IN_DATA_PRECISION_0-1:0] split_query [NUM_HEADS-1:0] [IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_1-1:0],
    output logic [NUM_HEADS-1:0] split_query_valid,
    input  logic [NUM_HEADS-1:0] split_query_ready,

    output logic [IN_DATA_PRECISION_0-1:0] split_key [NUM_HEADS-1:0] [IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_1-1:0],
    output logic [NUM_HEADS-1:0] split_key_valid,
    input  logic [NUM_HEADS-1:0] split_key_ready,

    output logic [IN_DATA_PRECISION_0-1:0] split_value [NUM_HEADS-1:0] [IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_1-1:0],
    output logic [NUM_HEADS-1:0] split_value_valid,
    input  logic [NUM_HEADS-1:0] split_value_ready
);

endmodule