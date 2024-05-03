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
    input  logic [NUM_HEADS-1:0] split_head_out_valid,
    output logic [NUM_HEADS-1:0] split_head_out_ready,

    output logic [IN_DATA_PRECISION_0-1:0] updated_tokens [IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_1-1:0],
    output logic updated_tokens_valid,
    input  logic updated_tokens_ready,
);

endmodule