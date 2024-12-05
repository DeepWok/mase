`timescale 1ns / 1ps
module mxint_fork2 #(
    parameter DATA_IN_0_PRECISION_0 = 8,  // mantissa width
    parameter DATA_IN_0_PRECISION_1 = 8,  // exponent width
    
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 20,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 20,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 20,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 20,
    parameter DATA_IN_0_PARALLELISM_DIM_2 = 1,

    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0,
    parameter DATA_OUT_0_PRECISION_1 = DATA_IN_0_PRECISION_1,
    parameter DATA_OUT_1_PRECISION_0 = DATA_IN_0_PRECISION_0,
    parameter DATA_OUT_1_PRECISION_1 = DATA_IN_0_PRECISION_1,

    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_2 = DATA_IN_0_TENSOR_SIZE_DIM_2,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_2 = DATA_IN_0_PARALLELISM_DIM_2,

    parameter DATA_OUT_1_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter DATA_OUT_1_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_1_TENSOR_SIZE_DIM_2 = DATA_IN_0_TENSOR_SIZE_DIM_2,
    parameter DATA_OUT_1_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,
    parameter DATA_OUT_1_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,
    parameter DATA_OUT_1_PARALLELISM_DIM_2 = DATA_IN_0_PARALLELISM_DIM_2,
    localparam FIFO_DEPTH = DATA_OUT_0_TENSOR_SIZE_DIM_0 * DATA_OUT_0_TENSOR_SIZE_DIM_1 / (DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1),

    localparam BLOCK_SIZE = DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1
) (
    input wire clk,
    input wire rst,
    
    // Input interface
    input logic [DATA_IN_0_PRECISION_0-1:0] mdata_in_0[BLOCK_SIZE-1:0],
    input logic [DATA_IN_0_PRECISION_1-1:0] edata_in_0,
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    // FIFO output interface (output 0)
    output logic [DATA_OUT_0_PRECISION_0-1:0] mdata_out_0[BLOCK_SIZE-1:0],
    output logic [DATA_OUT_0_PRECISION_1-1:0] edata_out_0,
    output logic data_out_0_valid,
    input logic data_out_0_ready,

    // Straight output interface (output 1)
    output logic [DATA_OUT_1_PRECISION_0-1:0] mdata_out_1[BLOCK_SIZE-1:0],
    output logic [DATA_OUT_1_PRECISION_1-1:0] edata_out_1,
    output logic data_out_1_valid,
    input logic data_out_1_ready
);

    // Flatten the input data
    logic [DATA_IN_0_PRECISION_0 * BLOCK_SIZE + DATA_IN_0_PRECISION_1 - 1:0] data_in_flatten;
    logic [DATA_IN_0_PRECISION_0 * BLOCK_SIZE + DATA_IN_0_PRECISION_1 - 1:0] fifo_data_out_flatten;
    logic [DATA_IN_0_PRECISION_0 * BLOCK_SIZE + DATA_IN_0_PRECISION_1 - 1:0] straight_data_out_flatten;

    // Input flattening
    for (genvar i = 0; i < BLOCK_SIZE; i++) begin : reshape
        assign data_in_flatten[i*DATA_IN_0_PRECISION_0 +: DATA_IN_0_PRECISION_0] = mdata_in_0[i];
    end
    assign data_in_flatten[DATA_IN_0_PRECISION_0*BLOCK_SIZE +: DATA_IN_0_PRECISION_1] = edata_in_0;

    // Split2 instance
    split2_with_data #(
        .DATA_WIDTH(DATA_IN_0_PRECISION_0 * BLOCK_SIZE + DATA_IN_0_PRECISION_1),
        .FIFO_DEPTH(FIFO_DEPTH)
    ) split2_with_data_i (
        .clk(clk),
        .rst(rst),
        .data_in(data_in_flatten),
        .data_in_valid(data_in_0_valid),
        .data_in_ready(data_in_0_ready),
        .fifo_data_out(fifo_data_out_flatten),
        .fifo_data_out_valid(data_out_0_valid),
        .fifo_data_out_ready(data_out_0_ready),
        .straight_data_out(straight_data_out_flatten),
        .straight_data_out_valid(data_out_1_valid),
        .straight_data_out_ready(data_out_1_ready)
    );

    // Unflatten FIFO output (output 0)
    for (genvar i = 0; i < BLOCK_SIZE; i++) begin : unreshape_fifo
        assign mdata_out_0[i] = fifo_data_out_flatten[i*DATA_OUT_0_PRECISION_0 +: DATA_OUT_0_PRECISION_0];
    end
    assign edata_out_0 = fifo_data_out_flatten[DATA_OUT_0_PRECISION_0*BLOCK_SIZE +: DATA_OUT_0_PRECISION_1];

    // Unflatten straight output (output 1)
    for (genvar i = 0; i < BLOCK_SIZE; i++) begin : unreshape_straight
        assign mdata_out_1[i] = straight_data_out_flatten[i*DATA_OUT_1_PRECISION_0 +: DATA_OUT_1_PRECISION_0];
    end
    assign edata_out_1 = straight_data_out_flatten[DATA_OUT_1_PRECISION_0*BLOCK_SIZE +: DATA_OUT_1_PRECISION_1];

endmodule
