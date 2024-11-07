module unpacked_split2_with_data #(
    parameter DEPTH = 8,
    parameter DATA_WIDTH = 8,
    parameter IN_SIZE = 8
) (
    input clk,
    input rst,
    // Input interface
    input [DATA_WIDTH-1:0] data_in[IN_SIZE - 1:0],
    input logic data_in_valid,
    output logic data_in_ready,
    // FIFO output interface
    output [DATA_WIDTH-1:0] fifo_data_out[IN_SIZE - 1:0],
    output logic fifo_data_out_valid,
    input logic fifo_data_out_ready,
    // Straight output interface
    output [DATA_WIDTH-1:0] straight_data_out[IN_SIZE - 1:0],
    output logic straight_data_out_valid,
    input logic straight_data_out_ready
);
    // Flatten the input data
    logic [DATA_WIDTH * IN_SIZE - 1:0] data_in_flatten;
    logic [DATA_WIDTH * IN_SIZE - 1:0] fifo_data_out_flatten;
    logic [DATA_WIDTH * IN_SIZE - 1:0] straight_data_out_flatten;
    
    // Input flattening
    for (genvar i = 0; i < IN_SIZE; i++) begin : reshape
        assign data_in_flatten[i*DATA_WIDTH+DATA_WIDTH-1:i*DATA_WIDTH] = data_in[i];
    end
    
    // Split2 instance
    split2_with_data #(
        .DATA_WIDTH(DATA_WIDTH * IN_SIZE),
        .FIFO_DEPTH(DEPTH)
    ) split2_with_data_i (
        .clk(clk),
        .rst(rst),
        .data_in(data_in_flatten),
        .data_in_valid(data_in_valid),
        .data_in_ready(data_in_ready),
        .fifo_data_out(fifo_data_out_flatten),
        .fifo_data_out_valid(fifo_data_out_valid),
        .fifo_data_out_ready(fifo_data_out_ready),
        .straight_data_out(straight_data_out_flatten),
        .straight_data_out_valid(straight_data_out_valid),
        .straight_data_out_ready(straight_data_out_ready)
    );
    
    // Unflatten FIFO output
    for (genvar i = 0; i < IN_SIZE; i++) begin : unreshape_fifo
        assign fifo_data_out[i] = fifo_data_out_flatten[i*DATA_WIDTH+DATA_WIDTH-1:i*DATA_WIDTH];
    end

    // Unflatten straight output
    for (genvar i = 0; i < IN_SIZE; i++) begin : unreshape_straight
        assign straight_data_out[i] = straight_data_out_flatten[i*DATA_WIDTH+DATA_WIDTH-1:i*DATA_WIDTH];
    end

endmodule