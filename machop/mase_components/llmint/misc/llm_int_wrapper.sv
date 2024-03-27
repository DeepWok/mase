`timescale 1ns / 1ps


module top(
    // Testbench signals
    input logic sys_clk,
    input logic sys_rst,
    input data_in_valid,
    output data_in_ready,
    input weight_valid,
    output weight_ready,
    input data_out_ready,
    output data_out_valid,
    input [31:0] data_in[3:0],
    output signed [31:0] data_out[3:0]
);



(* DONT_TOUCH = "yes" *) wire signed [31:0] weights[31:0];


genvar i;
generate
    for (i = 0; i < 32; i = i + 1) begin : gen_weights
        // Assuming you want each weight to have a unique, fixed value
        // This approach assigns a constant value to each wire in the array
        assign weights[i] = i + 1; // Example of direct assignment
    end
endgenerate


// Instantiate the LLMint module
LLMint #(
    .ORIGINAL_PRECISION(16),
    .REDUCED_PRECISION(8),
    .TENSOR_SIZE_DIM(16),
    .WEIGHT_DIM_0(16),
    .WEIGHT_DIM_1(16),
    .HIGH_SLOTS(2),
    .THRESHOLD(6)
) llm_int (
    .clk(sys_clk),
    .rst(!sys_rst),
    .data_in_valid(data_in_valid),
    .data_in_ready(data_in_ready),
    .weight_valid(weight_valid),
    .weight_ready(weight_ready),
    .data_out_ready(data_out_ready),
    .data_out_valid(data_out_valid),
    .data_in(data_in),
    .weights(weights),
    .data_out(data_out)
);

// Here you would add logic to drive and monitor these signals,
// such as testbench stimulus or actual logic that would use this module in a larger design.

endmodule
