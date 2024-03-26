`timescale 1ns / 1ps

module tb_scatter_threshold;

// Parameters
localparam PRECISION = 8;
localparam TENSOR_SIZE_DIM = 4;
localparam HIGH_SLOTS = 2;
localparam THRESHOLD = 6;

// Testbench Signals
reg clk, rst;
reg [PRECISION-1:0] data_in [TENSOR_SIZE_DIM-1:0];
wire [PRECISION-1:0] o_high_precision [TENSOR_SIZE_DIM-1:0];
wire [PRECISION-1:0] o_low_precision [TENSOR_SIZE_DIM-1:0];

// Instantiate the Unit Under Test (UUT)
scatter_threshold #(
    .PRECISION(PRECISION),
    .TENSOR_SIZE_DIM(TENSOR_SIZE_DIM),
    .HIGH_SLOTS(HIGH_SLOTS),
    .THRESHOLD(THRESHOLD)
) uut (
    .clk(clk),
    .rst(rst),
    .data_in(data_in),
    .o_high_precision(o_high_precision),
    .o_low_precision(o_low_precision)
);

initial begin
    // Initialize Inputs
    clk = 0;
    rst = 1;
    // Reset the system
    #10;
    rst = 0;

    // Apply input stimuli
    #10;
    data_in[0] = 8;
    data_in[1] = 5;
    data_in[2] = 7;
    data_in[3] = 1;

    #10; // Wait for the logic to process

    // Check outputs
    // Assertions or conditional checks can be added here to validate the outputs
    // For example:
    // assert(o_high_precision[0] == 8 && o_high_precision[2] == 7) 
    // "High precision output does not match expected values."

    // Additional test vectors can be applied and checked similarly
end

// Clock generation
always #5 clk = ~clk;

endmodule
