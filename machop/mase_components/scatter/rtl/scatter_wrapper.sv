module top #(
    parameter PRECISION = 8,                // Default precision
    parameter TENSOR_SIZE_DIM = 4,          // Default tensor size dimension
    parameter HIGH_SLOTS = 2,               // Default number of high slots
    parameter THRESHOLD = 6,
    parameter DESIGN = 1               
)(
    input logic sys_clk,                         // Clock input from FPGA
    input logic sys_rst,                         // Reset input from FPGA
    input wire [PRECISION-1:0] fpga_data_in [TENSOR_SIZE_DIM-1:0], // Data input
    output wire [PRECISION-1:0] fpga_o_high_precision [TENSOR_SIZE_DIM-1:0], // High precision output
    output wire [PRECISION-1:0] fpga_o_low_precision [TENSOR_SIZE_DIM-1:0]   // Low precision output
);

    // Instantiate the scatter_threshold module with external parameters
    scatter_threshold #(
        .PRECISION(PRECISION),
        .TENSOR_SIZE_DIM(TENSOR_SIZE_DIM),
        .HIGH_SLOTS(HIGH_SLOTS),
        .THRESHOLD(THRESHOLD),
        .DESIGN(DESIGN)
    ) core_module (
        .clk(sys_clk),
        .rst(!sys_rst),
        .data_in(fpga_data_in),
        .o_high_precision(fpga_o_high_precision),
        .o_low_precision(fpga_o_low_precision)
    );

endmodule
