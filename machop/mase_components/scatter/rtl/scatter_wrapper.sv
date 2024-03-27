module top #(
    parameter PRECISION = 16,                // Default precision
    parameter TENSOR_SIZE_DIM = 16,          // Default tensor size dimension
    parameter HIGH_SLOTS = 2,               // Default number of high slots
    parameter THRESHOLD = 6,
    parameter DESIGN = 1               
)(
    input clk,                         // Clock input from FPGA
    input rst
);


    (* dont_touch="yes" *)  wire [PRECISION-1:0] data_in [TENSOR_SIZE_DIM-1:0]; // Data input
    (* dont_touch="yes" *) wire [PRECISION-1:0] o_high_precision [TENSOR_SIZE_DIM-1:0]; // High precision output
    (* dont_touch="yes" *) wire [PRECISION-1:0] o_low_precision [TENSOR_SIZE_DIM-1:0];  
    // Instantiate the scatter_threshold module with external parameters
    scatter_threshold #(
        .PRECISION(PRECISION),
        .TENSOR_SIZE_DIM(TENSOR_SIZE_DIM),
        .HIGH_SLOTS(HIGH_SLOTS),
        .THRESHOLD(THRESHOLD),
        .DESIGN(DESIGN)
    ) core_module (
        .clk(clk),
        .rst(!rst),
        .data_in(data_in),
        .o_high_precision(o_high_precision),
        .o_low_precision(o_low_precision)
    );

endmodule
