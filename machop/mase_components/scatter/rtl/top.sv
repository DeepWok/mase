module top (
    input sys_clk,                         // Clock input from FPGA
    input logic sys_rst,                         // Reset input from FPGA
    input wire [16-1:0] fpga_data_in [16-1:0], // Data input
    output wire [16-1:0] fpga_o_high_precision [16-1:0], // High precision output
    output wire [16-1:0] fpga_o_low_precision [16-1:0]   // Low precision output
);

    wrapper #(
        .PRECISION(16),
        .TENSOR_SIZE_DIM(16),
        .HIGH_SLOTS(2),
        .THRESHOLD(6),
        .DESIGN(1)
    ) core_module (
        .clk(sys_clk),
        .rst(!sys_rst),
        .fpga_data_in(fpga_data_in),
        .fpga_o_high_precision(fpga_o_high_precision),
        .fpga_o_low_precision(fpga_o_low_precision)
    );

endmodule