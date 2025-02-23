`timescale 1 ns / 1 ps
module mxint_div #(
    parameter DATA_DIVIDEND_PRECISION_0 = 8,
    parameter DATA_DIVIDEND_PRECISION_1 = 8,
    parameter DATA_DIVISOR_PRECISION_0 = 8,
    parameter DATA_DIVISOR_PRECISION_1 = 8,
    parameter DATA_QUOTIENT_PRECISION_0 = 8,
    parameter DATA_QUOTIENT_PRECISION_1 = 8,
    parameter BLOCK_SIZE = 4,
    parameter DATA_IN_0_DIM = 8  // Add this parameter
) (
    input logic clk,
    input logic rst,
    input logic [DATA_DIVIDEND_PRECISION_0-1:0] mdividend_data[BLOCK_SIZE - 1:0],
    input logic [DATA_DIVIDEND_PRECISION_1-1:0] edividend_data,
    input logic dividend_data_valid,
    output logic dividend_data_ready,
    input logic [DATA_DIVISOR_PRECISION_0-1:0] mdivisor_data[BLOCK_SIZE - 1:0],
    input logic [DATA_DIVISOR_PRECISION_1-1:0] edivisor_data,
    input logic divisor_data_valid,
    output logic divisor_data_ready,
    output logic [DATA_QUOTIENT_PRECISION_0-1:0] mquotient_data[BLOCK_SIZE - 1:0],
    output logic [DATA_QUOTIENT_PRECISION_1-1:0] equotient_data,
    output logic quotient_data_valid,
    input logic quotient_data_ready
);
    // Signal declarations
    logic [DATA_DIVIDEND_PRECISION_0-1:0] straight_mdividend_data[BLOCK_SIZE - 1:0];
    logic straight_mdividend_data_valid;
    logic straight_mdividend_data_ready;
    
    logic [DATA_DIVISOR_PRECISION_0-1:0] straight_mdivisor_data[BLOCK_SIZE - 1:0];
    logic straight_mdivisor_data_valid;
    logic straight_mdivisor_data_ready;
    
    logic [DATA_DIVIDEND_PRECISION_1-1:0] fifo_edividend_data;
    logic fifo_edividend_data_valid;
    logic fifo_edividend_data_ready;
    
    logic [DATA_DIVISOR_PRECISION_1-1:0] fifo_edivisor_data;
    logic fifo_edivisor_data_valid;
    logic fifo_edivisor_data_ready;
    
    logic mquotient_data_valid;
    logic mquotient_data_ready;

    // First split2 instance (for dividend)
    unpacked_mx_split2_with_data #(
        .DEPTH(DATA_IN_0_DIM),
        .MAN_WIDTH(DATA_DIVIDEND_PRECISION_0),
        .EXP_WIDTH(DATA_DIVIDEND_PRECISION_1),
        .IN_SIZE(BLOCK_SIZE)
    ) split2_dividend (  // Renamed instance
        .clk(clk),
        .rst(rst),
        // Input from circular buffer
        .mdata_in(mdividend_data),
        .edata_in(edividend_data),
        .data_in_valid(dividend_data_valid),
        .data_in_ready(dividend_data_ready),
        .fifo_mdata_out(),
        .fifo_edata_out(fifo_edividend_data),
        .fifo_data_out_valid(fifo_edividend_data_valid),
        .fifo_data_out_ready(fifo_edividend_data_ready),
        // Straight output path
        .straight_mdata_out(straight_mdividend_data),  // Connect to the same signals previously used
        .straight_edata_out(),
        .straight_data_out_valid(straight_mdividend_data_valid),
        .straight_data_out_ready(straight_mdividend_data_ready)
    );

    // Second split2 instance (for divisor)
    unpacked_mx_split2_with_data #(
        .DEPTH(DATA_IN_0_DIM),
        .MAN_WIDTH(DATA_DIVISOR_PRECISION_0),
        .EXP_WIDTH(DATA_DIVISOR_PRECISION_1),
        .IN_SIZE(BLOCK_SIZE)
    ) split2_divisor (  // Renamed instance
        .clk(clk),
        .rst(rst),
        // Input from circular buffer
        .mdata_in(mdivisor_data),
        .edata_in(edivisor_data),
        .data_in_valid(divisor_data_valid),
        .data_in_ready(divisor_data_ready),
        .fifo_mdata_out(),
        .fifo_edata_out(fifo_edivisor_data),
        .fifo_data_out_valid(fifo_edivisor_data_valid),
        .fifo_data_out_ready(fifo_edivisor_data_ready),
        // Straight output path
        .straight_mdata_out(straight_mdivisor_data),  // Connect to the same signals previously used
        .straight_edata_out(),
        .straight_data_out_valid(straight_mdivisor_data_valid),
        .straight_data_out_ready(straight_mdivisor_data_ready)
    );
    // Integer division instance
    int_div #(
        .IN_NUM(BLOCK_SIZE),
        .DIVIDEND_WIDTH(DATA_DIVIDEND_PRECISION_0),
        .DIVISOR_WIDTH(DATA_DIVISOR_PRECISION_0),
        .QUOTIENT_WIDTH(DATA_QUOTIENT_PRECISION_0)
    ) div_inst (
        .clk(clk),
        .rst(rst),
        .dividend_data(straight_mdividend_data),
        .dividend_data_valid(straight_mdividend_data_valid),  // Updated to use skid buffer
        .dividend_data_ready(straight_mdividend_data_ready),  // Updated to use skid buffer
        .divisor_data(straight_mdivisor_data),
        .divisor_data_valid(straight_mdivisor_data_valid),  // Updated to use skid buffer
        .divisor_data_ready(straight_mdivisor_data_ready),  // Updated to use skid buffer
        .quotient_data(mquotient_data),
        .quotient_data_valid(mquotient_data_valid),
        .quotient_data_ready(mquotient_data_ready)
    );

    // Exponent calculation and join logic
    assign equotient_data = $signed(fifo_edividend_data) - $signed(fifo_edivisor_data);

    // Join the handshake signals
    join_n #(
        .NUM_HANDSHAKES(3)
    ) join_n_inst (
        .data_in_valid({mquotient_data_valid, fifo_edividend_data_valid, fifo_edivisor_data_valid}),
        .data_in_ready({mquotient_data_ready, fifo_edividend_data_ready, fifo_edivisor_data_ready}),
        .data_out_valid(quotient_data_valid),
        .data_out_ready(quotient_data_ready)
    );

endmodule