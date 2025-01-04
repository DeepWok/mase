`timescale 1 ns / 1 ps
module int_div #(
    parameter IN_NUM = 8,
    parameter FIFO_DEPTH = 8,
    parameter DIVIDEND_WIDTH = 8,
    parameter DIVISOR_WIDTH = 8,
    parameter QUOTIENT_WIDTH = 8
) (
    input logic clk,
    input logic rst,
    input logic [DIVIDEND_WIDTH-1:0] dividend_data[IN_NUM - 1:0],
    input logic dividend_data_valid,
    output logic dividend_data_ready,
    input logic [DIVISOR_WIDTH-1:0] divisor_data[IN_NUM - 1:0],
    input logic divisor_data_valid,
    output logic divisor_data_ready,
    output logic [QUOTIENT_WIDTH-1:0] quotient_data[IN_NUM - 1:0],
    output logic quotient_data_valid,
    input logic quotient_data_ready
);

    // Add signals for skid buffers
    logic [DIVIDEND_WIDTH-1:0] dividend_data_skid[IN_NUM - 1:0];
    logic dividend_valid_skid;
    logic dividend_ready_skid;
    
    logic [DIVISOR_WIDTH-1:0] divisor_data_skid[IN_NUM - 1:0];
    logic divisor_valid_skid;
    logic divisor_ready_skid;
    
    // Replace skid buffers with unpacked versions
    unpacked_skid_buffer #(
        .DATA_WIDTH(DIVIDEND_WIDTH),
        .IN_NUM(IN_NUM)
    ) dividend_skid (
        .clk(clk),
        .rst(rst),
        .data_in(dividend_data),
        .data_in_valid(dividend_data_valid),
        .data_in_ready(dividend_data_ready),
        .data_out(dividend_data_skid),
        .data_out_valid(dividend_valid_skid),
        .data_out_ready(dividend_ready_skid)
    );

    unpacked_skid_buffer #(
        .DATA_WIDTH(DIVISOR_WIDTH),
        .IN_NUM(IN_NUM)
    ) divisor_skid (
        .clk(clk),
        .rst(rst),
        .data_in(divisor_data),
        .data_in_valid(divisor_data_valid),
        .data_in_ready(divisor_data_ready),
        .data_out(divisor_data_skid),
        .data_out_valid(divisor_valid_skid),
        .data_out_ready(divisor_ready_skid)
    );

    // Update join2 instance to use skid buffer outputs
    join2 #(
    ) join2_inst (
        .data_in_valid({dividend_valid_skid, divisor_valid_skid}),
        .data_in_ready({dividend_ready_skid, divisor_ready_skid}),
        .data_out_valid(quotient_data_valid),
        .data_out_ready(quotient_data_ready)
    );
    
    // Update division operation to use skid buffer outputs
    for(genvar i = 0; i < IN_NUM; i++) begin 
        assign quotient_data[i] = dividend_data_skid[i] / divisor_data_skid[i];
    end

endmodule