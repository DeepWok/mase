`timescale 1 ns / 1 ps
module mxint_straightm_fifoe #(
    parameter DEPTH = 8,
    parameter MAN_WIDTH = 8,
    parameter EXP_WIDTH = 8,
    parameter IN_SIZE = 8
) (
    input clk,
    input rst,
    // Input interface
    input [MAN_WIDTH-1:0] mdata_in[IN_SIZE - 1:0],
    input [EXP_WIDTH-1:0] edata_in,
    input logic data_in_valid,
    output logic data_in_ready,
    // FIFO output interface
    output [EXP_WIDTH-1:0] fifo_edata_out,
    output logic fifo_edata_out_valid,
    input logic fifo_edata_out_ready,
    // Straight output interface
    output [MAN_WIDTH-1:0] straight_mdata_out[IN_SIZE - 1:0],
    output logic straight_mdata_out_valid,
    input logic straight_mdata_out_ready
);
    logic mdata_in_ready, edata_in_ready;
    logic mdata_in_valid, edata_in_valid;

    // Split the input valid signal for mantissa and exponent paths
    split2 #() data_out_n_split_i (
        .data_in_valid (data_in_valid),
        .data_in_ready (data_in_ready),
        .data_out_valid({mdata_in_valid, edata_in_valid}),
        .data_out_ready({mdata_in_ready, edata_in_ready})
    );

    // FIFO for exponent data
    fifo #(
        .DEPTH(DEPTH),
        .DATA_WIDTH(EXP_WIDTH)
    ) ff_inst (
        .clk(clk),
        .rst(rst),
        .in_data(edata_in),
        .in_valid(edata_in_valid),
        .in_ready(edata_in_ready),
        .out_data(fifo_edata_out),
        .out_valid(fifo_edata_out_valid),
        .out_ready(fifo_edata_out_ready),
        .empty(),
        .full()
    );

    // Unpacked FIFO for mantissa data
    unpacked_fifo #(
        .DEPTH(1),
        .DATA_WIDTH(MAN_WIDTH),
        .IN_NUM(IN_SIZE)
    ) mdata_fifo (
        .clk(clk),
        .rst(rst),
        .data_in(mdata_in),
        .data_in_valid(mdata_in_valid),
        .data_in_ready(mdata_in_ready),
        .data_out(straight_mdata_out),
        .data_out_valid(straight_mdata_out_valid),
        .data_out_ready(straight_mdata_out_ready)
    );

endmodule
