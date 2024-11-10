`timescale 1ns / 1ps
module fixed_taylor_exp #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_WIDTH = 4,
    parameter DATA_IN_FRAC_WIDTH = 4,
    parameter DATA_OUT_WIDTH = 8,
    parameter DATA_OUT_FRAC_WIDTH = 8
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input rst,
    input clk,
    input logic [DATA_IN_WIDTH-1:0] data_in_0,
    input  logic data_in_0_valid,
    output logic data_in_0_ready,

    output logic [DATA_OUT_WIDTH-1:0] data_out_0,
    output logic data_out_0_valid,
    input  logic data_out_0_ready
);

    localparam ORDERS = 4;
    logic [DATA_OUT_WIDTH-1:0] powers [ORDERS - 1:0];
    logic [DATA_OUT_WIDTH-1:0] powers_register_in [ORDERS - 1:0];
    logic [DATA_OUT_WIDTH-1:0] powers_with_coefficient [ORDERS - 1:0];
    logic powers_valid, powers_ready;
    
    power #(
        .DATA_IN_WIDTH(DATA_IN_WIDTH),
        .DATA_IN_FRAC_WIDTH(DATA_IN_FRAC_WIDTH),
        .DATA_OUT_WIDTH(DATA_OUT_WIDTH),
        .DATA_OUT_FRAC_WIDTH(DATA_OUT_FRAC_WIDTH),
        .ORDERS(ORDERS)
    ) power_inst (
        .data_in(data_in_0),
        .data_out(powers_register_in)
    );
    unpacked_register_slice #(
        .DATA_WIDTH(DATA_OUT_WIDTH),
        .IN_SIZE   (ORDERS)
    ) register_slice_i (
        .clk(clk),
        .rst(rst),

        .data_in(powers_register_in),
        .data_in_valid(data_in_0_valid),
        .data_in_ready(data_in_0_ready),

        .data_out(powers),
        .data_out_valid(powers_valid),
        .data_out_ready(powers_ready)
    ); 
    assign powers_with_coefficient[0] = powers[0];
    assign powers_with_coefficient[1] = powers[1];
    assign powers_with_coefficient[2] = powers[2]>>>1;
    assign powers_with_coefficient[3] = $signed(powers[3]) * 4'b0111>>>5;

    assign data_out_0 = powers_with_coefficient[0] + powers_with_coefficient[1] + powers_with_coefficient[2] + powers_with_coefficient[3];
    assign data_out_0_valid = powers_valid;
    assign powers_ready = data_out_0_ready;
    

endmodule

module power #(
    parameter DATA_IN_WIDTH = 8,
    parameter DATA_IN_FRAC_WIDTH = 4,
    parameter DATA_OUT_WIDTH = 8,
    parameter DATA_OUT_FRAC_WIDTH = 4,
    parameter ORDERS = 4
) (
    input logic [DATA_IN_WIDTH-1:0] data_in,
    output logic [DATA_OUT_WIDTH-1:0] data_out [ORDERS - 1:0]
);

    assign data_out[0] = 1<<DATA_OUT_FRAC_WIDTH;
    
    for (genvar i = 0; i < ORDERS - 1; i++) begin
        logic [DATA_IN_WIDTH * 2 - 1:0] intermediate_data_out;
        assign intermediate_data_out = $signed(data_out[i]) * $signed(data_in);
        fixed_signed_cast #(
            .IN_WIDTH(DATA_IN_WIDTH * 2),
            .IN_FRAC_WIDTH(DATA_IN_FRAC_WIDTH * 2),
            .OUT_WIDTH(DATA_OUT_WIDTH),
            .OUT_FRAC_WIDTH(DATA_OUT_FRAC_WIDTH),
            .ROUND_FLOOR(1)
        ) fr_inst (
            .in_data (intermediate_data_out),
            .out_data(data_out[i + 1])
        );
    end 
    
endmodule