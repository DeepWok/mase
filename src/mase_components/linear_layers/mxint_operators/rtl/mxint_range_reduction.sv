`timescale 1ns / 1ps
/*
 This code actually input mxint and then output rounded integer n,
 In the first version, we just keep the width of n is 8
 which means like output n range from [-128:127]
*/
module mxint_range_reduction #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_MAN_WIDTH = 4,
    parameter DATA_IN_EXP_WIDTH = 8,
    parameter BLOCK_SIZE = 16,
    parameter DATA_OUT_N_WIDTH = 8
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input rst,
    input clk,
    input logic [DATA_IN_MAN_WIDTH-1:0] mdata_in_0[BLOCK_SIZE - 1:0],
    input logic [DATA_IN_EXP_WIDTH-1:0] edata_in_0,
    input  logic data_in_0_valid,
    output logic data_in_0_ready,

    output logic [DATA_OUT_N_WIDTH-1:0] data_out_n [BLOCK_SIZE - 1 : 0],
    output logic data_out_n_valid,
    input  logic data_out_n_ready,

    output logic [9-1:0] data_out_r [BLOCK_SIZE - 1 : 0],
    output logic data_out_r_valid,
    input  logic data_out_r_ready
);
    localparam signed MLOG2_E = 8'd92;
    localparam signed ELOG2_E = 4'd1;
    localparam signed MLN_2 = 8'd88;
    localparam signed ELN_2 = 4'd0;

    localparam DATA_LOG2_E_MAN_WIDTH = DATA_IN_MAN_WIDTH + 8;
    localparam DATA_LOG2_E_MAN_FRAC_WIDTH = DATA_IN_MAN_WIDTH - 1 + 8 - 1;
    localparam DATA_LOG2_E_EXP_WIDTH = DATA_IN_EXP_WIDTH;

    localparam DATA_LN_2_MAN_WIDTH = DATA_OUT_N_WIDTH + 8;
    localparam DATA_LN_2_MAN_FRAC_WIDTH = 8 - 1; // N is integer
    localparam DATA_LN_2_EXP_WIDTH = DATA_IN_EXP_WIDTH;

    localparam SHIFT_WIDTH = DATA_IN_EXP_WIDTH;

    logic [DATA_IN_MAN_WIDTH-1:0] fifo_mdata_in [BLOCK_SIZE - 1:0];
    logic [DATA_IN_EXP_WIDTH-1:0] fifo_edata_in;
    logic fifo_data_in_valid;
    logic fifo_data_in_ready;

    logic [DATA_IN_MAN_WIDTH-1:0] straight_mdata_in [BLOCK_SIZE - 1:0];
    logic [DATA_IN_EXP_WIDTH-1:0] straight_edata_in;
    logic straight_data_in_valid;
    logic straight_data_in_ready;

    logic [DATA_LOG2_E_MAN_WIDTH - 1:0] mdata_in_0_log2_e [BLOCK_SIZE - 1:0];
    logic [DATA_LOG2_E_EXP_WIDTH - 1:0] edata_in_0_log2_e;

    logic [DATA_OUT_N_WIDTH-1:0] temp_data_out_n [BLOCK_SIZE - 1 : 0];
    logic temp_data_out_n_valid, temp_data_out_n_ready;
    
    logic [DATA_OUT_N_WIDTH-1:0] straight_data_out_n [BLOCK_SIZE - 1 : 0];
    logic straight_data_out_n_valid, straight_data_out_n_ready;

    logic [DATA_LN_2_MAN_WIDTH - 1:0] mn_ln_2 [BLOCK_SIZE - 1:0];
    logic [DATA_LN_2_EXP_WIDTH - 1:0] en_ln_2;

    logic [DATA_LN_2_MAN_WIDTH - 1:0] shifted_fifo_mdata_in [BLOCK_SIZE - 1:0];
    logic [SHIFT_WIDTH - 1:0] shift_value;

    logic [DATA_LN_2_MAN_WIDTH - 1:0] clamped_in [BLOCK_SIZE - 1:0];
    logic [9 - 1:0] regi_r_in [BLOCK_SIZE - 1:0];
    logic regi_r_in_valid, regi_r_in_ready;

    unpacked_mx_split2_with_data #(
        .DEPTH(2),
        .MAN_WIDTH(DATA_IN_MAN_WIDTH),
        .EXP_WIDTH(DATA_IN_EXP_WIDTH),
        .IN_SIZE(BLOCK_SIZE)
    ) unpacked_mx_split2_with_data_i (
        .clk(clk),
        .rst(rst),
        .mdata_in(mdata_in_0),
        .edata_in(edata_in_0),
        .data_in_valid(data_in_0_valid),
        .data_in_ready(data_in_0_ready),
        .fifo_mdata_out(fifo_mdata_in),
        .fifo_edata_out(fifo_edata_in),
        .fifo_data_out_valid(fifo_data_in_valid),
        .fifo_data_out_ready(fifo_data_in_ready),
        .straight_mdata_out(straight_mdata_in),
        .straight_edata_out(straight_edata_in),
        .straight_data_out_valid(straight_data_in_valid),
        .straight_data_out_ready(straight_data_in_ready)
    );

    for (genvar i = 0; i < BLOCK_SIZE; i++) begin
        assign mdata_in_0_log2_e[i] = $signed(straight_mdata_in[i])*MLOG2_E;
    end
    assign edata_in_0_log2_e = $signed(straight_edata_in) + ELOG2_E;

    mxint_hardware_round #(
        .DATA_IN_MAN_WIDTH(DATA_LOG2_E_MAN_WIDTH),
        .DATA_IN_MAN_FRAC_WIDTH(DATA_LOG2_E_MAN_FRAC_WIDTH),
        .DATA_IN_EXP_WIDTH(DATA_LOG2_E_EXP_WIDTH),
        .BLOCK_SIZE(BLOCK_SIZE),
        .DATA_OUT_WIDTH(DATA_OUT_N_WIDTH)
    ) mxint_hardware_round_i (
        .rst(rst),
        .clk(clk),
        .mdata_in_0(mdata_in_0_log2_e),
        .edata_in_0(edata_in_0_log2_e),
        .data_in_0_valid(straight_data_in_valid),
        .data_in_0_ready(straight_data_in_ready),
        .data_out_0(temp_data_out_n),
        .data_out_0_valid(temp_data_out_n_valid),
        .data_out_0_ready(temp_data_out_n_ready)
    );

    unpacked_split2_with_data #(
        .DEPTH(3),
        .DATA_WIDTH(DATA_OUT_N_WIDTH),
        .IN_SIZE(BLOCK_SIZE)
    ) unpacked_split2_with_data_i (
        .clk(clk),
        .rst(rst),
        .data_in(temp_data_out_n),
        .data_in_valid(temp_data_out_n_valid),
        .data_in_ready(temp_data_out_n_ready),
        .fifo_data_out(data_out_n),
        .fifo_data_out_valid(data_out_n_valid),
        .fifo_data_out_ready(data_out_n_ready),
        .straight_data_out(straight_data_out_n),
        .straight_data_out_valid(straight_data_out_n_valid),
        .straight_data_out_ready(straight_data_out_n_ready)
    );
    for (genvar i = 0; i < BLOCK_SIZE; i++) begin
        assign mn_ln_2[i] = $signed(straight_data_out_n[i]) * MLN_2;
    end
    assign en_ln_2 = ELN_2;

    assign shift_value = en_ln_2 - $signed(fifo_edata_in) + DATA_IN_MAN_WIDTH - 1 - 7;  
    join2 #() acc_join_inst (
        .data_in_ready ({straight_data_out_n_ready, fifo_data_in_ready}),
        .data_in_valid ({straight_data_out_n_valid, fifo_data_in_valid}),
        .data_out_valid(regi_r_in_valid),
        .data_out_ready(regi_r_in_ready)
    );
    optimized_right_shift #(
        .IN_WIDTH(DATA_IN_MAN_WIDTH),
        .SHIFT_WIDTH(SHIFT_WIDTH),
        .OUT_WIDTH(DATA_LN_2_MAN_WIDTH),
        .BLOCK_SIZE(BLOCK_SIZE)
    ) ovshift_inst (
        .data_in(fifo_mdata_in),
        .shift_value(shift_value),
        .data_out(shifted_fifo_mdata_in)
    );

    for (genvar i=0; i < BLOCK_SIZE; i++) begin
        assign clamped_in[i] = $signed(shifted_fifo_mdata_in[i]) - $signed(mn_ln_2[i]);
    end

    for (genvar i = 0; i < BLOCK_SIZE; i++) begin
        signed_clamp #(
            .IN_WIDTH (DATA_LN_2_MAN_WIDTH),
            .OUT_WIDTH(9)
        ) data_clamp (
            .in_data (clamped_in[i]),
            .out_data(regi_r_in[i])
        );
    end 
    unpacked_register_slice #(
        .DATA_WIDTH(9),
        .IN_SIZE   (BLOCK_SIZE)
    ) register_slice_i (
        .clk(clk),
        .rst(rst),

        .data_in(regi_r_in),
        .data_in_valid(regi_r_in_valid),
        .data_in_ready(regi_r_in_ready),

        .data_out(data_out_r),
        .data_out_valid(data_out_r_valid),
        .data_out_ready(data_out_r_ready)
    );
endmodule

