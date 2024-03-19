/*
Module      : batch_norm_2d
Description : This module calculates the 2d batch normalisation layer.
              https://arxiv.org/abs/1502.03167

*/

`timescale 1ns/1ps

module batch_norm_2d #(
    // Dimensions
    parameter TOTAL_DIM0          = 4,
    parameter TOTAL_DIM1          = 4,
    parameter COMPUTE_DIM0        = 2,
    parameter COMPUTE_DIM1        = 2,
    parameter NUM_CHANNELS        = 2,

    // Data widths
    parameter IN_WIDTH            = 8,
    parameter IN_FRAC_WIDTH       = 4,
    parameter OUT_WIDTH           = 8,
    parameter OUT_FRAC_WIDTH      = 4,

    // Scale and Shift LUTs
`ifdef COCOTB_SIM
    parameter MEM_ID              = 0,
    parameter AFFINE              = 0,
`endif
    parameter SCALE_LUT_MEMFILE   = "",
    parameter SHIFT_LUT_MEMFILE   = ""
) (
    input  logic                 clk,
    input  logic                 rst,

    input  logic [IN_WIDTH-1:0]  in_data  [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    input  logic                 in_valid,
    output logic                 in_ready,

    output logic [OUT_WIDTH-1:0] out_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    output logic                 out_valid,
    input  logic                 out_ready
);

    // Derived params
    localparam DEPTH_DIM0 = TOTAL_DIM0 / COMPUTE_DIM0;
    localparam DEPTH_DIM1 = TOTAL_DIM1 / COMPUTE_DIM1;
    localparam CH_BITS = $clog2(NUM_CHANNELS);

    localparam TEMP_MULT_WIDTH = 2 * IN_WIDTH;
    localparam TEMP_MULT_FRAC_WIDTH = IN_FRAC_WIDTH;

    localparam EXT_OUT_WIDTH = TEMP_MULT_WIDTH + 1;
    localparam EXT_OUT_FRAC_WIDTH = 2*IN_FRAC_WIDTH;

    localparam EXT_SHIFT_WIDTH = TEMP_MULT_WIDTH;
    //localparam EXT_SHIFT_FRAC_WIDTH = TEMP_FRA
    
    logic[CH_BITS-1:0] current_channel;
    logic[IN_WIDTH-1:0] scale_value;
    logic[IN_WIDTH-1:0] shift_value;
    logic[TEMP_MULT_WIDTH-1:0] temp_mult [COMPUTE_DIM0*COMPUTE_DIM1-1:0];
    logic[EXT_OUT_WIDTH-1:0] ext_out [COMPUTE_DIM0*COMPUTE_DIM1-1:0];
    logic signed[EXT_SHIFT_WIDTH-1:0] ext_shift_value;
    logic signed[EXT_SHIFT_WIDTH-1:0] temp_shift_ext;
    
    channel_selection #(
        .NUM_CHANNELS(NUM_CHANNELS)
    ) channel_selection_inst (
        .clk(clk),
        .rst(rst | !in_valid),
        .channel(current_channel)
    );
    
    lut #(
        .DATA_WIDTH(IN_WIDTH),
        .SIZE(NUM_CHANNELS),
        .OUTPUT_REG(0),
        .MEM_FILE(SCALE_LUT_MEMFILE)
    ) scale_lut_inst (
        .clk('0), // Tie off clock
        .addr(current_channel),
        .data(scale_value)
    );

    
    lut #(
        .DATA_WIDTH(IN_WIDTH),
        .SIZE(NUM_CHANNELS),
        .OUTPUT_REG(0),
        .MEM_FILE(SHIFT_LUT_MEMFILE)
    ) shift_lut_inst (
        .clk('0), // Tie off clock
        .addr(current_channel),
        .data(shift_value)
    );
    
    assign ext_shift_value = $signed(shift_value);
    assign temp_shift_ext = ext_shift_value << IN_FRAC_WIDTH;

    for (genvar i = 0; i < COMPUTE_DIM0 * COMPUTE_DIM1; i++) begin : compute_pipe
        assign temp_mult[i] = ($signed(in_data[i]) * $signed(scale_value));
        assign ext_out[i] = $signed(temp_mult[i]) + ($signed(temp_shift_ext));

        // Output Rounding Stage
        fixed_signed_cast #(
            .IN_WIDTH(EXT_OUT_WIDTH),
            .IN_FRAC_WIDTH(EXT_OUT_FRAC_WIDTH),
            .OUT_WIDTH(OUT_WIDTH),
            .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),
            .SYMMETRIC(0),
            .ROUND_FLOOR(1)
        ) output_cast (
            .in_data(ext_out[i]),
            .out_data(out_data[i])
        );

    end

    assign out_valid = in_valid;
    assign in_ready = out_ready;

endmodule
