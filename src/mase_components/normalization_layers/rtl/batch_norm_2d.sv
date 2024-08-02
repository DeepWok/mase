/*
Module      : batch_norm_2d
Description : This module implements 2D batch normalisation.
              https://arxiv.org/abs/1502.03167
*/

`timescale 1ns / 1ps
module batch_norm_2d #(
    // Dimensions
    parameter TOTAL_DIM0   = 4,
    parameter TOTAL_DIM1   = 4,
    parameter COMPUTE_DIM0 = 2,
    parameter COMPUTE_DIM1 = 2,
    parameter NUM_CHANNELS = 2,

    // Data widths
    parameter IN_WIDTH       = 8,
    parameter IN_FRAC_WIDTH  = 4,
    parameter OUT_WIDTH      = 8,
    parameter OUT_FRAC_WIDTH = 4,

    // Scale and Shift LUTs
    /* verilator lint_off UNUSEDPARAM */
    parameter MEM_ID            = 0,
    /* verilator lint_on UNUSEDPARAM */
`ifdef COCOTB_SIM
    parameter AFFINE            = 0,
`endif
    parameter SCALE_LUT_MEMFILE = "",
    parameter SHIFT_LUT_MEMFILE = ""
) (
    input logic clk,
    input logic rst,

    input  logic [IN_WIDTH-1:0] in_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    input  logic                in_valid,
    output logic                in_ready,

    output logic [OUT_WIDTH-1:0] out_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    output logic                 out_valid,
    input  logic                 out_ready
);

  // Derived params
  localparam DEPTH_DIM0 = TOTAL_DIM0 / COMPUTE_DIM0;
  localparam DEPTH_DIM1 = TOTAL_DIM1 / COMPUTE_DIM1;
  localparam CH_BITS = $clog2(NUM_CHANNELS);

  localparam IN_FLAT_WIDTH = IN_WIDTH * COMPUTE_DIM0 * COMPUTE_DIM1;

  localparam TEMP_MULT_WIDTH = 2 * IN_WIDTH;

  localparam EXT_OUT_WIDTH = TEMP_MULT_WIDTH + 1;
  localparam EXT_OUT_FRAC_WIDTH = 2 * IN_FRAC_WIDTH;

  localparam EXT_SHIFT_WIDTH = TEMP_MULT_WIDTH;

  logic [CH_BITS-1:0] current_channel;
  logic [IN_WIDTH-1:0] reg_data[COMPUTE_DIM0*COMPUTE_DIM1-1:0];
  logic [IN_WIDTH-1:0] scale_value[1:0];
  logic [IN_WIDTH-1:0] shift_value[2:0];
  logic [TEMP_MULT_WIDTH-1:0] temp_mult_in[COMPUTE_DIM0*COMPUTE_DIM1-1:0];
  logic [TEMP_MULT_WIDTH-1:0] temp_mult_out[COMPUTE_DIM0*COMPUTE_DIM1-1:0];
  logic [EXT_OUT_WIDTH-1:0] ext_out[COMPUTE_DIM0*COMPUTE_DIM1-1:0];
  logic signed [EXT_SHIFT_WIDTH-1:0] ext_shift_value;
  logic signed [EXT_SHIFT_WIDTH-1:0] temp_shift_ext;
  logic [OUT_WIDTH-1:0] out_round_data[COMPUTE_DIM0*COMPUTE_DIM1-1:0];
  logic increment_count;
  logic [IN_FLAT_WIDTH-1:0] in_data_flat, reg_data_flat;
  logic shift_scale_valid;
  logic shift_scale_ready[COMPUTE_DIM0*COMPUTE_DIM1-1:0];
  logic mult_valid[COMPUTE_DIM0*COMPUTE_DIM1-1:0];
  logic mult_ready[COMPUTE_DIM0*COMPUTE_DIM1-1:0];
  logic out_reg_ready;
  logic out_reg_valid[COMPUTE_DIM0*COMPUTE_DIM1-1:0];

  assign increment_count = in_valid && in_ready;

  channel_selection #(
      .NUM_CHANNELS(NUM_CHANNELS),
      .NUM_SPATIAL_BLOCKS(DEPTH_DIM0 * DEPTH_DIM1)
  ) channel_selection_inst (
      .clk(clk),
      .rst(rst),
      .inc(increment_count),
      .channel(current_channel)
  );

  lut #(
      .DATA_WIDTH(IN_WIDTH),
      .SIZE(NUM_CHANNELS),
      .OUTPUT_REG(0),
      .MEM_FILE(SCALE_LUT_MEMFILE)
  ) scale_lut_inst (
      .clk('0),  // Tie off clock
      .addr(current_channel),
      .data(scale_value[0])
  );

  lut #(
      .DATA_WIDTH(IN_WIDTH),
      .SIZE(NUM_CHANNELS),
      .OUTPUT_REG(0),
      .MEM_FILE(SHIFT_LUT_MEMFILE)
  ) shift_lut_inst (
      .clk('0),  // Tie off clock
      .addr(current_channel),
      .data(shift_value[0])
  );

  // ------
  // Pipeline Reg 0 - in_data, scale_value, shift_value
  // ------

  matrix_flatten #(
      .DATA_WIDTH(IN_WIDTH),
      .DIM0(COMPUTE_DIM0),
      .DIM1(COMPUTE_DIM1)
  ) in_data_flatten (
      .data_in (in_data),
      .data_out(in_data_flat)
  );

  skid_buffer #(
      .DATA_WIDTH(IN_FLAT_WIDTH + 2 * IN_WIDTH)
  ) pipe_reg_0 (
      .clk(clk),
      .rst(rst),
      .data_in({in_data_flat, scale_value[0], shift_value[0]}),
      .data_in_valid(in_valid),
      .data_in_ready(in_ready),
      .data_out({reg_data_flat, scale_value[1], shift_value[1]}),
      .data_out_valid(shift_scale_valid),
      .data_out_ready(shift_scale_ready[0])
  );

  matrix_unflatten #(
      .DATA_WIDTH(IN_WIDTH),
      .DIM0(COMPUTE_DIM0),
      .DIM1(COMPUTE_DIM1)
  ) reg_data_unflatten (
      .data_in (reg_data_flat),
      .data_out(reg_data)
  );

  // ------
  // Pipeline Reg 1 - shift_value
  // ------
  skid_buffer #(
      .DATA_WIDTH(IN_WIDTH)
  ) pipe_reg_1_shift_val (
      .clk(clk),
      .rst(rst),
      .data_in({shift_value[1]}),
      .data_in_valid(shift_scale_valid),
      .data_in_ready(),
      .data_out({shift_value[2]}),
      .data_out_valid(),
      .data_out_ready(mult_ready[0])
  );

  assign ext_shift_value = $signed(shift_value[2]);
  assign temp_shift_ext  = ext_shift_value << IN_FRAC_WIDTH;

  for (genvar i = 0; i < COMPUTE_DIM0 * COMPUTE_DIM1; i++) begin : compute_pipe

    assign temp_mult_in[i] = ($signed(reg_data[i]) * $signed(scale_value[1]));

    // ------
    // Pipeline Reg 1 - temp_mult, shift_value
    // ------

    skid_buffer #(
        .DATA_WIDTH(TEMP_MULT_WIDTH)
    ) pipe_reg_1 (
        .clk(clk),
        .rst(rst),
        .data_in({temp_mult_in[i]}),
        .data_in_valid(shift_scale_valid),
        .data_in_ready(shift_scale_ready[i]),
        .data_out({temp_mult_out[i]}),
        .data_out_valid(mult_valid[i]),
        .data_out_ready(mult_ready[i])
    );

    assign ext_out[i] = $signed(temp_mult_out[i]) + ($signed(temp_shift_ext));

    // Output Rounding Stage
    fixed_signed_cast #(
        .IN_WIDTH(EXT_OUT_WIDTH),
        .IN_FRAC_WIDTH(EXT_OUT_FRAC_WIDTH),
        .OUT_WIDTH(OUT_WIDTH),
        .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),
        .SYMMETRIC(0),
        .ROUND_FLOOR(1)
    ) output_cast (
        .in_data (ext_out[i]),
        .out_data(out_round_data[i])
    );

    // Output Register

    skid_buffer #(
        .DATA_WIDTH(OUT_WIDTH)
    ) out_reg (
        .clk(clk),
        .rst(rst),
        .data_in(out_round_data[i]),
        .data_in_valid(mult_valid[i]),
        .data_in_ready(mult_ready[i]),
        .data_out(out_data[i]),
        .data_out_valid(out_reg_valid[i]),
        .data_out_ready(out_reg_ready)
    );

  end

  // Output handshake
  assign out_valid = out_reg_valid[0];
  assign out_reg_ready = out_ready;

endmodule
