/*
Module      : norm
Description : Module which unifies all types of normalization.

              Currently supports:
              - Batch Norm (no affine)
              - Layer Norm (no affine)
              - Instance Norm (no affine)
              - Group Norm (no affine)
              - RMS Norm (optional affine)
*/

`timescale 1ns / 1ps
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off UNUSEDPARAM */
module norm #(

    // -----
    // SOFTWARE PARAMETERS
    // -----

    // DIMENSIONS:
    // batch (3), channel (2), dim1 (1), dim0 (0)

    // PRECISION:
    // width (0), frac_width (1)

    parameter DATA_IN_0_PRECISION_0       = -1,
    parameter DATA_IN_0_PRECISION_1       = -1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = -1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = -1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = -1,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = -1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = -1,
    parameter DATA_IN_0_PARALLELISM_DIM_2 = -1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_3 = -1,
    parameter DATA_IN_0_PARALLELISM_DIM_3 = -1,

    parameter WEIGHT_PRECISION_0       = -1,
    parameter WEIGHT_PRECISION_1       = -1,
    parameter WEIGHT_TENSOR_SIZE_DIM_0 = -1,
    parameter WEIGHT_TENSOR_SIZE_DIM_1 = -1,
    parameter WEIGHT_PARALLELISM_DIM_0 = -1,
    parameter WEIGHT_PARALLELISM_DIM_1 = -1,

    parameter DATA_OUT_0_PRECISION_0       = -1,
    parameter DATA_OUT_0_PRECISION_1       = -1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = -1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = -1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = -1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = -1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_2 = -1,
    parameter DATA_OUT_0_PARALLELISM_DIM_2 = -1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_3 = -1,
    parameter DATA_OUT_0_PARALLELISM_DIM_3 = -1,

    // -----
    // HARDWARE ALIASES
    // -----

    // Dimensions
    localparam TOTAL_DIM0   = DATA_IN_0_TENSOR_SIZE_DIM_0,
    localparam TOTAL_DIM1   = DATA_IN_0_TENSOR_SIZE_DIM_1,
    localparam COMPUTE_DIM0 = DATA_IN_0_PARALLELISM_DIM_0,
    localparam COMPUTE_DIM1 = DATA_IN_0_PARALLELISM_DIM_1,

    localparam DEPTH_DIM0 = TOTAL_DIM0 / COMPUTE_DIM0,
    localparam DEPTH_DIM1 = TOTAL_DIM1 / COMPUTE_DIM1,

    // Layer: CHANNELS should be set to total number of channels
    // RMS: CHANNELS should be set to total number of channels
    // Group: CHANNELS can be set to any factor of total channels
    localparam CHANNELS = DATA_IN_0_TENSOR_SIZE_DIM_2,

    // Data widths
    localparam IN_WIDTH          = DATA_IN_0_PRECISION_0,
    localparam IN_FRAC_WIDTH     = DATA_IN_0_PRECISION_1,
    localparam WEIGHT_WIDTH      = WEIGHT_PRECISION_0,
    localparam WEIGHT_FRAC_WIDTH = WEIGHT_PRECISION_1,
    localparam OUT_WIDTH         = DATA_OUT_0_PRECISION_0,
    localparam OUT_FRAC_WIDTH    = DATA_OUT_0_PRECISION_1,

    // Inverse sqrt unit LUT file
    parameter ISQRT_LUT_MEMFILE = "",

    // Batch norm lut files
    parameter SCALE_LUT_MEMFILE = "",
    parameter SHIFT_LUT_MEMFILE = "",
    parameter MEM_ID            = 0,

    // Norm select
    // BATCH_NORM, LAYER_NORM, INSTANCE_NORM, GROUP_NORM, RMS_NORM
    parameter NORM_TYPE = "LAYER_NORM"
) (
    input logic clk,
    input logic rst,

    input  logic [IN_WIDTH-1:0] data_in_0      [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    input  logic                data_in_0_valid,
    output logic                data_in_0_ready,

    input  logic [WEIGHT_WIDTH-1:0] weight      [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    input  logic                    weight_valid,
    output logic                    weight_ready,

    output logic [OUT_WIDTH-1:0] data_out_0      [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    output logic                 data_out_0_valid,
    input  logic                 data_out_0_ready
);

  localparam BATCH_NORM = (NORM_TYPE == "BATCH_NORM");
  localparam LAYER_NORM = (NORM_TYPE == "LAYER_NORM");
  localparam INSTANCE_NORM = (NORM_TYPE == "INSTANCE_NORM");
  localparam GROUP_NORM = (NORM_TYPE == "GROUP_NORM");
  localparam RMS_NORM = (NORM_TYPE == "RMS_NORM");

  localparam NORM_CHANNELS = INSTANCE_NORM ? 1 : CHANNELS;

  generate

    if (BATCH_NORM) begin : batch_norm

      batch_norm_2d #(
          .TOTAL_DIM0(TOTAL_DIM0),
          .TOTAL_DIM1(TOTAL_DIM1),
          .COMPUTE_DIM0(COMPUTE_DIM0),
          .COMPUTE_DIM1(COMPUTE_DIM1),
          .NUM_CHANNELS(CHANNELS),
          .IN_WIDTH(IN_WIDTH),
          .IN_FRAC_WIDTH(IN_FRAC_WIDTH),
          .OUT_WIDTH(OUT_WIDTH),
          .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),
          .SCALE_LUT_MEMFILE(SCALE_LUT_MEMFILE),
          .SHIFT_LUT_MEMFILE(SHIFT_LUT_MEMFILE)
      ) batch_norm_inst (
          .clk(clk),
          .rst(rst),
          .in_data(data_in_0),
          .in_valid(data_in_0_valid),
          .in_ready(data_in_0_ready),
          .out_data(data_out_0),
          .out_valid(data_out_0_valid),
          .out_ready(data_out_0_ready)
      );

      // Weights not implemented
      assign weight_ready = '0;

    end else if (LAYER_NORM || INSTANCE_NORM || GROUP_NORM) begin : group_norm

      group_norm_2d #(
          .TOTAL_DIM0(TOTAL_DIM0),
          .TOTAL_DIM1(TOTAL_DIM1),
          .COMPUTE_DIM0(COMPUTE_DIM0),
          .COMPUTE_DIM1(COMPUTE_DIM1),
          .GROUP_CHANNELS(NORM_CHANNELS),
          .IN_WIDTH(IN_WIDTH),
          .IN_FRAC_WIDTH(IN_FRAC_WIDTH),
          .OUT_WIDTH(OUT_WIDTH),
          .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),
          .ISQRT_LUT_MEMFILE(ISQRT_LUT_MEMFILE)
      ) group_norm_inst (
          .clk(clk),
          .rst(rst),
          .in_data(data_in_0),
          .in_valid(data_in_0_valid),
          .in_ready(data_in_0_ready),
          .out_data(data_out_0),
          .out_valid(data_out_0_valid),
          .out_ready(data_out_0_ready)
      );

      // Weights not implemented
      assign weight_ready = '0;

    end else if (RMS_NORM) begin : rms_norm

      rms_norm_2d #(
          .TOTAL_DIM0(TOTAL_DIM0),
          .TOTAL_DIM1(TOTAL_DIM1),
          .COMPUTE_DIM0(COMPUTE_DIM0),
          .COMPUTE_DIM1(COMPUTE_DIM1),
          .CHANNELS(CHANNELS),
          .IN_WIDTH(IN_WIDTH),
          .IN_FRAC_WIDTH(IN_FRAC_WIDTH),
          .OUT_WIDTH(OUT_WIDTH),
          .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),
          .ISQRT_LUT_MEMFILE(ISQRT_LUT_MEMFILE)
      ) rms_norm_inst (
          .clk(clk),
          .rst(rst),
          .in_data(data_in_0),
          .in_valid(data_in_0_valid),
          .in_ready(data_in_0_ready),
          .weight_data(weight),
          .weight_valid(weight_valid),
          .weight_ready(weight_ready),
          .out_data(data_out_0),
          .out_valid(data_out_0_valid),
          .out_ready(data_out_0_ready)
      );

    end

  endgenerate

endmodule
