/*
Module      : mxint_linear
Description : This module does a matrix multiplcation between matrices X & Y.

              Python equivalent:
              out = torch.nn.functional.linear(x, w, bias)

              x should be the dimension of (DATA_IN_0_TENSOR_SIZE_DIM_1, DATA_IN_0_TENSOR_SIZE_DIM_0)
              w should be the dimension of (WEIGHT_TENSOR_SIZE_DIM_1, WEIGHT_TENSOR_SIZE_DIM_0)
              bias should be the dimension of (BIAS_TENSOR_SIZE_DIM_1, BIAS_TENSOR_SIZE_DIM_0)
              output will be (DATA_IN_0_TENSOR_SIZE_DIM_1, WEIGHT_TENSOR_SIZE_DIM_1)
              so the WEIGHT_TENSOR_SIZE_DIM_0 should be equal to TENSOR_SIZE_DIM_0
*/
`timescale 1ns / 1ps

module mxint_linear #(
    /* verilator lint_off UNUSEDPARAM */
    parameter HAS_BIAS = 1,
    parameter CIRCULAR_WEIGHT = 0,
    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 20,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 20,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 4,  // must equal WEIGHT_PARALLELISM_DIM_1
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_2 = 1,
    localparam IN_0_DEPTH_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0,
    localparam IN_0_DEPTH_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1 / DATA_IN_0_PARALLELISM_DIM_1,

    parameter WEIGHT_PRECISION_0 = 16,
    parameter WEIGHT_PRECISION_1 = 3,
    parameter WEIGHT_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter WEIGHT_TENSOR_SIZE_DIM_1 = 20,
    parameter WEIGHT_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,
    parameter WEIGHT_PARALLELISM_DIM_1 = 4,

    // Inferred precision of the output data
    // if the data out precision will be replaced by the setting
    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
        DATA_IN_0_TENSOR_SIZE_DIM_0
    ) + HAS_BIAS,
    parameter DATA_OUT_0_PRECISION_1 = DATA_IN_0_PRECISION_1 + WEIGHT_PRECISION_1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = WEIGHT_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = WEIGHT_PARALLELISM_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_2 = 1,

    parameter UNDERFLOW_BITS = 0,
    parameter ROUND_BITS = 4,
    parameter BIAS_PRECISION_0 = 16,
    parameter BIAS_PRECISION_1 = 3,
    parameter BIAS_TENSOR_SIZE_DIM_0 = DATA_OUT_0_TENSOR_SIZE_DIM_0,
    parameter BIAS_TENSOR_SIZE_DIM_1 = 1,
    parameter BIAS_PARALLELISM_DIM_0 = DATA_OUT_0_PARALLELISM_DIM_0,
    parameter BIAS_PARALLELISM_DIM_1 = 1
) (
    input clk,
    input rst,

    // input port for data_inivations
    input logic [DATA_IN_0_PRECISION_0-1:0] mdata_in_0 [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input logic [DATA_IN_0_PRECISION_1-1:0] edata_in_0,
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    // input port for weight
    input logic [WEIGHT_PRECISION_0-1:0] mweight [WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1-1:0],
    input logic [WEIGHT_PRECISION_1-1:0] eweight,
    input logic weight_valid,
    output logic weight_ready,

    input logic [BIAS_PRECISION_0-1:0] mbias[BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1 -1:0],
    input logic [BIAS_PRECISION_1-1:0] ebias,
    input logic bias_valid,
    output logic bias_ready,

    output logic [DATA_OUT_0_PRECISION_0-1:0] mdata_out_0 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic [DATA_OUT_0_PRECISION_1-1:0] edata_out_0,
    output logic data_out_0_valid,
    input logic data_out_0_ready
);
  localparam FDP_WIDTH = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
      DATA_IN_0_PARALLELISM_DIM_0
  );
  localparam FDP_EXP_WIDTH = (WEIGHT_PRECISION_1 > DATA_IN_0_PRECISION_1)? WEIGHT_PRECISION_1 + 1: DATA_IN_0_PRECISION_1 + 1;
  localparam ACC_WIDTH = FDP_WIDTH + $clog2(IN_0_DEPTH_DIM_0) + UNDERFLOW_BITS;
  localparam ACC_FRAC_WIDTH = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + UNDERFLOW_BITS - 2;
  localparam ACC_EXP_WIDTH = FDP_EXP_WIDTH;
  localparam LOSSLESS_OUT_WIDTH = ACC_WIDTH + HAS_BIAS;
  localparam LOSSLESS_OUT_EXP_WIDTH = ACC_EXP_WIDTH;
  localparam LOSSLESS_OUT_FRAC_WIDTH = ACC_FRAC_WIDTH;

  logic [DATA_IN_0_PRECISION_0-1:0]circular_mdata_in_0[DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [DATA_IN_0_PRECISION_1-1:0] circular_edata_in_0;
  logic circular_data_in_0_valid, circular_data_in_0_ready;

  logic [WEIGHT_PRECISION_0-1:0]circular_mweight[WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1-1:0];
  logic [WEIGHT_PRECISION_1-1:0] circular_eweight;
  logic circular_weight_valid, circular_weight_ready;

  logic [BIAS_PRECISION_0-1:0] circular_mbias [BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1-1:0];
  logic [BIAS_PRECISION_1-1:0] circular_ebias;
  logic circular_bias_valid, circular_bias_ready;

  logic [FDP_EXP_WIDTH-1:0] fdp_edata_out [DATA_IN_0_PARALLELISM_DIM_1 * WEIGHT_PARALLELISM_DIM_1 - 1:0];
  logic [DATA_IN_0_PARALLELISM_DIM_1 * WEIGHT_PARALLELISM_DIM_1 - 1:0] fdp_data_out_valid;

  // Change signal names to reflect data flow: dp -> skid -> acc
  logic [FDP_WIDTH-1:0] dp_mdata_out [DATA_IN_0_PARALLELISM_DIM_1 * WEIGHT_PARALLELISM_DIM_1 - 1:0];
  logic [FDP_EXP_WIDTH-1:0] dp_edata_out;
  logic dp_data_out_valid, dp_data_out_ready;

  logic [FDP_WIDTH-1:0] skid_mdata_out[DATA_IN_0_PARALLELISM_DIM_1 * WEIGHT_PARALLELISM_DIM_1 - 1:0];
  logic [FDP_EXP_WIDTH-1:0] skid_edata_out;
  logic skid_data_out_valid, skid_data_out_ready;

  logic [ACC_WIDTH-1:0] acc_mdata_out [DATA_IN_0_PARALLELISM_DIM_1 * DATA_OUT_0_PARALLELISM_DIM_0-1:0];
  logic [FDP_EXP_WIDTH-1:0] acc_edata_out;
  logic acc_data_out_valid, acc_data_out_ready;
  logic [LOSSLESS_OUT_WIDTH-1:0] cast_mdata_out_0[DATA_OUT_0_PARALLELISM_DIM_1 * DATA_OUT_0_PARALLELISM_DIM_0-1:0];
  logic [LOSSLESS_OUT_EXP_WIDTH-1:0] cast_edata_out_0;
  logic cast_data_out_0_valid, cast_data_out_0_ready;

  logic [LOSSLESS_OUT_WIDTH-1:0] fifo_mdata_in[DATA_OUT_0_PARALLELISM_DIM_1 * DATA_OUT_0_PARALLELISM_DIM_0-1:0];
  logic [LOSSLESS_OUT_EXP_WIDTH-1:0] fifo_edata_in;
  logic fifo_data_in_valid, fifo_data_in_ready;
  // Add signals for FIFO
  logic [LOSSLESS_OUT_WIDTH-1:0] fifo_mdata_out[DATA_OUT_0_PARALLELISM_DIM_1 * DATA_OUT_0_PARALLELISM_DIM_0-1:0];
  logic [LOSSLESS_OUT_EXP_WIDTH-1:0] fifo_edata_out;
  logic fifo_data_out_valid, fifo_data_out_ready;
  mxint_circular #(
      .DATA_PRECISION_0(DATA_IN_0_PRECISION_0),
      .DATA_PRECISION_1(DATA_IN_0_PRECISION_1),
      .IN_NUM          (DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1),
      .REPEAT          (WEIGHT_TENSOR_SIZE_DIM_1 / WEIGHT_PARALLELISM_DIM_1),
      .BUFFER_SIZE     (IN_0_DEPTH_DIM_0)
  ) data_in_0_buffer (
      .clk,
      .rst,
      // Input streaming port
      .mdata_in(mdata_in_0),
      .edata_in(edata_in_0),
      .data_in_valid(data_in_0_valid),
      .data_in_ready(data_in_0_ready),
      // Output streaming port
      .mdata_out(circular_mdata_in_0),
      .edata_out(circular_edata_in_0),
      .data_out_valid(circular_data_in_0_valid),
      .data_out_ready(circular_data_in_0_ready)
  );
  if (CIRCULAR_WEIGHT == 1) begin
    mxint_circular #(
        .DATA_PRECISION_0(WEIGHT_PRECISION_0),
        .DATA_PRECISION_1(WEIGHT_PRECISION_1),
        .IN_NUM(WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1),
        .REPEAT(IN_0_DEPTH_DIM_1),
        .BUFFER_SIZE(WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1 / (WEIGHT_PARALLELISM_DIM_0*WEIGHT_PARALLELISM_DIM_1))
    ) weight_buffer (
        .clk,
        .rst,
        // Input streaming port
        .mdata_in(mweight),
        .edata_in(eweight),
        .data_in_valid(weight_valid),
        .data_in_ready(weight_ready),
        // Output streaming port
        .mdata_out(circular_mweight),
        .edata_out(circular_eweight),
        .data_out_valid(circular_weight_valid),
        .data_out_ready(circular_weight_ready)
    );

    mxint_circular #(
        .DATA_PRECISION_0(BIAS_PRECISION_0),
        .DATA_PRECISION_1(BIAS_PRECISION_1),
        .IN_NUM          (BIAS_PARALLELISM_DIM_0),
        .REPEAT          (IN_0_DEPTH_DIM_1),
        .BUFFER_SIZE     (BIAS_TENSOR_SIZE_DIM_0 / (BIAS_PARALLELISM_DIM_0))
    ) bias_buffer (
        .clk,
        .rst,
        // Input streaming port
        .mdata_in(mbias),
        .edata_in(ebias),
        .data_in_valid(bias_valid),
        .data_in_ready(bias_ready),
        // Output streaming port
        .mdata_out(circular_mbias),
        .edata_out(circular_ebias),
        .data_out_valid(circular_bias_valid),
        .data_out_ready(circular_bias_ready)
    );
  end else begin
    mxint_skid_buffer #(
        .DATA_PRECISION_0(BIAS_PRECISION_0),
        .DATA_PRECISION_1(BIAS_PRECISION_1),
        .IN_NUM(WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1)
    ) weight_reg_inst (
        .clk(clk),
        .rst(rst),
        .mdata_in(mweight),
        .edata_in(eweight),
        .data_in_valid(weight_valid),
        .data_in_ready(weight_ready),
        .mdata_out(circular_mweight),
        .edata_out(circular_eweight),
        .data_out_valid(circular_weight_valid),
        .data_out_ready(circular_weight_ready)
    );
    mxint_skid_buffer #(
        .DATA_PRECISION_0(BIAS_PRECISION_0),
        .DATA_PRECISION_1(BIAS_PRECISION_1),
        .IN_NUM(DATA_OUT_0_PARALLELISM_DIM_0)
    ) bias_reg_inst (
        .clk(clk),
        .rst(rst),
        .mdata_in(mbias),
        .edata_in(ebias),
        .data_in_valid(bias_valid),
        .data_in_ready(bias_ready),
        .mdata_out(circular_mbias),
        .edata_out(circular_ebias),
        .data_out_valid(circular_bias_valid),
        .data_out_ready(circular_bias_ready)
    );
  end
  /* verilator lint_off UNUSEDSIGNAL */
  // Assume the parallelised hardware above have the same arrival time
  // which means that they always have the same state. So we can just
  // pick one of the valid signal to use.
  logic [DATA_IN_0_PARALLELISM_DIM_1*WEIGHT_PARALLELISM_DIM_1-1:0]
      fdp_data_ready, fdp_weight_ready;
  assign circular_weight_ready = fdp_weight_ready[0];
  assign circular_data_in_0_ready = fdp_data_ready[0];

  // There are WEIGHT_PARALLELISM_DIM_0 number of dot product instances with DATA_IN_0_TENSOR_SIZE_DIM_0 inputs
  // and each one computes for IN_0_DEPTH iterations for each inputs.
  for (genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_1; i = i + 1) begin : out_dim_1
    for (genvar j = 0; j < WEIGHT_PARALLELISM_DIM_1; j = j + 1) begin : out_dim_0
      // Assume the weight are transposed and partitioned
      logic [WEIGHT_PRECISION_0-1:0] current_mweight[WEIGHT_PARALLELISM_DIM_0-1:0];
      logic [WEIGHT_PRECISION_1-1:0] current_eweight;
      logic [DATA_IN_0_PRECISION_0-1:0] current_mdata[WEIGHT_PARALLELISM_DIM_0-1:0];
      logic [DATA_IN_0_PRECISION_1-1:0] current_edata;
      assign current_mweight = circular_mweight[WEIGHT_PARALLELISM_DIM_0*(j+1)-1:WEIGHT_PARALLELISM_DIM_0*j];
      assign current_eweight = circular_eweight;
      assign current_mdata = circular_mdata_in_0[DATA_IN_0_PARALLELISM_DIM_0*(i+1)-1:DATA_IN_0_PARALLELISM_DIM_0*i];
      assign current_edata = circular_edata_in_0;

      // The inputs are already sync-ed by the previous join
      mxint_dot_product #(
          .DATA_IN_0_PRECISION_0(DATA_IN_0_PRECISION_0),
          .DATA_IN_0_PRECISION_1(DATA_IN_0_PRECISION_1),
          .WEIGHT_PRECISION_0(WEIGHT_PRECISION_0),
          .WEIGHT_PRECISION_1(WEIGHT_PRECISION_1),
          .BLOCK_SIZE(DATA_IN_0_PARALLELISM_DIM_0),
          .DATA_OUT_0_PRECISION_0(FDP_WIDTH),
          .DATA_OUT_0_PRECISION_1(FDP_EXP_WIDTH)
      ) mxdp_inst (
          .clk(clk),
          .rst(rst),
          .mdata_in_0(current_mdata),
          .edata_in_0(current_edata),
          .data_in_0_valid(circular_data_in_0_valid),
          .data_in_0_ready(fdp_data_ready[i*WEIGHT_PARALLELISM_DIM_1+j]),
          .mweight(current_mweight),
          .eweight(current_eweight),
          .weight_valid(circular_weight_valid),
          .weight_ready(fdp_weight_ready[i*WEIGHT_PARALLELISM_DIM_1+j]),
          .mdata_out_0(dp_mdata_out[i*WEIGHT_PARALLELISM_DIM_1+j]),
          .edata_out_0(fdp_edata_out[i*WEIGHT_PARALLELISM_DIM_1+j]),
          .data_out_0_valid(fdp_data_out_valid[i*WEIGHT_PARALLELISM_DIM_1+j]),
          .data_out_0_ready(dp_data_out_ready)
      );
    end
  end
  assign dp_data_out_valid = fdp_data_out_valid[0];
  assign dp_edata_out = fdp_edata_out[0];

  // Insert skid buffer between dot product and accumulator
  mxint_skid_buffer #(
      .DATA_PRECISION_0(FDP_WIDTH),
      .DATA_PRECISION_1(FDP_EXP_WIDTH),
      .IN_NUM(DATA_IN_0_PARALLELISM_DIM_1 * WEIGHT_PARALLELISM_DIM_1)
  ) skid_buffer_inst (
      .clk(clk),
      .rst(rst),
      .mdata_in(dp_mdata_out),
      .edata_in(dp_edata_out),
      .data_in_valid(dp_data_out_valid),
      .data_in_ready(dp_data_out_ready),
      .mdata_out(skid_mdata_out),
      .edata_out(skid_edata_out),
      .data_out_valid(skid_data_out_valid),
      .data_out_ready(skid_data_out_ready)
  );

  mxint_accumulator #(
      .DATA_IN_0_PRECISION_0(FDP_WIDTH),
      .DATA_IN_0_PRECISION_1(FDP_EXP_WIDTH),
      .UNDERFLOW_BITS(UNDERFLOW_BITS),
      .IN_DEPTH(IN_0_DEPTH_DIM_0),
      .BLOCK_SIZE(DATA_OUT_0_PARALLELISM_DIM_1 * DATA_OUT_0_PARALLELISM_DIM_0)
  ) accumulator_inst (
      .clk(clk),
      .rst(rst),
      .mdata_in_0(skid_mdata_out),  // Changed from acc_mdata_in
      .edata_in_0(skid_edata_out),  // Changed from acc_edata_in
      .data_in_0_valid(skid_data_out_valid),  // Changed from acc_data_in_valid
      .data_in_0_ready(skid_data_out_ready),  // Changed from acc_data_in_ready
      .mdata_out_0(acc_mdata_out),
      .edata_out_0(acc_edata_out),
      .data_out_0_valid(acc_data_out_valid),
      .data_out_0_ready(acc_data_out_ready)
  );

  logic [BIAS_PRECISION_0-1:0] mbias_sext[DATA_OUT_0_PARALLELISM_DIM_1 * DATA_OUT_0_PARALLELISM_DIM_0-1:0];
  logic [LOSSLESS_OUT_WIDTH-1:0] shifted_mbias[DATA_OUT_0_PARALLELISM_DIM_1 * DATA_OUT_0_PARALLELISM_DIM_0-1:0];
  logic [FDP_EXP_WIDTH - 1:0] exp_difference;
  logic [FDP_EXP_WIDTH - 1:0] abs_shift_value;
  if (HAS_BIAS) begin : bias_cast
    for (genvar k = 0; k < DATA_OUT_0_PARALLELISM_DIM_1; k++)
      assign mbias_sext[(k+1)*DATA_OUT_0_PARALLELISM_DIM_0 - 1:k*DATA_OUT_0_PARALLELISM_DIM_0] = circular_mbias;
    join2 #() acc_join_inst (
        .data_in_ready ({circular_bias_ready, acc_data_out_ready}),
        .data_in_valid ({circular_bias_valid, acc_data_out_valid}),
        .data_out_valid(cast_data_out_0_valid),
        .data_out_ready(cast_data_out_0_ready)
    );
    assign exp_difference = -($signed(
        circular_ebias
    ) - $signed(
        acc_edata_out
    ) + LOSSLESS_OUT_FRAC_WIDTH - (BIAS_PRECISION_0 - 1));

    optimized_right_shift #(
        .IN_WIDTH(BIAS_PRECISION_0),
        .SHIFT_WIDTH(FDP_EXP_WIDTH),
        .OUT_WIDTH(LOSSLESS_OUT_WIDTH),
        .BLOCK_SIZE(DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1)
    ) ovshift_inst (
        .data_in(mbias_sext),
        .shift_value(exp_difference),
        .data_out(shifted_mbias)
    );
    for (genvar m = 0; m < DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1; m++) begin
      assign cast_mdata_out_0[m] = $signed(shifted_mbias[m]) + $signed(acc_mdata_out[m]);
    end
    assign cast_edata_out_0 = acc_edata_out;
  end else begin
    assign acc_data_out_ready = cast_data_out_0_ready;
    assign cast_data_out_0_valid = acc_data_out_valid;
    assign cast_mdata_out_0 = acc_mdata_out;
    assign cast_edata_out_0 = acc_edata_out;
    assign circular_bias_ready = 1;
  end

    mxint_skid_buffer #(
        .DATA_PRECISION_0(LOSSLESS_OUT_WIDTH),
        .DATA_PRECISION_1(LOSSLESS_OUT_EXP_WIDTH),
        .IN_NUM(DATA_OUT_0_PARALLELISM_DIM_1 * DATA_OUT_0_PARALLELISM_DIM_0)
    ) fifo_reg_inst (
        .clk(clk),
        .rst(rst),
        .mdata_in(cast_mdata_out_0),
        .edata_in(cast_edata_out_0),
        .data_in_valid(cast_data_out_0_valid),
        .data_in_ready(cast_data_out_0_ready),
        .mdata_out(fifo_mdata_in),
        .edata_out(fifo_edata_in),
        .data_out_valid(fifo_data_in_valid),
        .data_out_ready(fifo_data_in_ready)
    );
  // We need a fifo here to make match the throughput between different layers
  unpacked_mx_fifo #(
    .MAN_WIDTH(LOSSLESS_OUT_WIDTH),
    .EXP_WIDTH(LOSSLESS_OUT_EXP_WIDTH),
    .IN_SIZE(DATA_OUT_0_PARALLELISM_DIM_1 * DATA_OUT_0_PARALLELISM_DIM_0),
    .DEPTH(DATA_OUT_0_TENSOR_SIZE_DIM_0 / DATA_OUT_0_PARALLELISM_DIM_0)
  ) cast_fifo (
    .clk(clk),
    .rst(rst),
    .mdata_in(fifo_mdata_in),
    .edata_in(fifo_edata_in),
    .data_in_valid(fifo_data_in_valid),
    .data_in_ready(fifo_data_in_ready),
    .mdata_out(fifo_mdata_out),
    .edata_out(fifo_edata_out),
    .data_out_valid(fifo_data_out_valid),
    .data_out_ready(fifo_data_out_ready)
  );

  // Update cast instance to use FIFO outputs
  mxint_cast #(
      .IN_MAN_WIDTH(LOSSLESS_OUT_WIDTH),
      .IN_MAN_FRAC_WIDTH(LOSSLESS_OUT_FRAC_WIDTH),
      .IN_EXP_WIDTH(LOSSLESS_OUT_EXP_WIDTH),
      .OUT_MAN_WIDTH(DATA_OUT_0_PRECISION_0),
      .OUT_EXP_WIDTH(DATA_OUT_0_PRECISION_1),
      .ROUND_BITS(ROUND_BITS),
      .BLOCK_SIZE(DATA_OUT_0_PARALLELISM_DIM_1 * DATA_OUT_0_PARALLELISM_DIM_0)
  ) cast_i (
      .clk(clk),
      .rst(rst),
      .mdata_in(fifo_mdata_out),  // Changed from skid_mdata_out
      .edata_in(fifo_edata_out),  // Changed from skid_edata_out 
      .data_in_valid(fifo_data_out_valid),  // Changed from skid_data_out_valid
      .data_in_ready(fifo_data_out_ready),  // Changed from skid_data_out_ready
      .mdata_out(mdata_out_0),
      .edata_out(edata_out_0),
      .data_out_valid(data_out_0_valid),
      .data_out_ready(data_out_0_ready)
  );

endmodule

