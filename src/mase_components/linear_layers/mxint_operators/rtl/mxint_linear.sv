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

    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 20,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 20,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 4,  // must equal WEIGHT_PARALLELISM_DIM_1
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 4,
    localparam IN_0_DEPTH_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0,
    localparam IN_0_DEPTH_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1 / DATA_IN_0_PARALLELISM_DIM_1,

    parameter WEIGHT_PRECISION_0 = 16,
    parameter WEIGHT_PRECISION_1 = 3,
    parameter WEIGHT_TENSOR_SIZE_DIM_0 = 20,
    parameter WEIGHT_TENSOR_SIZE_DIM_1 = 20,
    parameter WEIGHT_PARALLELISM_DIM_0 = 4,
    parameter WEIGHT_PARALLELISM_DIM_1 = 4,

    // Inferred precision of the output data
    // if the data out precision will be replaced by the setting
    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
        DATA_IN_0_TENSOR_SIZE_DIM_0
    ) + HAS_BIAS,
    parameter DATA_OUT_0_PRECISION_1 = DATA_IN_0_PRECISION_1 + WEIGHT_PRECISION_1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = WEIGHT_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = WEIGHT_PARALLELISM_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,

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
  initial begin
    assert (DATA_IN_0_PARALLELISM_DIM_0 == WEIGHT_PARALLELISM_DIM_1);

    assert (WEIGHT_PARALLELISM_DIM_1 == WEIGHT_PARALLELISM_DIM_0);

    assert ((DATA_IN_0_TENSOR_SIZE_DIM_0 % DATA_IN_0_PARALLELISM_DIM_0) == 0);
    assert ((DATA_IN_0_TENSOR_SIZE_DIM_1 % DATA_IN_0_PARALLELISM_DIM_1) == 0);
  end

  logic [DATA_IN_0_PRECISION_0-1:0]circular_mdata_in_0[DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [DATA_IN_0_PRECISION_1-1:0] circular_edata_in_0;
  logic circular_data_in_0_valid, circular_data_in_0_ready;

  logic [WEIGHT_PRECISION_0-1:0]circular_mweight[WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1-1:0];
  logic [WEIGHT_PRECISION_1-1:0] circular_eweight;
  logic circular_weight_valid, circular_weight_ready;

  logic [BIAS_PRECISION_0-1:0] circular_mbias [BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1-1:0];
  logic [BIAS_PRECISION_1-1:0] circular_ebias;
  logic circular_bias_valid, circular_bias_ready;
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
  localparam FDP_WIDTH = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
      DATA_IN_0_PARALLELISM_DIM_0
  );
  localparam FDP_EXP_WIDTH_IN =(WEIGHT_PRECISION_1 > DATA_IN_0_PRECISION_1)? WEIGHT_PRECISION_1 + 1: DATA_IN_0_PRECISION_1 + 1;
  localparam FDP_EXP_WIDTH_OUT = FDP_EXP_WIDTH_IN + $clog2($clog2(IN_0_DEPTH_DIM_0 + HAS_BIAS) + 1);
  localparam ACC_WIDTH = FDP_WIDTH + $clog2(IN_0_DEPTH_DIM_0 + HAS_BIAS) + 2 ** FDP_EXP_WIDTH_IN;
  localparam LOSSLESS_OUT_WIDTH = ACC_WIDTH;
  localparam LOSSLESS_OUT_EXP_WIDTH = FDP_EXP_WIDTH_OUT;
  /* verilator lint_off UNUSEDSIGNAL */
  // Assume the parallelised hardware above have the same arrival time
  // which means that they always have the same state. So we can just
  // pick one of the valid signal to use.
  logic [DATA_IN_0_PARALLELISM_DIM_1*WEIGHT_PARALLELISM_DIM_1-1:0]
      fdp_data_ready, fdp_weight_ready;
  assign circular_weight_ready = fdp_weight_ready[0];
  assign circular_data_in_0_ready = fdp_data_ready[0];
  logic [FDP_EXP_WIDTH_IN-1:0] fdp_edata_out [DATA_IN_0_PARALLELISM_DIM_1 * WEIGHT_PARALLELISM_DIM_1 - 1:0];
  logic [DATA_IN_0_PARALLELISM_DIM_1 * WEIGHT_PARALLELISM_DIM_1 - 1:0] fdp_data_out_valid;

  logic [FDP_WIDTH-1:0] acc_mdata_in[DATA_IN_0_PARALLELISM_DIM_1 * WEIGHT_PARALLELISM_DIM_1 - 1:0];
  logic [FDP_EXP_WIDTH_IN-1:0] acc_edata_in;
  logic acc_data_in_valid, acc_data_in_ready;

  logic [         ACC_WIDTH-1:0] acc_mdata_out   [DATA_IN_0_PARALLELISM_DIM_1 * DATA_OUT_0_PARALLELISM_DIM_0-1:0];
  logic [FDP_EXP_WIDTH_OUT-1:0] acc_edata_out;
  logic acc_data_out_valid, acc_data_out_ready;
  logic [LOSSLESS_OUT_WIDTH-1:0] cast_mdata_out_0[DATA_OUT_0_PARALLELISM_DIM_1 * DATA_OUT_0_PARALLELISM_DIM_0-1:0];
  logic [LOSSLESS_OUT_EXP_WIDTH-1:0] cast_edata_out_0;
  logic cast_data_out_0_valid, cast_data_out_0_ready;
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
          .BLOCK_SIZE(DATA_IN_0_PARALLELISM_DIM_0)
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
          .mdata_out_0(acc_mdata_in[i*WEIGHT_PARALLELISM_DIM_1+j]),
          .edata_out_0(fdp_edata_out[i*WEIGHT_PARALLELISM_DIM_1+j]),
          .data_out_0_valid(fdp_data_out_valid[i*WEIGHT_PARALLELISM_DIM_1+j]),
          .data_out_0_ready(acc_data_in_ready)
      );
    end
  end
  assign acc_data_in_valid = fdp_data_out_valid[0];
  assign acc_edata_in = fdp_edata_out[0];

  localparam FDP_EXP_BIAS = 2 ** (FDP_EXP_WIDTH_IN - 1) - 1;
  localparam BIAS_EXP_BIAS = 2 ** (BIAS_PRECISION_1 - 1) - 1;

  logic [$clog2(IN_0_DEPTH_DIM_0 + HAS_BIAS):0] accum_count;
  logic acc_ready;
  logic acc_valid;
  logic add_bias;

  logic [FDP_WIDTH-1:0] acc_mdata[DATA_IN_0_PARALLELISM_DIM_1 * WEIGHT_PARALLELISM_DIM_1 - 1:0];
  logic [FDP_EXP_WIDTH_IN-1:0] acc_edata;
  logic acc_data_valid, acc_data_ready;

  mxint_accumulator #(
      .DATA_IN_0_PRECISION_0(FDP_WIDTH),
      .DATA_IN_0_PRECISION_1(FDP_EXP_WIDTH_IN),
      .IN_DEPTH             (IN_0_DEPTH_DIM_0 + HAS_BIAS),
      .BLOCK_SIZE           (DATA_OUT_0_PARALLELISM_DIM_1 * DATA_OUT_0_PARALLELISM_DIM_0)
  ) accumulator_inst (
      .clk             (clk),
      .rst             (rst),
      .mdata_in_0      (acc_mdata),
      .edata_in_0      (acc_edata),
      .data_in_0_valid (acc_valid),
      .data_in_0_ready (acc_ready),
      .mdata_out_0     (acc_mdata_out),
      .edata_out_0     (acc_edata_out),
      .data_out_0_valid(acc_data_out_valid),
      .data_out_0_ready(acc_data_out_ready),
      .accum_count     (accum_count)
  );

  if (HAS_BIAS) begin : bias_cast

    assign add_bias = accum_count == IN_0_DEPTH_DIM_0;
    assign circular_bias_ready = add_bias && acc_ready;
    assign acc_data_in_ready = !add_bias && acc_ready;
    assign acc_valid = add_bias ? circular_bias_valid : acc_data_in_valid;
    assign acc_edata = add_bias ? circular_ebias - BIAS_EXP_BIAS + FDP_EXP_BIAS : acc_edata_in;

    for (genvar j = 0; j < DATA_IN_0_PARALLELISM_DIM_1; j++) begin
      for (genvar i = 0; i < BIAS_PARALLELISM_DIM_0; i++) begin
        assign acc_mdata[j*DATA_IN_0_PARALLELISM_DIM_0+i] = add_bias ? {circular_mbias[i], {(FDP_WIDTH - BIAS_PRECISION_0){1'b0}}} : acc_mdata_in[j*DATA_IN_0_PARALLELISM_DIM_0+i];
      end
    end

  end else begin
    assign acc_mdata         = acc_mdata_in;
    assign acc_edata         = acc_edata_in;
    assign acc_data_in_ready = acc_ready;
    assign acc_valid         = acc_data_in_valid;
  end

  always_comb begin
    for (int i = 0; i < DATA_OUT_0_PARALLELISM_DIM_1 * DATA_OUT_0_PARALLELISM_DIM_0; i++) begin
      cast_mdata_out_0[i] = acc_mdata_out[i];
    end
  end

  assign acc_data_out_ready    = cast_data_out_0_ready;
  assign cast_data_out_0_valid = acc_data_out_valid;
  assign cast_edata_out_0      = acc_edata_out;
  assign bias_ready            = 1;

  mxint_cast #(
      .IN_MAN_WIDTH(LOSSLESS_OUT_WIDTH),
      .IN_EXP_WIDTH(LOSSLESS_OUT_EXP_WIDTH),
      .OUT_MAN_WIDTH(DATA_OUT_0_PRECISION_0),
      .OUT_EXP_WIDTH(DATA_OUT_0_PRECISION_1),
      .BLOCK_SIZE(DATA_OUT_0_PARALLELISM_DIM_1 * DATA_OUT_0_PARALLELISM_DIM_0)
  ) cast_i (
      .clk,
      .rst,
      .mdata_in(cast_mdata_out_0),
      .edata_in(cast_edata_out_0),
      .data_in_valid(cast_data_out_0_valid),
      .data_in_ready(cast_data_out_0_ready),
      .mdata_out(mdata_out_0),
      .edata_out(edata_out_0),
      .data_out_valid(data_out_0_valid),
      .data_out_ready(data_out_0_ready)
  );

endmodule

