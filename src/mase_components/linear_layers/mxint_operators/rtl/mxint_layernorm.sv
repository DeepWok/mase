`timescale 1ns / 1ps
module mxint_layernorm #(
    // Dimensions
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 2,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 4,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 2,
    parameter DATA_IN_0_PARALLELISM_DIM_2 = 1,

    // Data widths
    parameter DATA_IN_0_PRECISION_0        = 8,
    parameter DATA_IN_0_PRECISION_1        = 4,
    parameter WEIGHT_PRECISION_0           = 8,
    parameter WEIGHT_PRECISION_1           = 4,
    parameter BIAS_PRECISION_0             = 8,
    parameter BIAS_PRECISION_1             = 4,
    parameter ELEMENTWISE_AFFINE           = 0,
    parameter HAS_BIAS                     = 1,

    parameter ISQRT_IN_PRECISION_0         = 8, //PREICISION_0 for ISQRT is integer width
    parameter ISQRT_IN_PRECISION_1         = 8, //PREICISION_1 for ISQRT is integer frac width
    parameter ISQRT_OUT_PRECISION_0        = 8,
    parameter ISQRT_OUT_PRECISION_1        = 4,
    parameter NORM_OUT_PRECISION_0        = 8,
    parameter NORM_OUT_FRAC_WIDTH         = 4,
    parameter NORM_OUT_PRECISION_1        = 4,

    parameter BIAS_TENSOR_SIZE_DIM_0       = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter BIAS_PARALLELISM_DIM_0       = DATA_IN_0_PARALLELISM_DIM_0,
    parameter BIAS_TENSOR_SIZE_DIM_1       = 1,
    parameter BIAS_PARALLELISM_DIM_1       = 1,
    parameter WEIGHT_TENSOR_SIZE_DIM_0     = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter WEIGHT_PARALLELISM_DIM_0     = DATA_IN_0_PARALLELISM_DIM_0,
    parameter WEIGHT_TENSOR_SIZE_DIM_1     = 1,
    parameter WEIGHT_PARALLELISM_DIM_1     = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_2 = DATA_IN_0_TENSOR_SIZE_DIM_2,
    parameter DATA_OUT_0_PARALLELISM_DIM_2 = DATA_IN_0_PARALLELISM_DIM_2,
    parameter DATA_OUT_0_PRECISION_0       = 8,
    parameter DATA_OUT_0_PRECISION_1       = 4
) (
    input logic clk,
    input logic rst,

    input  logic [DATA_IN_0_PRECISION_0-1:0] mdata_in_0 [DATA_IN_0_PARALLELISM_DIM_1*DATA_IN_0_PARALLELISM_DIM_0-1:0],
    input  logic [DATA_IN_0_PRECISION_1-1:0] edata_in_0,
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    input  logic [WEIGHT_PRECISION_0-1:0] mweight      [DATA_IN_0_PARALLELISM_DIM_0 - 1 : 0],
    input  logic [WEIGHT_PRECISION_1-1:0] eweight,
    input  logic                          weight_valid,
    output logic                          weight_ready,

    input  logic [BIAS_PRECISION_0-1:0] mbias      [DATA_IN_0_PARALLELISM_DIM_0 - 1 : 0],
    input  logic [BIAS_PRECISION_1-1:0] ebias,
    input  logic                        bias_valid,
    output logic                        bias_ready,

    output logic [DATA_OUT_0_PRECISION_0-1:0] mdata_out_0 [DATA_IN_0_PARALLELISM_DIM_1*DATA_IN_0_PARALLELISM_DIM_0-1:0],
    output logic [DATA_OUT_0_PRECISION_1-1:0] edata_out_0,
    output logic data_out_0_valid,
    input logic data_out_0_ready
);

  localparam AFFINE_PRECISION_0 = DATA_OUT_0_PRECISION_0 + WEIGHT_PRECISION_0 + 1;
  localparam AFFINE_PRECISION_1 = DATA_OUT_0_PRECISION_1 + WEIGHT_PRECISION_1;

  localparam WD_PRECISION_0 = NORM_OUT_PRECISION_0 + WEIGHT_PRECISION_0 + 1;
  localparam WD_MAN_FRAC_WIDTH = NORM_OUT_PRECISION_1 + WEIGHT_PRECISION_0 - 1;
  localparam WD_PRECISION_1 = NORM_OUT_PRECISION_1 + 1;

  logic [NORM_OUT_PRECISION_0 - 1:0] mnorm_out [DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1 - 1:0];
  logic [NORM_OUT_PRECISION_1 - 1:0] enorm_out;
  logic [DATA_IN_0_PARALLELISM_DIM_1 - 1:0] parallel_norm_in_valid, parallel_norm_in_ready;
  logic [DATA_IN_0_PARALLELISM_DIM_1 - 1:0] parallel_norm_out_valid, parallel_norm_out_ready;
  logic [DATA_OUT_0_PRECISION_1 - 1:0] parallel_enorm_out [DATA_IN_0_PARALLELISM_DIM_1 - 1:0];
  logic norm_out_valid, norm_out_ready;
  logic [AFFINE_PRECISION_0 -1:0] uncast_data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1 - 1:0];

  logic [BIAS_PRECISION_0-1:0] mbias_buffered  [DATA_IN_0_PARALLELISM_DIM_0 - 1 : 0];
  logic [BIAS_PRECISION_1-1:0] ebias_buffered;
  logic bias_buffered_valid, bias_buffered_ready;

  logic [WEIGHT_PRECISION_0-1:0]     mweight_buffered [DATA_IN_0_PARALLELISM_DIM_0 - 1 : 0];
  logic [WEIGHT_PRECISION_1-1:0]     eweight_buffered;
  logic                              weight_buffered_ready, weight_buffered_valid;

  logic [WD_PRECISION_0 - 1:0]       mwd_out          [DATA_OUT_0_PARALLELISM_DIM_1*DATA_OUT_0_PARALLELISM_DIM_0 - 1 : 0];
  logic [WD_PRECISION_1 - 1:0]       ewd_out;
  logic                              wd_out_valid, wd_out_ready;

  logic                              affine_out_ready, affine_out_valid;
  localparam SHIFT_WIDTH = WEIGHT_PRECISION_1 + 2;
  logic [SHIFT_WIDTH - 1:0]   shift_value;
  logic [WD_PRECISION_0 - 1:0]       casted_bias      [DATA_OUT_0_PARALLELISM_DIM_0 - 1:0];

  logic [WD_PRECISION_0 - 1:0]       maffine_out     [DATA_OUT_0_PARALLELISM_DIM_1*DATA_OUT_0_PARALLELISM_DIM_0 - 1:0];
  logic [WD_PRECISION_1 - 1:0]       eaffine_out;

  for (genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_1; i++) begin : parallel_dim_1
    assign parallel_norm_in_valid[i] = data_in_0_valid;
    assign parallel_norm_out_ready[i] = norm_out_ready;
    mxint_layernorm_1d #(
        .DATA_IN_0_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0),
        .DATA_IN_0_PARALLELISM_DIM_0(DATA_IN_0_PARALLELISM_DIM_0),
        .DATA_IN_0_MAN_WIDTH(DATA_IN_0_PRECISION_0),
        .DATA_IN_0_EXP_WIDTH(DATA_IN_0_PRECISION_1),
        .ISQRT_IN_MAN_WIDTH(ISQRT_IN_PRECISION_0),
        .ISQRT_IN_MAN_FRAC_WIDTH(ISQRT_IN_PRECISION_1),
        .ISQRT_OUT_MAN_WIDTH(ISQRT_OUT_PRECISION_0),
        .ISQRT_OUT_MAN_FRAC_WIDTH(ISQRT_OUT_PRECISION_1),
        .DATA_OUT_0_MAN_WIDTH(NORM_OUT_PRECISION_0),
        .DATA_OUT_0_MAN_FRAC_WIDTH(NORM_OUT_FRAC_WIDTH),
        .DATA_OUT_0_EXP_WIDTH(NORM_OUT_PRECISION_1)
    ) layer_norm_inst (
        .clk,
        .rst,
        .mdata_in_0(mdata_in_0[i*DATA_IN_0_PARALLELISM_DIM_0 + DATA_IN_0_PARALLELISM_DIM_0 - 1:  i*DATA_IN_0_PARALLELISM_DIM_0]),
        .edata_in_0(edata_in_0),
        .data_in_0_valid(parallel_norm_in_valid[i]),
        .data_in_0_ready(parallel_norm_in_ready[i]),
        .mdata_out_0(mnorm_out[i*DATA_IN_0_PARALLELISM_DIM_0 + DATA_IN_0_PARALLELISM_DIM_0 - 1:  i*DATA_IN_0_PARALLELISM_DIM_0]),
        .edata_out_0(parallel_enorm_out[i]),
        .data_out_0_valid(parallel_norm_out_valid[i]),
        .data_out_0_ready(parallel_norm_out_ready[i])
    );
  end
  //TODO: Bug here, notice, our module currently can only support parallel in the dimension 0;
  assign enorm_out = parallel_enorm_out[0];
  assign data_in_0_ready = parallel_norm_in_ready[0];
  assign norm_out_valid  = parallel_norm_out_valid[0];

  if (ELEMENTWISE_AFFINE == 1) begin
    mxint_circular #(
      .DATA_PRECISION_0(BIAS_PRECISION_0),
      .DATA_PRECISION_1(BIAS_PRECISION_1),
      .IN_NUM(DATA_IN_0_PARALLELISM_DIM_0),
      .REPEAT(DATA_IN_0_TENSOR_SIZE_DIM_1 / DATA_IN_0_PARALLELISM_DIM_1),
      .BUFFER_SIZE(DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0)
    ) bias_buffer_inst (
        .clk(clk),
        .rst(rst),
        .mdata_in(mbias),
        .edata_in(ebias),
        .data_in_valid(bias_valid),
        .data_in_ready(bias_ready),
        .mdata_out(mbias_buffered),
        .edata_out(ebias_buffered),
        .data_out_valid(bias_buffered_valid),
        .data_out_ready(bias_buffered_ready)
    );

    mxint_circular #(
        .DATA_PRECISION_0(WEIGHT_PRECISION_0),
        .DATA_PRECISION_1(WEIGHT_PRECISION_1),
        .IN_NUM(DATA_IN_0_PARALLELISM_DIM_0),
        .REPEAT(DATA_IN_0_TENSOR_SIZE_DIM_1 / DATA_IN_0_PARALLELISM_DIM_1),
        .BUFFER_SIZE(DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0)
    ) weight_buffer_inst (
        .clk(clk),
        .rst(rst),
        .mdata_in(mweight),
        .edata_in(eweight),
        .data_in_valid(weight_valid),
        .data_in_ready(weight_ready),
        .mdata_out(mweight_buffered),
        .edata_out(eweight_buffered),
        .data_out_valid(weight_buffered_valid),
        .data_out_ready(weight_buffered_ready)
    );

    join2 weight_data_join_inst (
        .data_in_valid ({weight_buffered_valid, norm_out_valid}),
        .data_in_ready ({weight_buffered_ready, norm_out_ready}),
        .data_out_valid(wd_out_valid),
        .data_out_ready(wd_out_ready)
    );

    for(genvar i = 0; i < DATA_OUT_0_PARALLELISM_DIM_1; i++) begin : affine_bias_parallel
        for(genvar j = 0; j < DATA_OUT_0_PARALLELISM_DIM_0; j++) begin : affine_bias_parallel
            localparam int k = i * DATA_IN_0_PARALLELISM_DIM_0 + j;
            assign mwd_out[k] = $signed(mweight_buffered[j]) * $signed(mnorm_out[k]);
        end
    end
    assign ewd_out = $signed(eweight_buffered) + $signed(enorm_out);

    join2 wd_bias_join_inst (
        .data_in_valid ({wd_out_valid, bias_buffered_valid}),
        .data_in_ready ({wd_out_ready, bias_buffered_ready}),
        .data_out_valid(affine_out_valid),
        .data_out_ready(affine_out_ready)
    );
    localparam MWD_FRAC_WIDTH = NORM_OUT_FRAC_WIDTH + WEIGHT_PRECISION_0 - 1;
    localparam BIAS_FRAC_WIDTH = BIAS_PRECISION_0 - 1;
    assign shift_value = $signed(ewd_out) - $signed(ebias_buffered) - MWD_FRAC_WIDTH + BIAS_FRAC_WIDTH;
    optimized_right_shift #(
        .IN_WIDTH(BIAS_PRECISION_0),
        .SHIFT_WIDTH(SHIFT_WIDTH),
        .OUT_WIDTH(WD_PRECISION_0),
        .BLOCK_SIZE(DATA_OUT_0_PARALLELISM_DIM_0)
    ) ovshift_inst (
        .data_in(mbias_buffered),
        .shift_value(shift_value),
        .data_out(casted_bias)
    );
    for(genvar i = 0; i < DATA_OUT_0_PARALLELISM_DIM_1; i++) begin : affine_bias_parallel_dim_1
        for(genvar j = 0; j < DATA_OUT_0_PARALLELISM_DIM_0; j++) begin : affine_bias_parallel_dim_0
            localparam int k = i * DATA_IN_0_PARALLELISM_DIM_0 + j;
            assign maffine_out[k] = $signed(casted_bias[j]) + $signed(mwd_out[k]);
        end
    end
    assign eaffine_out = ewd_out;

    mxint_cast #(
        .IN_MAN_WIDTH(WD_PRECISION_0),
        .IN_MAN_FRAC_WIDTH(WD_MAN_FRAC_WIDTH),
        .IN_EXP_WIDTH(WD_PRECISION_0),
        .OUT_MAN_WIDTH(DATA_OUT_0_PRECISION_0),
        .OUT_EXP_WIDTH(DATA_OUT_0_PRECISION_1),
        .BLOCK_SIZE(DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1)
    ) u_mxint_cast (
        .clk(clk),
        .rst(rst),
        .mdata_in(maffine_out),
        .edata_in(eaffine_out),
        .data_in_valid(affine_out_valid),
        .data_in_ready(affine_out_ready),
        .mdata_out(mdata_out_0),
        .edata_out(edata_out_0),
        .data_out_valid(data_out_0_valid),
        .data_out_ready(data_out_0_ready)
    );
  end else begin
    mxint_cast #(
        .IN_MAN_WIDTH(NORM_OUT_PRECISION_0),
        .IN_MAN_FRAC_WIDTH(NORM_OUT_PRECISION_1),
        .IN_EXP_WIDTH(DATA_OUT_0_PRECISION_1),
        .OUT_MAN_WIDTH(DATA_OUT_0_PRECISION_0),
        .OUT_EXP_WIDTH(DATA_OUT_0_PRECISION_1),
        .BLOCK_SIZE(DATA_OUT_0_PARALLELISM_DIM_1*DATA_OUT_0_PARALLELISM_DIM_0)
    ) u_mxint_cast (
        .clk(clk),
        .rst(rst),
        .mdata_in(mnorm_out),
        .edata_in(enorm_out),
        .data_in_valid(norm_out_valid),
        .data_in_ready(norm_out_ready),
        .mdata_out(mdata_out_0),
        .edata_out(edata_out_0),
        .data_out_valid(data_out_0_valid),
        .data_out_ready(data_out_0_ready)
    );
  end
endmodule
