`timescale 1ns / 1ps
module mxint_patch_embed #(
    // 
    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter CONV_WEIGHT_PRECISION_0    = 8,
    parameter CONV_WEIGHT_PRECISION_1    = 4,
    parameter CONV_BIAS_PRECISION_0      = 8,
    parameter CONV_BIAS_PRECISION_1      = 4,

    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 224,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 224,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = 3,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_2 = 3,

    parameter CONV_WEIGHT_TENSOR_SIZE_DIM_0 = 224,
    parameter CONV_WEIGHT_TENSOR_SIZE_DIM_1 = 224,
    parameter CONV_WEIGHT_PARALLELISM_DIM_0 = 1,
    parameter CONV_WEIGHT_PARALLELISM_DIM_1 = 1,

    parameter CONV_BIAS_TENSOR_SIZE_DIM_0 = 224,
    parameter CONV_BIAS_TENSOR_SIZE_DIM_1 = 224,
    parameter CONV_BIAS_PARALLELISM_DIM_0 = 1,
    parameter CONV_BIAS_PARALLELISM_DIM_1 = 1,

    parameter CLS_TOKEN_PRECISION_0 = DATA_IN_0_PRECISION_0,
    parameter CLS_TOKEN_PRECISION_1 = DATA_IN_0_PRECISION_1,

    parameter DISTILL_TOKEN_PRECISION_0 = DATA_IN_0_PRECISION_0,
    parameter DISTILL_TOKEN_PRECISION_1 = DATA_IN_0_PRECISION_1,

    parameter CLS_TOKEN_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter CLS_TOKEN_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter CLS_TOKEN_TENSOR_SIZE_DIM_2 = DATA_IN_0_TENSOR_SIZE_DIM_2,

    parameter CLS_TOKEN_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,
    parameter CLS_TOKEN_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,
    parameter CLS_TOKEN_PARALLELISM_DIM_2 = DATA_IN_0_PARALLELISM_DIM_2,

    parameter DISTILL_TOKEN_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter DISTILL_TOKEN_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter DISTILL_TOKEN_TENSOR_SIZE_DIM_2 = DATA_IN_0_TENSOR_SIZE_DIM_2,

    parameter DISTILL_TOKEN_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,
    parameter DISTILL_TOKEN_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,
    parameter DISTILL_TOKEN_PARALLELISM_DIM_2 = DATA_IN_0_PARALLELISM_DIM_2,

    parameter PATCH_SIZE = 16,

    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 192,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 4,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = (DATA_IN_0_TENSOR_SIZE_DIM_0 * DATA_IN_0_TENSOR_SIZE_DIM_1) / (PATCH_SIZE*PATCH_SIZE),
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_2 = 1,

    parameter IN_X = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter IN_Y = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter IN_C = DATA_IN_0_TENSOR_SIZE_DIM_2,

    parameter KERNEL_X = PATCH_SIZE,
    parameter KERNEL_Y = PATCH_SIZE,
    parameter OUT_C = DATA_OUT_0_TENSOR_SIZE_DIM_0,
    parameter UNROLL_OUT_C = DATA_OUT_0_PARALLELISM_DIM_0,

    parameter BIAS_SIZE = UNROLL_OUT_C,

    parameter HAS_BIAS = 1,

    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 4
) (
    input clk,
    input rst,

    input  [DATA_IN_0_PRECISION_0 - 1:0] mdata_in_0      [IN_C - 1 : 0],
    input [DATA_IN_0_PRECISION_1 - 1:0] edata_in_0,
    input                                data_in_0_valid,
    output                               data_in_0_ready,

    input  [CONV_WEIGHT_PRECISION_0-1:0] mconv_weight      [UNROLL_OUT_C * IN_C -1:0],
    input [CONV_WEIGHT_PRECISION_1-1:0] econv_weight,
    input                           conv_weight_valid,
    output                          conv_weight_ready,

    input  [CONV_BIAS_PRECISION_0-1:0] mconv_bias      [UNROLL_OUT_C-1:0],
    input [CONV_BIAS_PRECISION_1-1:0] econv_bias,
    input                         conv_bias_valid,
    output                        conv_bias_ready,

    input  [DATA_OUT_0_PRECISION_0 - 1:0] mcls_token      [UNROLL_OUT_C - 1 : 0],
    input [DATA_OUT_0_PRECISION_1 - 1:0] ecls_token,
    input                                cls_token_valid,
    output  logic                               cls_token_ready,

    input  [DATA_OUT_0_PRECISION_0 - 1:0] mdistill_token      [UNROLL_OUT_C - 1 : 0],
    input [DATA_OUT_0_PRECISION_1 - 1:0] edistill_token,
    input                                distill_token_valid,
    output logic                                distill_token_ready,

    output logic [DATA_OUT_0_PRECISION_0 - 1:0] mdata_out_0      [UNROLL_OUT_C - 1:0],
    output logic [DATA_OUT_0_PRECISION_1 - 1:0] edata_out_0,
    output logic                                data_out_0_valid,
    input  logic                                 data_out_0_ready
);
    localparam OUT_Y = (IN_Y) / (KERNEL_Y);
    localparam OUT_X = (IN_X) / (KERNEL_X);
    localparam SLIDING_NUM = OUT_Y * OUT_X;
    localparam MAXIMUM_OUT = (IN_X * IN_Y / (KERNEL_X * KERNEL_Y) + 2)* (OUT_C / UNROLL_OUT_C);
    localparam COUNT_WIDTH = $clog2(MAXIMUM_OUT);

    logic [CONV_WEIGHT_PRECISION_0-1:0] circular_mweight      [UNROLL_OUT_C * IN_C -1:0];
    logic [CONV_WEIGHT_PRECISION_1-1:0] circular_eweight;
    logic circular_weight_valid;
    logic circular_weight_ready;

    logic [CONV_BIAS_PRECISION_0-1:0] circular_mbias      [UNROLL_OUT_C-1:0];
    logic [CONV_BIAS_PRECISION_1-1:0] circular_ebias;
    logic circular_bias_valid;
    logic circular_bias_ready;

    logic [COUNT_WIDTH - 1:0] count;

    enum {CLS_TOKEN, DISTILL_TOKEN, CONV_OUT} state;

    logic [DATA_OUT_0_PRECISION_0 - 1:0] mconv_out      [UNROLL_OUT_C - 1:0];
    logic [DATA_OUT_0_PRECISION_1 - 1:0] econv_out;
    logic conv_out_valid;
    logic conv_out_ready;
  mxint_circular #(
      .DATA_PRECISION_0(CONV_WEIGHT_PRECISION_0),
      .DATA_PRECISION_1(CONV_WEIGHT_PRECISION_1),
      .IN_NUM          (UNROLL_OUT_C * IN_C),
      .REPEAT          (SLIDING_NUM),
      .BUFFER_SIZE     (OUT_C / UNROLL_OUT_C)
  ) weight_buffer (
      .clk,
      .rst,
      // Input streaming port
      .mdata_in(mconv_weight),
      .edata_in(econv_weight),
      .data_in_valid(conv_weight_valid),
      .data_in_ready(conv_weight_ready),
      // Output streaming port
      .mdata_out(circular_mweight),
      .edata_out(circular_eweight),
      .data_out_valid(circular_weight_valid),
      .data_out_ready(circular_weight_ready)
  );
  mxint_circular #(
      .DATA_PRECISION_0(CONV_BIAS_PRECISION_0),
      .DATA_PRECISION_1(CONV_BIAS_PRECISION_1),
      .IN_NUM          (UNROLL_OUT_C),
      .REPEAT          (SLIDING_NUM),
      .BUFFER_SIZE     (OUT_C / UNROLL_OUT_C)
  ) bias_buffer (
      .clk,
      .rst,
      // Input streaming port
      .mdata_in(mconv_bias),
      .edata_in(econv_bias),
      .data_in_valid(conv_bias_valid),
      .data_in_ready(conv_bias_ready),
      // Output streaming port
      .mdata_out(circular_mbias),
      .edata_out(circular_ebias),
      .data_out_valid(circular_bias_valid),
      .data_out_ready(circular_bias_ready)
  );

mxint_patch_embed_conv #(
    .DATA_IN_0_PRECISION_0(DATA_IN_0_PRECISION_0),
    .DATA_IN_0_PRECISION_1(DATA_IN_0_PRECISION_1),
    .WEIGHT_PRECISION_0(CONV_WEIGHT_PRECISION_0),
    .WEIGHT_PRECISION_1(CONV_WEIGHT_PRECISION_1),
    .BIAS_PRECISION_0(CONV_BIAS_PRECISION_0),
    .BIAS_PRECISION_1(CONV_BIAS_PRECISION_1),
    .IN_X(IN_X),
    .IN_Y(IN_Y),
    .IN_C(IN_C),
    .KERNEL_X(KERNEL_X),
    .KERNEL_Y(KERNEL_Y),
    .OUT_C(OUT_C),
    .UNROLL_OUT_C(UNROLL_OUT_C),
    .HAS_BIAS(HAS_BIAS),
    .DATA_OUT_0_PRECISION_0(DATA_OUT_0_PRECISION_0),
    .DATA_OUT_0_PRECISION_1(DATA_OUT_0_PRECISION_1)
) conv_inst (
    .clk(clk),
    .rst(rst),
    .mdata_in_0(mdata_in_0),
    .edata_in_0(edata_in_0),
    .data_in_0_valid(data_in_0_valid),
    .data_in_0_ready(data_in_0_ready),
    .mweight(circular_mweight),
    .eweight(circular_eweight),
    .weight_valid(circular_weight_valid),
    .weight_ready(circular_weight_ready),
    .mbias(circular_mbias),
    .ebias(circular_ebias),
    .bias_valid(circular_bias_valid),
    .bias_ready(circular_bias_ready),
    .mdata_out_0(mconv_out),
    .edata_out_0(econv_out),
    .data_out_0_valid(conv_out_valid),
    .data_out_0_ready(conv_out_ready)
);

    always_ff @(posedge clk) begin
        if (rst) count <= 0;
        else if (data_out_0_valid && data_out_0_ready) 
            if (count == MAXIMUM_OUT - 1) count <= 0;
            else count <= count + 1;
        else count <= count;
    end
    
    always_comb begin
        if (count < OUT_C/UNROLL_OUT_C) state = CLS_TOKEN;
        else if (count < 2 * OUT_C/UNROLL_OUT_C) state = DISTILL_TOKEN;
        else state = CONV_OUT;
        case (state)
            CLS_TOKEN: begin
                mdata_out_0 = mcls_token;
                edata_out_0 = ecls_token;
                data_out_0_valid = cls_token_valid;
                cls_token_ready = data_out_0_ready;
                distill_token_ready = 0;
                conv_out_ready = 0;
            end
            DISTILL_TOKEN: begin
                mdata_out_0 = mdistill_token;
                edata_out_0 = edistill_token;
                data_out_0_valid = distill_token_valid;
                cls_token_ready = 0;
                distill_token_ready = data_out_0_ready;
                conv_out_ready = 0;
            end
            CONV_OUT: begin
                mdata_out_0 = mconv_out;
                edata_out_0 = econv_out;
                data_out_0_valid = conv_out_valid;
                cls_token_ready = 0;
                distill_token_ready = 0;
                conv_out_ready = data_out_0_ready;
            end
            default: begin
                mdata_out_0 = '{default:0};
                edata_out_0 = '0;
                data_out_0_valid = 0;
                cls_token_ready = 0;
                distill_token_ready = 0;
                conv_out_ready = 0;
            end
        endcase
    end

endmodule

module mxint_patch_embed_conv #(
    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter WEIGHT_PRECISION_0    = 8,
    parameter WEIGHT_PRECISION_1    = 4,
    parameter BIAS_PRECISION_0      = 8,
    parameter BIAS_PRECISION_1      = 4,

    parameter IN_X    = 3,
    parameter IN_Y   = 2,
    parameter IN_C = 4,

    parameter KERNEL_X = 2,
    parameter KERNEL_Y = 2,
    parameter OUT_C = 4,

    parameter UNROLL_OUT_C = 2,

    parameter BIAS_SIZE = UNROLL_OUT_C,

    parameter HAS_BIAS  = 1,

    parameter OUT_Y = (IN_Y) / (KERNEL_Y),
    parameter OUT_X = (IN_X) / (KERNEL_X),
    parameter SLIDING_NUM = OUT_Y * OUT_X,

    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 4
) (
    input clk,
    input rst,

    input  [DATA_IN_0_PRECISION_0 - 1:0] mdata_in_0      [IN_C - 1 : 0],
    input [DATA_IN_0_PRECISION_1 - 1:0] edata_in_0,
    input                                data_in_0_valid,
    output                               data_in_0_ready,

    input  [WEIGHT_PRECISION_0-1:0] mweight      [UNROLL_OUT_C * IN_C -1:0],
    input [WEIGHT_PRECISION_1-1:0] eweight,
    input                           weight_valid,
    output                          weight_ready,

    input  [BIAS_PRECISION_0-1:0] mbias      [UNROLL_OUT_C-1:0],
    input [BIAS_PRECISION_1-1:0] ebias,
    input                         bias_valid,
    output                        bias_ready,

    output [DATA_OUT_0_PRECISION_0 - 1:0] mdata_out_0      [UNROLL_OUT_C - 1:0],
    output [DATA_OUT_0_PRECISION_1 - 1:0] edata_out_0,
    output                                data_out_0_valid,
    input                                 data_out_0_ready
);
  initial begin
    assert (
        (KERNEL_X==KERNEL_Y)
    ) else $fatal("UNROLL parameter not set correctly");
  end

  localparam STRIDE = KERNEL_X;
  localparam UNCAST_OUT_WIDTH = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
      KERNEL_Y * KERNEL_X * IN_C
  ) + 1;
  localparam UNCAST_OUT_FRAC_WIDTH = DATA_IN_0_PRECISION_1 + WEIGHT_PRECISION_1;
  localparam ROUND_PRECISION_0 = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
      KERNEL_X * KERNEL_Y * IN_C
  );
  localparam ROUND_PRECISION_1 = DATA_IN_0_PRECISION_1 + WEIGHT_PRECISION_1;

  logic [DATA_IN_0_PRECISION_0 * IN_C  + DATA_IN_0_PRECISION_1 - 1:0] packed_data_in;
  logic [UNCAST_OUT_WIDTH - 1:0] uncast_data_out[UNROLL_OUT_C - 1:0];

  logic [DATA_IN_0_PRECISION_0 * IN_C + DATA_IN_0_PRECISION_1 - 1:0] packed_kernel[KERNEL_Y * KERNEL_X - 1:0];
  logic kernel_valid;
  logic kernel_ready;

  logic [DATA_IN_0_PRECISION_0 * IN_C + DATA_IN_0_PRECISION_1 - 1:0] packed_rolled_k[0:0];
  logic [DATA_IN_0_PRECISION_0 - 1:0] mrolled_k [IN_C - 1:0];
  logic [DATA_IN_0_PRECISION_1 - 1:0] erolled_k;
  logic rolled_k_valid;
  logic rolled_k_ready;

  logic [ROUND_PRECISION_0 -1:0] round_in[UNROLL_OUT_C-1:0];

  for (genvar i = 0; i < IN_C; i++)
  for (genvar j = 0; j < DATA_IN_0_PRECISION_0; j++)
    assign packed_data_in[i*DATA_IN_0_PRECISION_0+j] = mdata_in_0[i][j];
  assign packed_data_in[IN_C * DATA_IN_0_PRECISION_0 + DATA_IN_0_PRECISION_1 - 1 : IN_C * DATA_IN_0_PRECISION_0] = edata_in_0;

  sliding_window #(
      .IMG_WIDTH     (IN_X),
      .IMG_HEIGHT    (IN_Y),
      .KERNEL_WIDTH  (KERNEL_X),
      .KERNEL_HEIGHT (KERNEL_Y),
      .PADDING_WIDTH (0),
      .PADDING_HEIGHT(0),
      .CHANNELS      (1),
      .DATA_WIDTH    (IN_C * DATA_IN_0_PRECISION_0 + DATA_IN_0_PRECISION_1),
      .STRIDE        (STRIDE)
      /* verilator lint_off PINMISSING */
  ) sw_inst (
      .clk(clk),
      .rst(rst),
      .data_in(packed_data_in),
      .data_in_valid(data_in_0_valid),
      .data_in_ready(data_in_0_ready),

      .data_out(packed_kernel),
      .data_out_valid(kernel_valid),
      .data_out_ready(kernel_ready)
  );

  roller #(
      .DATA_WIDTH(IN_C * DATA_IN_0_PRECISION_0 + DATA_IN_0_PRECISION_1),
      .NUM(KERNEL_X * KERNEL_Y),
      .ROLL_NUM(1) // actually with only roll_num == 1, 
  ) roller_inst (
      .clk(clk),
      .rst(rst),
      .data_in(packed_kernel),
      .data_in_valid(kernel_valid),
      .data_in_ready(kernel_ready),
      .data_out(packed_rolled_k),
      .data_out_valid(rolled_k_valid),
      .data_out_ready(rolled_k_ready)
  );
  for (genvar i = 0; i < IN_C; i++)
    assign mrolled_k[i] = packed_rolled_k[0][(i+1)*DATA_IN_0_PRECISION_0 - 1 : i * DATA_IN_0_PRECISION_0];
  assign erolled_k = packed_rolled_k[0][IN_C * DATA_IN_0_PRECISION_0 + DATA_IN_0_PRECISION_1 - 1 : IN_C * DATA_IN_0_PRECISION_0];

  mxint_linear #(
      .HAS_BIAS              (HAS_BIAS),

      .DATA_IN_0_PRECISION_0      (DATA_IN_0_PRECISION_0),
      .DATA_IN_0_PRECISION_1      (DATA_IN_0_PRECISION_1),
      .DATA_IN_0_TENSOR_SIZE_DIM_0(KERNEL_X * KERNEL_Y * IN_C),
      .DATA_IN_0_TENSOR_SIZE_DIM_1(SLIDING_NUM),
      .DATA_IN_0_PARALLELISM_DIM_0(IN_C),
      .DATA_IN_0_PARALLELISM_DIM_1(1),

      .WEIGHT_PRECISION_0      (WEIGHT_PRECISION_0),
      .WEIGHT_PRECISION_1      (WEIGHT_PRECISION_1),
      .WEIGHT_TENSOR_SIZE_DIM_0(KERNEL_X * KERNEL_Y * IN_C),
      .WEIGHT_TENSOR_SIZE_DIM_1(OUT_C),
      .WEIGHT_PARALLELISM_DIM_0(IN_C),
      .WEIGHT_PARALLELISM_DIM_1(UNROLL_OUT_C),

      .BIAS_PRECISION_0      (BIAS_PRECISION_0),
      .BIAS_PRECISION_1      (BIAS_PRECISION_1),
      .BIAS_TENSOR_SIZE_DIM_0(OUT_C),
      .BIAS_TENSOR_SIZE_DIM_1(1),
      .BIAS_PARALLELISM_DIM_0(UNROLL_OUT_C),
      .BIAS_PARALLELISM_DIM_1(1),

      .DATA_OUT_0_PRECISION_0(DATA_OUT_0_PRECISION_0),
      .DATA_OUT_0_PRECISION_1(DATA_OUT_0_PRECISION_1)
  ) linear_inst (
      .clk(clk),
      .rst(rst),
      .mdata_in_0(mrolled_k),
      .edata_in_0(erolled_k),
      .data_in_0_valid(rolled_k_valid),
      .data_in_0_ready(rolled_k_ready),
      .mweight(mweight),
      .eweight(eweight),
      .weight_valid(weight_valid),
      .weight_ready(weight_ready),
      .mbias(mbias),
      .ebias(ebias),
      .bias_valid(bias_valid),
      .bias_ready(bias_ready),
      .mdata_out_0(mdata_out_0),
      .edata_out_0(edata_out_0),
      .data_out_0_valid(data_out_0_valid),
      .data_out_0_ready(data_out_0_ready)
  );

endmodule
