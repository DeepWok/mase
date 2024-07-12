`timescale 1ns / 1ps
module convolution_arith #(
    // assume output will only unroll_out_channels
    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter WEIGHT_PRECISION_0 = 8,
    parameter WEIGHT_PRECISION_1 = 4,
    parameter BIAS_PRECISION_0 = 8,
    parameter BIAS_PRECISION_1 = 4,
    parameter ROLL_IN_NUM = 4,
    parameter ROLL_OUT_NUM = 2,
    parameter IN_CHANNELS_DEPTH = 4,
    parameter OUT_CHANNELS_PARALLELISM = 2,
    parameter OUT_CHANNELS_DEPTH = 2,
    parameter WEIGHT_REPEATS = 4,
    parameter HAS_BIAS = 0,
    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
        ROLL_IN_NUM * IN_CHANNELS_DEPTH
    ),
    parameter DATA_OUT_0_PRECISION_1 = DATA_IN_0_PRECISION_1 + WEIGHT_PRECISION_1
) (
    input clk,
    input rst,

    input        [DATA_IN_0_PRECISION_0 - 1:0] data_in_0      [ROLL_OUT_NUM - 1 : 0],
    input                                      data_in_0_valid,
    output logic                               data_in_0_ready,

    input [WEIGHT_PRECISION_0 - 1:0] weight[OUT_CHANNELS_PARALLELISM * ROLL_OUT_NUM - 1 : 0],
    input weight_valid,
    output logic weight_ready,


    input        [BIAS_PRECISION_0-1:0] bias      [OUT_CHANNELS_PARALLELISM-1:0],
    input                               bias_valid,
    output logic                        bias_ready,

    output [DATA_OUT_0_PRECISION_0 - 1:0] data_out_0      [OUT_CHANNELS_PARALLELISM - 1:0],
    output                                data_out_0_valid,
    input                                 data_out_0_ready
);
  localparam FLATTENED_KNEREL_SIZE = IN_CHANNELS_DEPTH * ROLL_IN_NUM;
  localparam FLATTENED_CHUNKED_KNEREL_BLOCK_SIZE = ROLL_IN_NUM;
  localparam OUT_CHANNELS_SIZE = OUT_CHANNELS_PARALLELISM * OUT_CHANNELS_DEPTH;
  localparam FLATTENED_WEIGHT_SIZE = FLATTENED_KNEREL_SIZE * OUT_CHANNELS_SIZE;

  logic [DATA_IN_0_PRECISION_0 - 1:0] buffered_data_in_0[ROLL_OUT_NUM - 1 : 0];
  logic buffered_data_in_0_valid, buffered_data_in_0_ready;
  input_buffer #(
      .DATA_WIDTH (DATA_IN_0_PRECISION_0),
      // Repeat for number of rows in matrix A
      .REPEAT     (OUT_CHANNELS_DEPTH),
      .BUFFER_SIZE(FLATTENED_KNEREL_SIZE / ROLL_OUT_NUM),
      .IN_NUM     (ROLL_OUT_NUM)
  ) data_in_0_buffer (
      .clk           (clk),
      .rst           (rst),
      .data_in       (data_in_0),
      .data_in_valid (data_in_0_valid),
      .data_in_ready (data_in_0_ready),
      .data_out      (buffered_data_in_0),
      .data_out_valid(buffered_data_in_0_valid),
      .data_out_ready(buffered_data_in_0_ready)
  );
  localparam ARITH_OUT_PRECISION_0 = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
      ROLL_IN_NUM * IN_CHANNELS_DEPTH
  );
  localparam ARITH_OUT_PRECISION_1 = DATA_IN_0_PRECISION_1 + WEIGHT_PRECISION_1;
  logic [ARITH_OUT_PRECISION_0 - 1:0] arith_out[OUT_CHANNELS_PARALLELISM - 1:0];

  simple_convolution_arith #(
      .DATA_IN_0_PRECISION_0(DATA_IN_0_PRECISION_0),
      .DATA_IN_0_PRECISION_1(DATA_IN_0_PRECISION_1),
      .WEIGHT_PRECISION_0(WEIGHT_PRECISION_0),
      .WEIGHT_PRECISION_1(WEIGHT_PRECISION_1),
      .BIAS_PRECISION_0(BIAS_PRECISION_0),
      .BIAS_PRECISION_1(BIAS_PRECISION_1),
      .ROLL_IN_NUM(ROLL_IN_NUM),
      .ROLL_OUT_NUM(ROLL_OUT_NUM),
      .HAS_BIAS(HAS_BIAS),
      .IN_CHANNELS_DEPTH(IN_CHANNELS_DEPTH),
      .OUT_CHANNELS_PARALLELISM(OUT_CHANNELS_PARALLELISM)
  ) simple_convolution_arith_inst (
      .clk(clk),
      .rst(rst),
      .data_in_0(buffered_data_in_0),
      .data_in_0_valid(buffered_data_in_0_valid),
      .data_in_0_ready(buffered_data_in_0_ready),
      .weight(weight),
      .weight_valid(weight_valid),
      .weight_ready(weight_ready),
      .bias(bias),
      .bias_valid(bias_valid),
      .bias_ready(bias_ready),
      .data_out_0(data_out_0),
      .data_out_0_valid(data_out_0_valid),
      .data_out_0_ready(data_out_0_ready)
  );
  //   for (genvar i = 0; i < OUT_CHANNELS_PARALLELISM; i++) begin : round_parallism
  //     fixed_round #(
  //         .IN_WIDTH      (ARITH_OUT_PRECISION_0),
  //         .IN_FRAC_WIDTH (ARITH_OUT_PRECISION_1),
  //         .OUT_WIDTH     (DATA_OUT_0_PRECISION_0),
  //         .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1)
  //     ) round_inst (
  //         .data_in (arith_out[i]),
  //         .data_out(data_out_0[i])
  //     );end
endmodule

module simple_convolution_arith #(
    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter WEIGHT_PRECISION_0 = 8,
    parameter WEIGHT_PRECISION_1 = 4,
    parameter BIAS_PRECISION_0 = 8,
    parameter BIAS_PRECISION_1 = 4,
    parameter ROLL_IN_NUM = 4,
    parameter ROLL_OUT_NUM = 2,
    parameter IN_CHANNELS_DEPTH = 4,
    parameter OUT_CHANNELS_PARALLELISM = 2,
    parameter HAS_BIAS,
    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
        ROLL_IN_NUM * IN_CHANNELS_DEPTH
    ),
    parameter DATA_OUT_0_PRECISION_1 = DATA_IN_0_PRECISION_1 + WEIGHT_PRECISION_1
) (
    input clk,
    input rst,

    input        [DATA_IN_0_PRECISION_0 - 1:0] data_in_0      [ROLL_OUT_NUM - 1 : 0],
    input                                      data_in_0_valid,
    output logic                               data_in_0_ready,

    input [WEIGHT_PRECISION_0 - 1:0] weight[OUT_CHANNELS_PARALLELISM * ROLL_OUT_NUM - 1 : 0],
    input weight_valid,
    output logic weight_ready,


    input        [BIAS_PRECISION_0-1:0] bias      [OUT_CHANNELS_PARALLELISM-1:0],
    input                               bias_valid,
    output logic                        bias_ready,

    output [DATA_OUT_0_PRECISION_0 - 1:0] data_out_0      [OUT_CHANNELS_PARALLELISM - 1:0],
    output                                data_out_0_valid,
    input                                 data_out_0_ready
);
  initial begin
    assert ((ROLL_IN_NUM % ROLL_OUT_NUM == 0) && (ROLL_IN_NUM >= ROLL_OUT_NUM))
    else $fatal("Roll parameter not set correctly");
  end

  logic [OUT_CHANNELS_PARALLELISM -1:0]
      parallel_data_in_0_ready, parallel_weight_ready, parallel_acc_out_valid;
  localparam ACC_WIDTH = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
      ROLL_IN_NUM * IN_CHANNELS_DEPTH
  );
  logic [ACC_WIDTH - 1:0] acc_out[OUT_CHANNELS_PARALLELISM -1:0];
  logic acc_out_valid;
  logic acc_out_ready;

  always_comb begin
    data_in_0_ready = parallel_data_in_0_ready[0];
    weight_ready = parallel_weight_ready[0];
    acc_out_valid = parallel_acc_out_valid[0];
  end
  //   for (genvar i = 0; i < OUT_CHANNELS_PARALLELISM; i++) begin : unpack_w
  //     logic [WEIGHT_PRECISION_0-1:0]w[ROLL_OUT_NUM - 1:0];
  //     assign w = weight[(i+1)*ROLL_OUT_NUM-1 : i*ROLL_OUT_NUM];
  //   end

  for (genvar i = 0; i < OUT_CHANNELS_PARALLELISM; i++) begin : oc_parallelism
    dp_acc #(
        .DATA_IN_0_PRECISION_0(DATA_IN_0_PRECISION_0),
        .WEIGHT_PRECISION_0(WEIGHT_PRECISION_0),
        .DP_SIZE(ROLL_OUT_NUM),
        .ACC_DEPTH(ROLL_IN_NUM / ROLL_OUT_NUM * IN_CHANNELS_DEPTH),
    ) dp_acc_inst (
        .clk(clk),
        .rst(rst),
        .data_in_0(data_in_0),
        .data_in_0_valid(data_in_0_valid),
        .data_in_0_ready(parallel_data_in_0_ready[i]),
        .weight(weight[(i+1)*ROLL_OUT_NUM-1 : i*ROLL_OUT_NUM]),
        .weight_valid(weight_valid),
        .weight_ready(parallel_weight_ready[i]),
        .data_out_0(acc_out[i]),
        .data_out_0_valid(parallel_acc_out_valid[i]),
        .data_out_0_ready(acc_out_ready)
    );
  end
  // * Add bias
  logic [DATA_OUT_0_PRECISION_0 - 1:0] bias_casted[OUT_CHANNELS_PARALLELISM-1:0];
  if (HAS_BIAS == 1) begin
    join2 join2_acc_bias_i (
        .data_in_valid ({acc_out_valid, bias_valid}),
        .data_in_ready ({acc_out_ready, bias_ready}),
        .data_out_valid(data_out_0_valid),
        .data_out_ready(data_out_0_ready)
    );
    fixed_cast #(
        .IN_SIZE       (OUT_CHANNELS_PARALLELISM),
        .IN_WIDTH      (BIAS_PRECISION_0),
        .IN_FRAC_WIDTH (BIAS_PRECISION_1),
        .OUT_WIDTH     (DATA_OUT_0_PRECISION_0),
        .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1)
    ) bias_cast_i (
        .data_in (bias),
        .data_out(bias_casted)
    );

    for (genvar i = 0; i < OUT_CHANNELS_PARALLELISM; i++) begin
      assign data_out_0[i] = $signed(acc_out[i]) + $signed(bias_casted[i]);
    end

  end else begin
    assign data_out_0 = acc_out;
    assign data_out_0_valid = acc_out_valid;
    assign acc_out_ready = data_out_0_ready;
    assign bias_ready = 1'b1;
  end

endmodule

module dp_acc #(
    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter WEIGHT_PRECISION_0 = 8,
    parameter DP_SIZE = 4,
    parameter ACC_DEPTH = 4,
    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
        DP_SIZE * ACC_DEPTH
    )
) (
    input clk,
    input rst,

    input  logic [DATA_IN_0_PRECISION_0 - 1:0] data_in_0      [DP_SIZE - 1 : 0],
    input                                      data_in_0_valid,
    output                                     data_in_0_ready,

    input  logic [WEIGHT_PRECISION_0 - 1:0] weight      [DP_SIZE - 1:0],
    input                                   weight_valid,
    output                                  weight_ready,

    output [DATA_OUT_0_PRECISION_0 - 1:0] data_out_0,
    output                                data_out_0_valid,
    input                                 data_out_0_ready
);
  localparam DP_WIDTH = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(DP_SIZE);
  logic [DP_WIDTH - 1:0] middle_result;
  logic middle_result_valid, middle_result_ready;

  fixed_dot_product #(
      .IN_WIDTH    (DATA_IN_0_PRECISION_0),
      .IN_SIZE     (DP_SIZE),
      .WEIGHT_WIDTH(WEIGHT_PRECISION_0)
  ) dot_product_inst (
      .clk           (clk),
      .rst           (rst),
      .data_in       (data_in_0),
      .data_in_valid (data_in_0_valid),
      .data_in_ready (data_in_0_ready),
      .weight        (weight),
      .weight_valid  (weight_valid),
      .weight_ready  (weight_ready),
      .data_out      (middle_result),
      .data_out_valid(middle_result_valid),
      .data_out_ready(middle_result_ready)
  );

  fixed_accumulator #(
      .IN_DEPTH(ACC_DEPTH),
      .IN_WIDTH(DP_WIDTH)
  ) acc_inst (
      .clk           (clk),
      .rst           (rst),
      .data_in       (middle_result),
      .data_in_valid (middle_result_valid),
      .data_in_ready (middle_result_ready),
      .data_out      (data_out_0),
      .data_out_valid(data_out_0_valid),
      .data_out_ready(data_out_0_ready)
  );
endmodule
