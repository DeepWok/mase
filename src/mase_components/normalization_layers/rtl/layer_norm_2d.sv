/*
Module      : group_norm_2d
Description : This module calculates the generalised group norm.
              https://arxiv.org/abs/1803.08494v3

              This module can be easily trivially specialised into layer norm or
              instance norm by setting the GROUP_CHANNELS param to equal C or 1
              respectively.

              Group norm is independent of batch size, so the input shape is:
              (GROUP, DEPTH_DIM1 * DEPTH_DIM0, COMPUTE_DIM1 * COMPUTE_DIM0)
    assume we flatten layernorm.normalized_shape to, and then calculate it
    so it actually is dim_0 = prod(normalized_shape), x.reshape(-1, dim0), out = norm(dim_0)(x)
    2d means parallelism here
*/

`timescale 1ns / 1ps
module layer_norm_2d #(
    // Dimensions
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0     = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_0   = 2,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1     = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_1   = 2,

    // Data widths
    parameter DATA_IN_0_PRECISION_0       = 8,
    parameter DATA_IN_0_PRECISION_1  = 4,
    parameter WEIGHT_PRECISION_0       = 8,
    parameter WEIGHT_PRECISION_1  = 4,
    parameter BIAS_PRECISION_0       = 8,
    parameter BIAS_PRECISION_1  = 4,
    parameter ELEMENTWISE_AFFINE  = 0 ,
    parameter BIAS = 0,
    parameter ISQRT_IN_PRECISION_0 = 8,
    parameter ISQRT_IN_PRECISION_1 = 8,
    parameter ISQRT_OUT_PRECISION_0 = 8,
    parameter ISQRT_OUT_PRECISION_1 = 4,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0     = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter DATA_OUT_0_PARALLELISM_DIM_0   = DATA_IN_0_PARALLELISM_DIM_0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1     = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1   = DATA_IN_0_PARALLELISM_DIM_1,
    parameter DATA_OUT_0_PRECISION_0      = 8,
    parameter DATA_OUT_0_PRECISION_1 = 4
) (
    input logic clk,
    input logic rst,

    input  logic [DATA_IN_0_PRECISION_0-1:0] data_in_0 [DATA_IN_0_PARALLELISM_DIM_1*DATA_IN_0_PARALLELISM_DIM_0-1:0],
    input  logic                data_in_0_valid,
    output logic                data_in_0_ready,

    input logic [WEIGHT_PRECISION_0-1:0] weight [DATA_IN_0_PARALLELISM_DIM_0 - 1 : 0],
    input  logic                weight_valid,
    output logic                weight_ready,
    
    input logic [BIAS_PRECISION_0-1:0] bias [DATA_IN_0_PARALLELISM_DIM_0 - 1 : 0],
    input  logic                bias_valid,
    output logic                bias_ready,
     
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_IN_0_PARALLELISM_DIM_1*DATA_IN_0_PARALLELISM_DIM_0-1:0],
    output logic                 data_out_0_valid,
    input  logic                 data_out_0_ready
);
    logic [DATA_IN_0_PARALLELISM_DIM_1 - 1:0] parallel_norm_in_ready, parallel_norm_out_valid;
    logic join_out_valid, join_out_ready;
    logic [DATA_OUT_0_PRECISION_0 - 1:0] norm_out [DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1 - 1:0];
    logic [AFFINE_PRECISION_0 -1:0] uncast_data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1 - 1:0]; 
    logic [AFFINE_PRECISION_0 - 1:0] casted_bias[DATA_OUT_0_PARALLELISM_DIM_0-1:0];
    localparam AFFINE_PRECISION_0 = DATA_OUT_0_PRECISION_0 + WEIGHT_PRECISION_0 + 1;
    localparam AFFINE_PRECISION_1 = DATA_OUT_0_PRECISION_1 + WEIGHT_PRECISION_1;  
    logic [BIAS_PRECISION_0 - 1:0] bias_buffered [DATA_IN_0_PARALLELISM_DIM_0 - 1 :0];
    logic [WEIGHT_PRECISION_0 - 1:0] weight_buffered [DATA_IN_0_PARALLELISM_DIM_0 - 1 :0];
    logic bias_buffered_valid, bias_buffered_ready, weight_buffered_ready, weight_buffered_valid;
    for (genvar i=0; i<DATA_IN_0_PARALLELISM_DIM_1; i++) begin: parallel_dim_1
        layer_norm_1d #(
            .DATA_IN_0_TENSOR_SIZE_DIM_0,
            .DATA_IN_0_PARALLELISM_DIM_0,
            // Data widths
            .DATA_IN_0_PRECISION_0,
            .DATA_IN_0_PRECISION_1,
            .ISQRT_IN_PRECISION_0,
            .ISQRT_IN_PRECISION_1,
            .ISQRT_OUT_PRECISION_0,
            .ISQRT_OUT_PRECISION_1,
            .DATA_OUT_0_TENSOR_SIZE_DIM_0,
            .DATA_OUT_0_PARALLELISM_DIM_0,
            .DATA_OUT_0_PRECISION_0,
            .DATA_OUT_0_PRECISION_1
        ) layer_norm_inst (
            .clk,
            .rst,
            .data_in_0(data_in_0[i*DATA_IN_0_PARALLELISM_DIM_0 + DATA_IN_0_PARALLELISM_DIM_0 - 1:  i*DATA_IN_0_PARALLELISM_DIM_0]),
            .data_in_0_valid,
            .data_in_0_ready(parallel_norm_in_ready[i]),
            .data_out_0(norm_out[i*DATA_IN_0_PARALLELISM_DIM_0 + DATA_IN_0_PARALLELISM_DIM_0 - 1:  i*DATA_IN_0_PARALLELISM_DIM_0]),
            .data_out_0_valid(parallel_norm_out_valid[i]),
            .data_out_0_ready(join_out_ready)
        );
    end
    assign data_in_0_ready = parallel_norm_in_ready[0];
    assign join_out_valid = parallel_norm_out_valid[0];
    input_buffer #(
        .DATA_WIDTH(BIAS_PRECISION_0),
        .IN_NUM    (DATA_IN_0_PARALLELISM_DIM_0),
        .REPEAT    (DATA_IN_0_TENSOR_SIZE_DIM_1/DATA_IN_0_PARALLELISM_DIM_1),
        .BUFFER_SIZE      (DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0)
    ) bias_buffer_inst (
        .clk,
        .rst,

        // Input streaming port
        .data_in (bias),
        .data_in_valid(bias_valid),
        .data_in_ready(bias_ready),

        // Output streaming port
        .data_out (bias_buffered),
        .data_out_valid(bias_buffered_valid),
        .data_out_ready(bias_buffered_ready)
    );
    input_buffer #(
        .DATA_WIDTH(WEIGHT_PRECISION_0),
        .IN_NUM    (DATA_IN_0_PARALLELISM_DIM_0),
        .REPEAT    (DATA_IN_0_TENSOR_SIZE_DIM_1/DATA_IN_0_PARALLELISM_DIM_1),
        .BUFFER_SIZE      (DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0)
    ) weight_buffer_inst (
        .clk,
        .rst,

        // Input streaming port
        .data_in (weight),
        .data_in_valid(weight_valid),
        .data_in_ready(weight_ready),

        // Output streaming port
        .data_out (weight_buffered),
        .data_out_valid(weight_buffered_valid),
        .data_out_ready(weight_buffered_ready)
    );
    if (ELEMENTWISE_AFFINE == 1) begin
        logic wd_valid, wd_ready;
        join2 weight_data_join_inst (
            .data_in_valid ({weight_buffered_valid, join_out_valid}),
            .data_in_ready ({weight_buffered_ready, join_out_ready}),
            .data_out_valid(wd_valid),
            .data_out_ready(wd_ready));
        logic [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1 - 1:0] parallel_wd_ready, parallel_bias_ready, parallel_data_out_0_valid;
        assign bias_buffered_ready = parallel_bias_ready[0];
        assign wd_ready = parallel_wd_ready[0];
        assign data_out_0_valid = parallel_data_out_0_valid[0];
        for (genvar i=0; i<DATA_OUT_0_PARALLELISM_DIM_1; i++) begin: affine_parallel_dim1 
            for (genvar j=0; j<DATA_OUT_0_PARALLELISM_DIM_0; j++) begin: affine_parallel_dim0
                localparam int k = i*DATA_IN_0_PARALLELISM_DIM_0 + j; 
                if (BIAS == 1) begin
                    join2 wd_bias_join_inst (
                        .data_in_valid ({wd_valid, bias_buffered_valid}),
                        .data_in_ready ({parallel_wd_ready[k], parallel_bias_ready[k]}),
                        .data_out_valid(parallel_data_out_0_valid[k]),
                        .data_out_ready(data_out_0_ready));
                    fixed_signed_cast #(
                        .IN_WIDTH(BIAS_PRECISION_0),
                        .IN_FRAC_WIDTH(BIAS_PRECISION_1),
                        .OUT_WIDTH(AFFINE_PRECISION_0),
                        .OUT_FRAC_WIDTH(AFFINE_PRECISION_1),
                        .SYMMETRIC(0),
                        .ROUND_FLOOR(1)
                    ) variance_cast_i (
                        .in_data (bias_buffered[j]),
                        .out_data(casted_bias[j])
                    ); 
                    assign uncast_data_out_0[k] = $signed(norm_out[k]) * $signed(weight_buffered[j]) + $signed(casted_bias[j]);
                end else begin
                assign parallel_wd_ready[k] = data_out_0_ready;
                assign parallel_data_out_0_valid[k] = wd_valid;
                assign parallel_bias_ready[k] = 1;
                assign uncast_data_out_0[k]= $signed(norm_out[k]) * $signed(weight_buffered[j]);
                end 
                fixed_signed_cast #(
                    .IN_WIDTH(AFFINE_PRECISION_0),
                    .IN_FRAC_WIDTH(AFFINE_PRECISION_1),
                    .OUT_WIDTH(DATA_OUT_0_PRECISION_0),
                    .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1),
                    .SYMMETRIC(0),
                    .ROUND_FLOOR(1)
                ) variance_cast_i (
                    .in_data (uncast_data_out_0[k]),
                    .out_data(data_out_0[k])
                );
            end
        end
    end else begin
        assign join_out_ready = data_out_0_ready;
        assign data_out_0_valid = join_out_valid;
        assign data_out_0 = norm_out;
    end 
endmodule

module layer_norm_1d #(
    // Dimensions
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0     = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_0   = 2,
    // Data widths
    parameter DATA_IN_0_PRECISION_0       = 8,
    parameter DATA_IN_0_PRECISION_1  = 4,
    parameter ISQRT_IN_PRECISION_0 = 8,
    parameter ISQRT_IN_PRECISION_1 = 8,
    parameter ISQRT_OUT_PRECISION_0 = 8,
    parameter ISQRT_OUT_PRECISION_1 = 4,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0     = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter DATA_OUT_0_PARALLELISM_DIM_0   = DATA_IN_0_PARALLELISM_DIM_0,
    parameter DATA_OUT_0_PRECISION_0      = 8,
    parameter DATA_OUT_0_PRECISION_1 = 4
) (
    input logic clk,
    input logic rst,

    input  logic [DATA_IN_0_PRECISION_0-1:0] data_in_0 [DATA_IN_0_PARALLELISM_DIM_0-1:0],
    input  logic                data_in_0_valid,
    output logic                data_in_0_ready,

    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_IN_0_PARALLELISM_DIM_0-1:0],
    output logic                 data_out_0_valid,
    input  logic                 data_out_0_ready
);

  // Derived params
  localparam DEPTH_DIM0 = DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0;

  localparam NUM_VALUES = DATA_IN_0_TENSOR_SIZE_DIM_0;

  localparam NUM_ITERS = DEPTH_DIM0;
  localparam ITER_WIDTH = $clog2(NUM_ITERS);

  // Compute Pipeline Widths

  localparam ADDER_TREE_IN_SIZE = DATA_IN_0_PARALLELISM_DIM_0;
  localparam ADDER_TREE_OUT_WIDTH = $clog2(ADDER_TREE_IN_SIZE) + DATA_IN_0_PRECISION_0;

  localparam ACC_OUT_WIDTH = ITER_WIDTH + ADDER_TREE_OUT_WIDTH;

  localparam DIFF_WIDTH = DATA_IN_0_PRECISION_0 + 1;
  localparam DIFF_FRAC_WIDTH = DATA_IN_0_PRECISION_1;

  localparam SQUARE_WIDTH = DIFF_WIDTH * 2;
  localparam SQUARE_FRAC_WIDTH = DIFF_FRAC_WIDTH * 2;

  localparam SQUARES_ADDER_TREE_IN_SIZE = DATA_IN_0_PARALLELISM_DIM_0;
  localparam SQUARES_ADDER_TREE_OUT_WIDTH = $clog2(SQUARES_ADDER_TREE_IN_SIZE) + SQUARE_WIDTH;
  localparam SQUARES_ADDER_TREE_OUT_FRAC_WIDTH = SQUARE_FRAC_WIDTH;

  localparam VARIANCE_WIDTH = ITER_WIDTH + SQUARES_ADDER_TREE_OUT_WIDTH;
  localparam VARIANCE_FRAC_WIDTH = SQUARES_ADDER_TREE_OUT_FRAC_WIDTH;


  localparam NORM_WIDTH = ISQRT_OUT_PRECISION_0 + DIFF_WIDTH;
  localparam NORM_FRAC_WIDTH = ISQRT_OUT_PRECISION_1 + DIFF_FRAC_WIDTH;

  /* verilator lint_off UNUSEDSIGNAL */
  // Input FIFO
  logic [DATA_IN_0_PRECISION_0-1:0] fifo_data[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic fifo_out_valid, fifo_out_ready;
  logic fifo_in_valid, fifo_in_ready;

  // Input Adder Tree
  logic [ADDER_TREE_OUT_WIDTH-1:0] adder_tree_data;
  logic adder_tree_out_valid, adder_tree_out_ready;
  logic adder_tree_in_valid, adder_tree_in_ready;


  logic [ACC_OUT_WIDTH-1:0] mu_acc;
  logic mu_acc_valid, mu_acc_ready;

  logic [DATA_IN_0_PRECISION_0-1:0] mu_in, mu_out;
  logic mu_out_valid, mu_out_ready;

  logic [ACC_OUT_WIDTH + ACC_OUT_WIDTH:0] mu_acc_div;

  logic mu_fifo_valid, mu_fifo_ready;

  logic signed [DIFF_WIDTH-1:0] diff_in[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic signed [DIFF_WIDTH-1:0] diff_out[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic diff_in_ready[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic diff_out_valid[DATA_IN_0_PARALLELISM_DIM_0-1:0];

  logic [SQUARE_WIDTH-1:0] square_in[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic square_in_ready[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic square_out_valid[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic [SQUARE_WIDTH-1:0] square_out[DATA_IN_0_PARALLELISM_DIM_0-1:0];

  // Split2 for split in pipeline from diff
  logic fifo_diff_in_valid, fifo_diff_in_ready;
  logic fifo_diff_out_valid;

  // Squares adder tree
  logic [SQUARES_ADDER_TREE_OUT_WIDTH-1:0] squares_adder_tree_data;
  logic squares_adder_tree_out_valid, squares_adder_tree_out_ready;
  logic squares_adder_tree_in_valid, squares_adder_tree_in_ready;

  // Squares Accumulator
  logic [VARIANCE_WIDTH-1:0] squares_acc;
  logic squares_acc_valid, squares_acc_ready;

  // Take the accumulated squares and divide it to get variance
  logic [SQUARES_ADDER_TREE_OUT_WIDTH+VARIANCE_WIDTH:0] variance_buffer;
  logic [VARIANCE_WIDTH-1:0] variance_in, variance_out;
  logic variance_out_valid, variance_out_ready;

  logic [ISQRT_IN_PRECISION_0-1:0] variance_cast;
  logic [ISQRT_IN_PRECISION_0-1:0] inv_sqrt_in;
  logic [ISQRT_OUT_PRECISION_0-1:0] inv_sqrt_out;
  // Take inverse square root of variance
  logic [ISQRT_OUT_PRECISION_0-1:0] inv_sqrt_data;
  logic inv_sqrt_valid, inv_sqrt_ready;

  // Repeat circular buffer to hold inverse square root of variance during mult
  logic [ISQRT_OUT_PRECISION_0-1:0] isqrt_circ_data;
  logic isqrt_circ_valid, isqrt_circ_ready;
  logic norm_in_valid;

  // FIFO for storing X-mu differences
  logic [DIFF_WIDTH-1:0] diff_batch_in[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic diff_batch_in_valid, diff_batch_in_ready;
  logic [DIFF_WIDTH-1:0] diff_batch_out[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic diff_batch_out_valid, diff_batch_out_ready;

  logic [NORM_WIDTH-1:0] norm_in_data[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic [NORM_WIDTH-1:0] norm_out_data[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic [DATA_OUT_0_PRECISION_0-1:0] norm_round_out[DATA_IN_0_PARALLELISM_DIM_0-1:0];

  logic [DATA_OUT_0_PRECISION_0-1:0] norm_batch_data[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic output_reg_ready;

  logic norm_in_ready[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic norm_out_valid[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic norm_batch_ready[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic output_reg_valid[DATA_IN_0_PARALLELISM_DIM_0-1:0];

  /* verilator lint_on UNUSEDSIGNAL */

  matrix_fifo #(
      .DATA_WIDTH(DATA_IN_0_PRECISION_0),
      .DIM0      (DATA_IN_0_PARALLELISM_DIM_0),
      .DIM1      (1),
      .FIFO_SIZE (4 * NUM_ITERS)
  ) input_fifo_inst (
      .clk(clk),
      .rst(rst),
      .in_data(data_in_0),
      .in_valid(fifo_in_valid),
      .in_ready(fifo_in_ready),
      .out_data(fifo_data),
      .out_valid(fifo_out_valid),
      .out_ready(fifo_out_ready)
  );

  // Input Adder Tree
  fixed_adder_tree #(
      .IN_SIZE (DATA_IN_0_PARALLELISM_DIM_0),
      .IN_WIDTH(DATA_IN_0_PRECISION_0)
  ) sum_adder_tree (
      .clk(clk),
      .rst(rst),
      .data_in(data_in_0),
      .data_in_valid(adder_tree_in_valid),
      .data_in_ready(adder_tree_in_ready),
      .data_out(adder_tree_data),
      .data_out_valid(adder_tree_out_valid),
      .data_out_ready(adder_tree_out_ready)
  );

  // Split2 for input to FIFO & Adder Tree
  split2 input_fifo_adder_split (
      .data_in_valid (data_in_0_valid),
      .data_in_ready (data_in_0_ready),
      .data_out_valid({adder_tree_in_valid, fifo_in_valid}),
      .data_out_ready({adder_tree_in_ready, fifo_in_ready})
  );
  // Accumulator for mu
  fixed_accumulator #(
      .IN_DEPTH(NUM_ITERS),
      .IN_WIDTH(ADDER_TREE_OUT_WIDTH)
  ) mu_accumulator (
      .clk(clk),
      .rst(rst),
      .data_in(adder_tree_data),
      .data_in_valid(adder_tree_out_valid),
      .data_in_ready(adder_tree_out_ready),
      .data_out(mu_acc),
      .data_out_valid(mu_acc_valid),
      .data_out_ready(mu_acc_ready)
  );


  // Division by NUM_VALUES
  // ACC_WIDTH = DATA_IN_WIDTH + $clog2(NUM_VALUES)
  // BASICALLY the same thing 
  localparam bit [ACC_OUT_WIDTH+1:0] INV_NUMVALUES_0 = ((1 << ACC_OUT_WIDTH) / NUM_VALUES);
  assign mu_acc_div = ($signed(mu_acc) * $signed(INV_NUMVALUES_0));

    fixed_signed_cast #(
        .IN_WIDTH(ACC_OUT_WIDTH+ ACC_OUT_WIDTH + 1),
        .IN_FRAC_WIDTH(DATA_IN_0_PRECISION_1 + ACC_OUT_WIDTH),
        .OUT_WIDTH(DATA_IN_0_PRECISION_0),
        .OUT_FRAC_WIDTH(DATA_IN_0_PRECISION_1),
        .SYMMETRIC(0),
        .ROUND_FLOOR(1)
    ) acc_div_cast_i (
        .in_data (mu_acc_div),
        .out_data(mu_in)
    );

  single_element_repeat #(
      .DATA_WIDTH(DATA_IN_0_PRECISION_0),
      .REPEAT(NUM_ITERS)
  ) mu_buffer (
      .clk(clk),
      .rst(rst),
      .in_data(mu_in),
      .in_valid(mu_acc_valid),
      .in_ready(mu_acc_ready),
      .out_data(mu_out),
      .out_valid(mu_out_valid),
      .out_ready(mu_out_ready)
  );

  // Join 2 for combining fifo and mu buffer signals
  assign mu_fifo_ready = diff_in_ready[0];

  join2 mu_fifo_join2 (
      .data_in_valid ({mu_out_valid, fifo_out_valid}),
      .data_in_ready ({mu_out_ready, fifo_out_ready}),
      .data_out_valid(mu_fifo_valid),
      .data_out_ready(mu_fifo_ready)
  );

  // Compute pipeline

  for (genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_0; i++) begin : compute_pipe

    // Take the difference between input and mean: (X - mu)
    assign diff_in[i] = $signed(fifo_data[i]) - $signed(mu_out);

    skid_buffer #(
        .DATA_WIDTH(DIFF_WIDTH)
    ) subtract_reg (
        .clk(clk),
        .rst(rst),
        .data_in(diff_in[i]),
        .data_in_valid(mu_fifo_valid),
        .data_in_ready(diff_in_ready[i]),
        .data_out(diff_out[i]),
        .data_out_valid(diff_out_valid[i]),
        .data_out_ready(fifo_diff_in_ready)
    );

    // Assign the output of diff int batch to be buffered
    assign diff_batch_in[i] = diff_out[i];

    // There will be a split in the pipline here, split2 is down below.

    // Take the difference and square it: (X - mu) ^ 2

    assign square_in[i] = $signed(diff_batch_in[i]) * $signed(diff_batch_in[i]);

    skid_buffer #(
        .DATA_WIDTH(SQUARE_WIDTH)
    ) square_reg (
        .clk(clk),
        .rst(rst),
        .data_in(square_in[i]),
        .data_in_valid(fifo_diff_out_valid),
        .data_in_ready(square_in_ready[i]),
        .data_out(square_out[i]),
        .data_out_valid(square_out_valid[i]),
        .data_out_ready(squares_adder_tree_in_ready)
    );
  end

  assign fifo_diff_in_valid = diff_out_valid[0];
  split2 fifo_diff_split (
      .data_in_valid (fifo_diff_in_valid),
      .data_in_ready (fifo_diff_in_ready),
      .data_out_valid({diff_batch_in_valid, fifo_diff_out_valid}),
      .data_out_ready({diff_batch_in_ready, square_in_ready[0]})
  );

  assign squares_adder_tree_in_valid = square_out_valid[0];

  fixed_adder_tree #(
      .IN_SIZE (SQUARES_ADDER_TREE_IN_SIZE),
      .IN_WIDTH(SQUARE_WIDTH)
  ) squares_adder_tree (
      .clk(clk),
      .rst(rst),
      .data_in(square_out),
      .data_in_valid(squares_adder_tree_in_valid),
      .data_in_ready(squares_adder_tree_in_ready),
      .data_out(squares_adder_tree_data),
      .data_out_valid(squares_adder_tree_out_valid),
      .data_out_ready(squares_adder_tree_out_ready)
  );

  fixed_accumulator #(
      .IN_DEPTH(NUM_ITERS),
      .IN_WIDTH(SQUARES_ADDER_TREE_OUT_WIDTH)
  ) squares_accumulator (
      .clk(clk),
      .rst(rst),
      .data_in(squares_adder_tree_data),
      .data_in_valid(squares_adder_tree_out_valid),
      .data_in_ready(squares_adder_tree_out_ready),
      .data_out(squares_acc),
      .data_out_valid(squares_acc_valid),
      .data_out_ready(squares_acc_ready)
  );

  // Division by NUM_VALUES
  localparam bit [SQUARES_ADDER_TREE_OUT_WIDTH+1:0] INV_NUMVALUES_1 = ((1 << SQUARES_ADDER_TREE_OUT_WIDTH) / NUM_VALUES);
  assign variance_buffer = (squares_acc * INV_NUMVALUES_1) >> SQUARES_ADDER_TREE_OUT_WIDTH;
  assign variance_in = variance_buffer[VARIANCE_WIDTH-1:0];

  skid_buffer #(
      .DATA_WIDTH(VARIANCE_WIDTH)
  ) variance_reg (
      .clk(clk),
      .rst(rst),
      .data_in(variance_in),
      .data_in_valid(squares_acc_valid),
      .data_in_ready(squares_acc_ready),
      .data_out(variance_out),
      .data_out_valid(variance_out_valid),
      .data_out_ready(variance_out_ready)
  );


    fixed_signed_cast #(
        .IN_WIDTH(VARIANCE_WIDTH),
        .IN_FRAC_WIDTH(VARIANCE_FRAC_WIDTH),
        .OUT_WIDTH(ISQRT_IN_PRECISION_0),
        .OUT_FRAC_WIDTH(ISQRT_IN_PRECISION_1),
        .SYMMETRIC(0),
        .ROUND_FLOOR(1)
    ) variance_cast_i (
        .in_data (variance_out),
        .out_data(variance_cast)
    );
  register_slice #(
      .DATA_WIDTH(ISQRT_IN_PRECISION_0)
  ) register_slice (
      .clk           (clk),
      .rst           (rst),
      .data_in_valid (variance_out_valid),
      .data_in_ready (variance_out_ready),
      .data_in       (variance_cast),
      .data_out_valid(inv_sqrt_valid),
      .data_out_ready(inv_sqrt_ready),
      .data_out      (inv_sqrt_in)
  );

    isqrt_lut #(
        .DATA_IN_0_PRECISION_0 (ISQRT_IN_PRECISION_0),
        .DATA_IN_0_PRECISION_1 (ISQRT_IN_PRECISION_1),
        .DATA_OUT_0_PRECISION_0(ISQRT_OUT_PRECISION_0),
        .DATA_OUT_0_PRECISION_1(ISQRT_OUT_PRECISION_1)
    ) exp_map (
        .data_in_0 (inv_sqrt_in),
        .data_out_0(inv_sqrt_data)
    );


  single_element_repeat #(
      .DATA_WIDTH(ISQRT_OUT_PRECISION_0),
      .REPEAT(NUM_ITERS)
  ) isqrt_var_circ_buffer (
      .clk(clk),
      .rst(rst),
      .in_data(inv_sqrt_data),
      .in_valid(inv_sqrt_valid),
      .in_ready(inv_sqrt_ready),
      .out_data(isqrt_circ_data),
      .out_valid(isqrt_circ_valid),
      .out_ready(isqrt_circ_ready)
  );

  // Join2 for pipeline join at sqrt and diff fifo
  // logic inv_sqrt_ready;
  join2 diff_fifo_isqrt_join (
      .data_in_valid ({diff_batch_out_valid, isqrt_circ_valid}),
      .data_in_ready ({diff_batch_out_ready, isqrt_circ_ready}),
      .data_out_valid(norm_in_valid),
      .data_out_ready(norm_in_ready[0])
  );



  matrix_fifo #(
      .DATA_WIDTH(DIFF_WIDTH),
      .DIM0(DATA_IN_0_PARALLELISM_DIM_0),
      .DIM1(1),
      .FIFO_SIZE(4 * NUM_ITERS)
  ) diff_fifo_inst (
      .clk(clk),
      .rst(rst),
      .in_data(diff_batch_in),
      .in_valid(diff_batch_in_valid),
      .in_ready(diff_batch_in_ready),
      .out_data(diff_batch_out),
      .out_valid(diff_batch_out_valid),
      .out_ready(diff_batch_out_ready)
  );




  // Output chunks compute pipeline: final multiply and output cast

  for (genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_0; i++) begin : out_mult_pipe

    // Multiply difference with 1/sqrt(var) to get normalized result
    assign norm_in_data[i] = $signed({1'b0, isqrt_circ_data}) * $signed(diff_batch_out[i]);

    skid_buffer #(
        .DATA_WIDTH(NORM_WIDTH)
    ) norm_reg (
        .clk(clk),
        .rst(rst),
        .data_in(norm_in_data[i]),
        .data_in_valid(norm_in_valid),
        .data_in_ready(norm_in_ready[i]),
        .data_out(norm_out_data[i]),
        .data_out_valid(norm_out_valid[i]),
        .data_out_ready(norm_batch_ready[i])
    );

    // Output Rounding Stage
    fixed_signed_cast #(
        .IN_WIDTH(NORM_WIDTH),
        .IN_FRAC_WIDTH(NORM_FRAC_WIDTH),
        .OUT_WIDTH(DATA_OUT_0_PRECISION_0),
        .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1),
        .SYMMETRIC(0),
        .ROUND_FLOOR(1)
    ) output_cast (
        .in_data (norm_out_data[i]),
        .out_data(norm_round_out[i])
    );

    skid_buffer #(
        .DATA_WIDTH(DATA_OUT_0_PRECISION_0)
    ) output_reg (
        .clk(clk),
        .rst(rst),
        .data_in(norm_round_out[i]),
        .data_in_valid(norm_out_valid[i]),
        .data_in_ready(norm_batch_ready[i]),
        .data_out(norm_batch_data[i]),
        .data_out_valid(output_reg_valid[i]),
        .data_out_ready(output_reg_ready)
    );
  end

  // Final connection to output
  assign data_out_0 = norm_batch_data;
  assign data_out_0_valid = output_reg_valid[0];
  assign output_reg_ready = data_out_0_ready;

endmodule
