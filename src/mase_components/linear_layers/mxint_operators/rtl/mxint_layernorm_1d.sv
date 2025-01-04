`timescale 1ns / 1ps
module mxint_layernorm_1d #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 10,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,

    parameter DATA_IN_0_MAN_WIDTH = 8,
    parameter DATA_IN_0_MAN_FRAC_WIDTH = DATA_IN_0_MAN_WIDTH - 1,
    parameter DATA_IN_0_EXP_WIDTH = 4,

    parameter DATA_OUT_0_MAN_WIDTH = 8,
    parameter DATA_OUT_0_MAN_FRAC_WIDTH = DATA_OUT_0_MAN_WIDTH - 1,
    parameter DATA_OUT_0_EXP_WIDTH = 4,

    parameter ISQRT_IN_MAN_WIDTH = 8,
    parameter ISQRT_IN_MAN_FRAC_WIDTH = 4,
    parameter ISQRT_OUT_MAN_WIDTH = 8,
    parameter ISQRT_OUT_MAN_FRAC_WIDTH = 4
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input clk,
    input rst,

    input logic data_in_0_valid,
    output logic data_in_0_ready,
    input logic [DATA_IN_0_MAN_WIDTH-1:0] mdata_in_0[DATA_OUT_0_PARALLELISM_DIM_0-1:0],
    input logic [DATA_IN_0_EXP_WIDTH-1:0] edata_in_0,

    output logic data_out_0_valid,
    input logic data_out_0_ready,
    output logic [DATA_OUT_0_MAN_WIDTH-1:0] mdata_out_0[DATA_OUT_0_PARALLELISM_DIM_0-1:0],
    output logic [DATA_OUT_0_EXP_WIDTH-1:0] edata_out_0
);
    // Internal signals
    logic [DATA_IN_0_MAN_WIDTH-1:0] casted_mdata_in[DATA_IN_0_PARALLELISM_DIM_0-1:0];
    logic [DATA_IN_0_EXP_WIDTH-1:0] casted_edata_in;
    logic casted_data_in_valid;
    logic casted_data_in_ready;

    dim_0_cast #(
        .MAN_WIDTH(DATA_IN_0_MAN_WIDTH),
        .EXP_WIDTH(DATA_IN_0_EXP_WIDTH),
        .IN_DEPTH(DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0),
        .BLOCK_SIZE(DATA_IN_0_PARALLELISM_DIM_0)
    ) u_dim_0_cast (
        .clk(clk),
        .rst(rst),
        .data_in_0_valid(data_in_0_valid),
        .data_in_0_ready(data_in_0_ready),
        .mdata_in_0(mdata_in_0),
        .edata_in_0(edata_in_0),
        .data_out_0_valid(casted_data_in_valid),
        .data_out_0_ready(casted_data_in_ready),
        .mdata_out_0(casted_mdata_in),
        .edata_out_0(casted_edata_in)
    );

    layernorm_core #(
        .DATA_IN_0_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0),
        .DATA_IN_0_PARALLELISM_DIM_0(DATA_IN_0_PARALLELISM_DIM_0),
        .DATA_OUT_0_TENSOR_SIZE_DIM_0(DATA_OUT_0_TENSOR_SIZE_DIM_0),
        .DATA_OUT_0_PARALLELISM_DIM_0(DATA_OUT_0_PARALLELISM_DIM_0),
        // Data widths
        .DATA_IN_0_WIDTH(DATA_IN_0_MAN_WIDTH),
        .DATA_IN_0_FRAC_WIDTH(DATA_IN_0_MAN_FRAC_WIDTH),
        .ISQRT_IN_WIDTH(ISQRT_IN_MAN_WIDTH),
        .ISQRT_IN_FRAC_WIDTH(ISQRT_IN_MAN_FRAC_WIDTH),
        .ISQRT_OUT_WIDTH(ISQRT_OUT_MAN_WIDTH),
        .ISQRT_OUT_FRAC_WIDTH(ISQRT_OUT_MAN_FRAC_WIDTH),
        .DATA_OUT_0_WIDTH(DATA_OUT_0_MAN_WIDTH),
        .DATA_OUT_0_FRAC_WIDTH(DATA_OUT_0_MAN_FRAC_WIDTH)
    ) u_layer_norm_1d (
        .clk(clk),
        .rst(rst),
        .data_in_0(casted_mdata_in),
        .data_in_0_valid(casted_data_in_valid),
        .data_in_0_ready(casted_data_in_ready),
        .mdata_out_0(mdata_out_0),
        .edata_out_0(edata_out_0),
        .data_out_0_valid(data_out_0_valid),
        .data_out_0_ready(data_out_0_ready)
    );

endmodule
/*
layernorm 1d
*/
module layernorm_core #(
    // Dimensions
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0  = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_0  = 2,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,
    // Data widths
    parameter DATA_IN_0_WIDTH        = 8,
    parameter DATA_IN_0_FRAC_WIDTH        = 4,
    parameter ISQRT_IN_WIDTH         = 8,
    parameter ISQRT_IN_FRAC_WIDTH         = 8,
    
    parameter ISQRT_OUT_WIDTH        = 8,
    parameter ISQRT_OUT_FRAC_WIDTH        = 4,
    parameter DATA_OUT_0_WIDTH       = 8,
    parameter DATA_OUT_0_FRAC_WIDTH       = 4,
    parameter DATA_OUT_0_EXP_WIDTH       = 4
) (
    input logic clk,
    input logic rst,

    input  logic [DATA_IN_0_WIDTH-1:0] data_in_0      [DATA_IN_0_PARALLELISM_DIM_0-1:0],
    input  logic                             data_in_0_valid,
    output logic                             data_in_0_ready,

    output logic [DATA_OUT_0_WIDTH-1:0] mdata_out_0      [DATA_IN_0_PARALLELISM_DIM_0-1:0],
    output logic [DATA_OUT_0_EXP_WIDTH-1:0] edata_out_0,
    output logic                              data_out_0_valid,
    input  logic                              data_out_0_ready
);

  // Derived params
  localparam DEPTH_DIM0 = DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0;

  localparam NUM_VALUES = DATA_IN_0_TENSOR_SIZE_DIM_0;

  localparam NUM_ITERS = DEPTH_DIM0;
  localparam ITER_WIDTH = $clog2(NUM_ITERS);

  // Compute Pipeline Widths

  localparam ADDER_TREE_IN_SIZE = DATA_IN_0_PARALLELISM_DIM_0;
  localparam ADDER_TREE_OUT_WIDTH = $clog2(ADDER_TREE_IN_SIZE) + DATA_IN_0_WIDTH;

  localparam ACC_OUT_WIDTH = ITER_WIDTH + ADDER_TREE_OUT_WIDTH;

  localparam DIFF_WIDTH = DATA_IN_0_WIDTH + 1;
  localparam DIFF_FRAC_WIDTH = DATA_IN_0_FRAC_WIDTH;

  localparam SQUARE_WIDTH = DIFF_WIDTH * 2;
  localparam SQUARE_FRAC_WIDTH = DIFF_FRAC_WIDTH * 2;

  localparam SQUARES_ADDER_TREE_IN_SIZE = DATA_IN_0_PARALLELISM_DIM_0;
  localparam SQUARES_ADDER_TREE_OUT_WIDTH = $clog2(SQUARES_ADDER_TREE_IN_SIZE) + SQUARE_WIDTH;
  localparam SQUARES_ADDER_TREE_OUT_FRAC_WIDTH = SQUARE_FRAC_WIDTH;

  localparam VARIANCE_WIDTH = ITER_WIDTH + SQUARES_ADDER_TREE_OUT_WIDTH;
  localparam VARIANCE_FRAC_WIDTH = SQUARES_ADDER_TREE_OUT_FRAC_WIDTH;


  localparam NORM_WIDTH = ISQRT_OUT_WIDTH + DIFF_WIDTH;
  localparam NORM_FRAC_WIDTH = ISQRT_OUT_FRAC_WIDTH + DIFF_FRAC_WIDTH;

  /* verilator lint_off UNUSEDSIGNAL */
  // Input FIFO
  logic [DATA_IN_0_WIDTH-1:0] fifo_data[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic fifo_out_valid, fifo_out_ready;
  logic fifo_in_valid, fifo_in_ready;

  // Input Adder Tree
  logic [ADDER_TREE_OUT_WIDTH-1:0] adder_tree_data;
  logic adder_tree_out_valid, adder_tree_out_ready;
  logic adder_tree_in_valid, adder_tree_in_ready;


  logic [ACC_OUT_WIDTH-1:0] mu_acc;
  logic mu_acc_valid, mu_acc_ready;

  logic [DATA_IN_0_WIDTH-1:0] mu_in, mu_out;
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

  // From now it becomes mxint quantization
  logic [ISQRT_OUT_WIDTH-1:0] minv_sqrt_out;
  logic [DATA_OUT_0_EXP_WIDTH-1:0] einv_sqrt_out;
  logic inv_sqrt_out_valid, inv_sqrt_out_ready;

  // Repeat circular buffer to hold inverse square root of variance during mult
  logic [ISQRT_OUT_WIDTH-1:0] misqrt_circ;
  logic [DATA_OUT_0_EXP_WIDTH-1:0] eisqrt_circ;
  logic isqrt_circ_valid, isqrt_circ_ready;
  logic norm_in_valid;

  // FIFO for storing X-mu differences
  logic [DIFF_WIDTH-1:0] diff_batch_in[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic diff_batch_in_valid, diff_batch_in_ready;
  logic [DIFF_WIDTH-1:0] diff_batch_out[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic diff_batch_out_valid, diff_batch_out_ready;

  logic [NORM_WIDTH-1:0] mnorm_in_data[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic [DATA_OUT_0_EXP_WIDTH-1:0] enorm_in_data[DATA_IN_0_PARALLELISM_DIM_0-1:0];

  logic [NORM_WIDTH-1:0] mnorm_out_data[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic [DATA_OUT_0_EXP_WIDTH-1:0] enorm_out_data[DATA_IN_0_PARALLELISM_DIM_0-1:0];

  logic [DATA_OUT_0_WIDTH-1:0] mnorm_round_out[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic [DATA_OUT_0_EXP_WIDTH-1:0] enorm_round_out[DATA_IN_0_PARALLELISM_DIM_0-1:0];

  logic [DATA_OUT_0_WIDTH-1:0] mnorm_batch_data[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic [DATA_OUT_0_EXP_WIDTH-1:0] enorm_batch_data[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic output_reg_ready;

  logic norm_in_ready[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic norm_out_valid[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic norm_batch_ready[DATA_IN_0_PARALLELISM_DIM_0-1:0];
  logic output_reg_valid[DATA_IN_0_PARALLELISM_DIM_0-1:0];

  /* verilator lint_on UNUSEDSIGNAL */

  matrix_fifo #(
      .DATA_WIDTH(DATA_IN_0_WIDTH),
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
      .IN_WIDTH(DATA_IN_0_WIDTH)
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
      .IN_WIDTH(ACC_OUT_WIDTH + ACC_OUT_WIDTH + 1),
      .IN_FRAC_WIDTH(DATA_IN_0_FRAC_WIDTH + ACC_OUT_WIDTH),
      .OUT_WIDTH(DATA_IN_0_WIDTH),
      .OUT_FRAC_WIDTH(DATA_IN_0_FRAC_WIDTH),
      .SYMMETRIC(0),
      .ROUND_FLOOR(1)
  ) acc_div_cast_i (
      .in_data (mu_acc_div),
      .out_data(mu_in)
  );

  single_element_repeat #(
      .DATA_WIDTH(DATA_IN_0_WIDTH),
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

  mxint_isqrt_lut #(
      .IN_WIDTH(VARIANCE_WIDTH),
      .IN_FRAC_WIDTH(VARIANCE_FRAC_WIDTH),
      .VARIANCE_MAN_WIDTH(ISQRT_IN_WIDTH),
      .OUT_MAN_WIDTH(ISQRT_OUT_WIDTH),
      .OUT_MAN_FRAC_WIDTH(ISQRT_OUT_FRAC_WIDTH),
      .EXP_WIDTH(DATA_OUT_0_EXP_WIDTH)
  ) isqrt_lut_inst (
      .clk(clk),
      .rst(rst),
      .data_in_0 (variance_out),
      .data_in_0_valid(variance_out_valid),
      .data_in_0_ready(variance_out_ready),
      .mdata_out_0(minv_sqrt_out),
      .edata_out_0(einv_sqrt_out),
      .data_out_0_valid(inv_sqrt_out_valid),
      .data_out_0_ready(inv_sqrt_out_ready)
  );


  single_element_repeat #(
      .DATA_WIDTH(ISQRT_OUT_WIDTH + DATA_OUT_0_EXP_WIDTH),
      .REPEAT(NUM_ITERS)
  ) isqrt_var_circ_buffer (
      .clk(clk),
      .rst(rst),
      .in_data({minv_sqrt_out, einv_sqrt_out}),
      .in_valid(inv_sqrt_out_valid),
      .in_ready(inv_sqrt_out_ready),
      .out_data({misqrt_circ, eisqrt_circ}),
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
    assign mnorm_in_data[i] = $signed({1'b0, misqrt_circ}) * $signed(diff_batch_out[i]);
    assign enorm_in_data[i] = eisqrt_circ;

    skid_buffer #(
        .DATA_WIDTH(NORM_WIDTH + DATA_OUT_0_EXP_WIDTH)
    ) norm_reg (
        .clk(clk),
        .rst(rst),
        .data_in({mnorm_in_data[i], enorm_in_data[i]}),
        .data_in_valid(norm_in_valid),
        .data_in_ready(norm_in_ready[i]),
        .data_out({mnorm_out_data[i], enorm_out_data[i]}),
        .data_out_valid(norm_out_valid[i]),
        .data_out_ready(norm_batch_ready[i])
    );

    // Output Rounding Stage
    fixed_signed_cast #(
        .IN_WIDTH(NORM_WIDTH),
        .IN_FRAC_WIDTH(NORM_FRAC_WIDTH),
        .OUT_WIDTH(DATA_OUT_0_WIDTH),
        .OUT_FRAC_WIDTH(DATA_OUT_0_FRAC_WIDTH),
        .SYMMETRIC(0),
        .ROUND_FLOOR(1)
    ) output_cast (
        .in_data (mnorm_out_data[i]),
        .out_data(mnorm_round_out[i])
    );
    assign enorm_round_out[i] = enorm_out_data[i];

    skid_buffer #(
        .DATA_WIDTH(DATA_OUT_0_WIDTH + DATA_OUT_0_EXP_WIDTH)
    ) output_reg (
        .clk(clk),
        .rst(rst),
        .data_in({mnorm_round_out[i], enorm_round_out[i]}),
        .data_in_valid(norm_out_valid[i]),
        .data_in_ready(norm_batch_ready[i]),
        .data_out({mnorm_batch_data[i], enorm_batch_data[i]}),
        .data_out_valid(output_reg_valid[i]),
        .data_out_ready(output_reg_ready)
    );
  end

  // Final connection to output
  assign mdata_out_0 = mnorm_batch_data;
  assign edata_out_0 = enorm_batch_data[0];
  assign data_out_0_valid = output_reg_valid[0];
  assign output_reg_ready = data_out_0_ready;

endmodule

module dim_0_cast #(
    parameter MAN_WIDTH = 8,
    parameter EXP_WIDTH = 4,
    parameter IN_DEPTH = 10,
    parameter BLOCK_SIZE = 4
) (
    input logic clk,
    input logic rst,
    input logic data_in_0_valid,
    output logic data_in_0_ready,
    input logic [MAN_WIDTH-1:0] mdata_in_0[BLOCK_SIZE-1:0],
    input logic [EXP_WIDTH-1:0] edata_in_0,

    output logic data_out_0_valid,
    input logic data_out_0_ready,
    output logic [MAN_WIDTH-1:0] mdata_out_0[BLOCK_SIZE-1:0],
    output logic [EXP_WIDTH-1:0] edata_out_0
); 

    // Internal signals
    logic [MAN_WIDTH-1:0] mdata_in_0_fifo[BLOCK_SIZE-1:0];
    logic [EXP_WIDTH-1:0] edata_in_0_fifo;
    logic data_in_0_fifo_valid;
    logic data_in_0_fifo_ready;
    
    logic [EXP_WIDTH-1:0] edata_in_0_straight;
    logic data_in_0_straight_valid;
    logic data_in_0_straight_ready;
    
    logic [EXP_WIDTH-1:0] max_edata_in_0;
    logic max_edata_in_0_valid;
    logic max_edata_in_0_ready;
    
    logic [EXP_WIDTH-1:0] circular_max_edata_in_0 [0:0];
    logic circular_max_edata_in_0_valid;
    logic circular_max_edata_in_0_ready;

    logic signed [EXP_WIDTH:0] shift_value;

    // Split2 circuit for parallel processing
    unpacked_mx_split2_with_data #(
        .DEPTH(IN_DEPTH),
        .MAN_WIDTH(MAN_WIDTH),
        .EXP_WIDTH(EXP_WIDTH),
        .IN_SIZE(BLOCK_SIZE)
    ) split2_circ (
        .clk(clk),
        .rst(rst),
        // Input from circular buffer
        .mdata_in(mdata_in_0),
        .edata_in(edata_in_0),
        .data_in_valid(data_in_0_valid),
        .data_in_ready(data_in_0_ready),
        // FIFO output path (not used)
        .fifo_mdata_out(mdata_in_0_fifo),
        .fifo_edata_out(edata_in_0_fifo),
        .fifo_data_out_valid(data_in_0_fifo_valid),
        .fifo_data_out_ready(data_in_0_fifo_ready),
        // Straight output path
        .straight_mdata_out(),  // Connect to the same signals previously used
        .straight_edata_out(edata_in_0_straight),
        .straight_data_out_valid(data_in_0_straight_valid),
        .straight_data_out_ready(data_in_0_straight_ready)
    );

    // Sequential max finder
    sequential_max #(
        .IN_DEPTH(IN_DEPTH),
        .IN_WIDTH(EXP_WIDTH)
    ) sequential_max_inst (
        .clk            (clk),             // input
        .rst            (rst),             // input
        .data_in        (edata_in_0_straight),         // input  [IN_WIDTH-1:0]
        .data_in_valid  (data_in_0_straight_valid),   // input
        .data_in_ready  (data_in_0_straight_ready),   // output
        .data_out       (max_edata_in_0),        // output [IN_WIDTH-1:0]
        .data_out_valid (max_edata_in_0_valid),  // output
        .data_out_ready (max_edata_in_0_ready)   // input
    );

  input_buffer #(
      .DATA_WIDTH (EXP_WIDTH),
      .IN_NUM     (1),
      .REPEAT     (IN_DEPTH),
      .BUFFER_SIZE(1)
  ) mdata_in_0_buffer (
      .clk,
      .rst,
      // Input streaming port
      .data_in({max_edata_in_0}),
      .data_in_valid(max_edata_in_0_valid),
      .data_in_ready(max_edata_in_0_ready),
      // Output streaming port
      .data_out(circular_max_edata_in_0),
      .data_out_valid(circular_max_edata_in_0_valid),
      .data_out_ready(circular_max_edata_in_0_ready)
  );

    // Join circuit for output synchronization
    join2 data_out_join_inst (
        .data_in_ready({circular_max_edata_in_0_ready, data_in_0_fifo_ready}),
        .data_in_valid({circular_max_edata_in_0_valid, data_in_0_fifo_valid}),
        .data_out_valid(data_out_0_valid),
        .data_out_ready(data_out_0_ready)
    );

    // Calculate shift value and perform optimized right shift
    assign shift_value = $signed(max_edata_in_0) - $signed(circular_max_edata_in_0[0]);

    optimized_right_shift #(
        .IN_WIDTH(MAN_WIDTH),
        .SHIFT_WIDTH(EXP_WIDTH),
        .OUT_WIDTH(MAN_WIDTH),
        .BLOCK_SIZE(BLOCK_SIZE)
    ) ovshift_inst (
        .data_in(mdata_in_0_fifo),
        .shift_value(shift_value),
        .data_out(mdata_out_0)
    );
    // Assign final exponent output
    assign edata_out_0 = max_edata_in_0;
endmodule
