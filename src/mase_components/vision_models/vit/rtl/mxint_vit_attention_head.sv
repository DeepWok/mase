`timescale 1ns / 1ps
module mxint_vit_attention_head #(

    // * Queries, keys and values are assumed to have the same
    // * precision, dimensions and parallelism
    parameter IN_DATA_TENSOR_SIZE_DIM_0 = 32,
    parameter IN_DATA_TENSOR_SIZE_DIM_1 = 10,
    parameter IN_DATA_PARALLELISM_DIM_0 = 2,
    parameter IN_DATA_PARALLELISM_DIM_1 = 2,
    parameter IN_DATA_PRECISION_0 = 16,
    parameter IN_DATA_PRECISION_1 = 3,

    // * Output tokens are casted to requested precision
    parameter OUT_DATA_TENSOR_SIZE_DIM_0 = IN_DATA_TENSOR_SIZE_DIM_0,
    parameter OUT_DATA_TENSOR_SIZE_DIM_1 = IN_DATA_TENSOR_SIZE_DIM_1,
    parameter OUT_DATA_PARALLELISM_DIM_0 = IN_DATA_PARALLELISM_DIM_0,
    parameter OUT_DATA_PARALLELISM_DIM_1 = IN_DATA_PARALLELISM_DIM_1,

    parameter QKMM_OUT_PRECISION_0 = 16,
    parameter QKMM_OUT_PRECISION_1 = 16,
    parameter SOFTMAX_EXP_PRECISION_0 = 16,
    parameter SOFTMAX_EXP_PRECISION_1 = 16,
    parameter SOFTMAX_OUT_DATA_PRECISION_1 = 7,
    parameter SOFTMAX_OUT_DATA_PRECISION_0 = SOFTMAX_OUT_DATA_PRECISION_1 + 2,
    parameter OUT_DATA_PRECISION_0 = 16,
    parameter OUT_DATA_PRECISION_1 = 3

) (
    input logic clk,
    input logic rst,

    input logic [IN_DATA_PRECISION_0-1:0] query [IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_1-1:0],
    input logic [IN_DATA_PRECISION_1-1:0] query_exp,
    input logic query_valid,
    output logic query_ready,

    input logic [IN_DATA_PRECISION_0-1:0] key [IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_1-1:0],
    input logic [IN_DATA_PRECISION_1-1:0] key_exp,
    input logic key_valid,
    output logic key_ready,

    input logic [IN_DATA_PRECISION_0-1:0] value [IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_1-1:0],
    input logic [IN_DATA_PRECISION_1-1:0] value_exp,
    input logic value_valid,
    output logic value_ready,

    output logic [OUT_DATA_PRECISION_0-1:0] out [OUT_DATA_PARALLELISM_DIM_0*OUT_DATA_PARALLELISM_DIM_1-1:0],
    output logic [OUT_DATA_PRECISION_1-1:0] out_exp,
    output logic out_valid,
    input logic out_ready
);

  initial begin
    assert (OUT_DATA_TENSOR_SIZE_DIM_0 == IN_DATA_TENSOR_SIZE_DIM_0)
    else
      $fatal(
          "Module incorrectly parametrized. OUT_DATA_TENSOR_SIZE_DIM_0 != IN_DATA_TENSOR_SIZE_DIM_0"
      );

    assert (OUT_DATA_TENSOR_SIZE_DIM_1 == IN_DATA_TENSOR_SIZE_DIM_1)
    else
      $fatal(
          "Module incorrectly parametrized. OUT_DATA_TENSOR_SIZE_DIM_1 != IN_DATA_TENSOR_SIZE_DIM_1"
      );

    assert (OUT_DATA_PARALLELISM_DIM_0 == IN_DATA_PARALLELISM_DIM_0)
    else
      $fatal(
          "Parallelism conversion not yet supported. OUT_DATA_PARALLELISM_DIM_0 != IN_DATA_PARALLELISM_DIM_0"
      );

    assert (OUT_DATA_PARALLELISM_DIM_1 == IN_DATA_PARALLELISM_DIM_1)
    else
      $fatal(
          "Parallelism conversion not yet supported. OUT_DATA_PARALLELISM_DIM_1 != IN_DATA_PARALLELISM_DIM_1"
      );
  end

  parameter IN_DATA_DEPTH_0 = IN_DATA_TENSOR_SIZE_DIM_0 / IN_DATA_PARALLELISM_DIM_0;
  parameter IN_DATA_DEPTH_1 = IN_DATA_TENSOR_SIZE_DIM_1 / IN_DATA_PARALLELISM_DIM_1;

  // Query key transpose
  parameter QUERY_TRANSPOSE_PRECISION_0 = 2 * IN_DATA_PRECISION_0 + $clog2(
      IN_DATA_PARALLELISM_DIM_0
  ) + $clog2(
      IN_DATA_DEPTH_1
  );
  parameter QUERY_TRANSPOSE_PRECISION_1 = 2 * IN_DATA_PRECISION_1;

  // Attention scores
  // ! TO DO: check precision transformation post softmax
  parameter ATTENTION_SCORES_PRECISION_0 = QUERY_TRANSPOSE_PRECISION_0;
  parameter ATTENTION_SCORES_PRECISION_1 = QUERY_TRANSPOSE_PRECISION_1;

  parameter OUT_PRE_CAST_PRECISION_0 = IN_DATA_PRECISION_0 + ATTENTION_SCORES_PRECISION_0 + $clog2(
      IN_DATA_PARALLELISM_DIM_1
  ) + $clog2(
      IN_DATA_TENSOR_SIZE_DIM_1 / IN_DATA_PARALLELISM_DIM_1
  );
  parameter OUT_PRE_CAST_PRECISION_1 = IN_DATA_PRECISION_1 + ATTENTION_SCORES_PRECISION_1;

  // * Declarations
  // * =================================================================

  logic [IN_DATA_PRECISION_0-1:0] key_transpose [IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_1-1:0];
  logic [IN_DATA_PRECISION_1-1:0] key_transpose_exp;
  logic key_transpose_valid;
  logic key_transpose_ready;

  logic [QKMM_OUT_PRECISION_0-1:0] query_key_transpose [IN_DATA_PARALLELISM_DIM_1 * IN_DATA_PARALLELISM_DIM_1-1:0];
  logic [QKMM_OUT_PRECISION_1-1:0] query_key_transpose_exp;
  logic query_key_transpose_valid;
  logic query_key_transpose_ready;

  logic [QKMM_OUT_PRECISION_0-1:0] buffered_query_key_transpose [IN_DATA_PARALLELISM_DIM_1 * IN_DATA_PARALLELISM_DIM_1-1:0];
  logic buffered_query_key_transpose_valid;
  logic buffered_query_key_transpose_ready;

  logic [SOFTMAX_OUT_DATA_PRECISION_0 - 1:0] attention_scores [IN_DATA_PARALLELISM_DIM_1 * IN_DATA_PARALLELISM_DIM_1-1:0];
  logic [SOFTMAX_OUT_DATA_PRECISION_1-1:0] attention_scores_exp;
  logic attention_scores_valid;
  logic attention_scores_ready;

  logic [OUT_DATA_PRECISION_0-1:0] out_casted [OUT_DATA_PARALLELISM_DIM_0*OUT_DATA_PARALLELISM_DIM_1-1:0];
  logic out_cast_valid;
  logic out_cast_ready;


  logic [OUT_DATA_PRECISION_0-1:0] buffer_out [OUT_DATA_PARALLELISM_DIM_0*OUT_DATA_PARALLELISM_DIM_1-1:0];
  logic buffer_out_valid;
  logic buffer_out_ready;
  // * Instances
  // * =================================================================

  // * Transpose projected keys

  mxint_matrix_stream_transpose #(
      .TOTAL_DIM0  (IN_DATA_TENSOR_SIZE_DIM_0),
      .TOTAL_DIM1  (IN_DATA_TENSOR_SIZE_DIM_1),
      .COMPUTE_DIM0(IN_DATA_PARALLELISM_DIM_0),
      .COMPUTE_DIM1(IN_DATA_PARALLELISM_DIM_1),

      .MAN_WIDTH(IN_DATA_PRECISION_0),
      .EXP_WIDTH(IN_DATA_PRECISION_1)
  ) key_transpose_i (
      .clk,
      .rst,

      // In Matrix
      .mdata_in (key),
      .edata_in(key_exp),
      .data_in_valid(key_valid),
      .data_in_ready(key_ready),

      // Out Matrix
      .mdata_out (key_transpose),
      .edata_out(key_transpose_exp),
      .data_out_valid(key_transpose_valid),
      .data_out_ready(key_transpose_ready)
  );

  // * Query x Key^T

  mxint_matmul #(
      .A_TOTAL_DIM0(IN_DATA_TENSOR_SIZE_DIM_0),
      .A_TOTAL_DIM1(IN_DATA_TENSOR_SIZE_DIM_1),

      .B_TOTAL_DIM0(IN_DATA_TENSOR_SIZE_DIM_1),
      .B_TOTAL_DIM1(IN_DATA_TENSOR_SIZE_DIM_0),

      .A_COMPUTE_DIM0(IN_DATA_PARALLELISM_DIM_0),
      .A_COMPUTE_DIM1(IN_DATA_PARALLELISM_DIM_1),
      .B_COMPUTE_DIM0(IN_DATA_PARALLELISM_DIM_1),
      .B_COMPUTE_DIM1(IN_DATA_PARALLELISM_DIM_0),

      .A_MAN_WIDTH     (IN_DATA_PRECISION_0),
      .A_EXP_WIDTH(IN_DATA_PRECISION_1),

      .B_MAN_WIDTH     (IN_DATA_PRECISION_0),
      .B_EXP_WIDTH(IN_DATA_PRECISION_1),

      .OUT_MAN_WIDTH     (QKMM_OUT_PRECISION_0),
      .OUT_EXP_WIDTH(QKMM_OUT_PRECISION_1)

  ) query_key_transpose_matmul_i (
      .clk,
      .rst,

      .ma_data (query),
      .ea_data(query_exp),
      .a_valid(query_valid),
      .a_ready(query_ready),

      .mb_data (key_transpose),
      .eb_data(key_transpose_exp),
      .b_valid(key_transpose_valid),
      .b_ready(key_transpose_ready),

      .mout_data (query_key_transpose),
      .eout_data(query_key_transpose_exp),
      .out_valid(query_key_transpose_valid),
      .out_ready(query_key_transpose_ready)
  );


    //cut the long ready path
    unpacked_skid_buffer #(
        .DATA_WIDTH(QKMM_OUT_PRECISION_0),
        .IN_NUM    (IN_DATA_PARALLELISM_DIM_1 * IN_DATA_PARALLELISM_DIM_1)
    ) input_stream_reg_slice (
        .clk           (clk),
        .rst           (rst),
        .data_in       (query_key_transpose),
        .data_in_valid (query_key_transpose_valid),
        .data_in_ready (query_key_transpose_ready),
        .data_out      (buffered_query_key_transpose),
        .data_out_valid(buffered_query_key_transpose_valid),
        .data_out_ready(buffered_query_key_transpose_ready)
    );
  mxint_softmax #(
      .DATA_IN_0_PRECISION_0      (QKMM_OUT_PRECISION_0),
      .DATA_IN_0_PRECISION_1      (QKMM_OUT_PRECISION_1),
      .DATA_EXP_0_PRECISION_0     (SOFTMAX_EXP_PRECISION_0),
      .DATA_EXP_0_PRECISION_1     (SOFTMAX_EXP_PRECISION_1),
      .DATA_OUT_0_PRECISION_0     (SOFTMAX_OUT_DATA_PRECISION_0),
      .DATA_OUT_0_PRECISION_1     (SOFTMAX_OUT_DATA_PRECISION_1),
      .DATA_IN_0_TENSOR_SIZE_DIM_0(IN_DATA_TENSOR_SIZE_DIM_1),
      .DATA_IN_0_TENSOR_SIZE_DIM_1(IN_DATA_TENSOR_SIZE_DIM_1),
      .DATA_IN_0_PARALLELISM_DIM_0(IN_DATA_PARALLELISM_DIM_1),
      .DATA_IN_0_PARALLELISM_DIM_1(IN_DATA_PARALLELISM_DIM_1)
  ) fixed_softmax_i (
      .clk,
      .rst,

      .mdata_in_0      (buffered_query_key_transpose),
      .edata_in_0      (query_key_transpose_exp),
      .data_in_0_valid(buffered_query_key_transpose_valid),
      .data_in_0_ready(buffered_query_key_transpose_ready),

      .mdata_out_0      (attention_scores),
      .edata_out_0      (attention_scores_exp),
      .data_out_0_valid(attention_scores_valid),
      .data_out_0_ready(attention_scores_ready)
  );
  //   end

  // * Output: Attention scores x Value

  mxint_matmul #(
      .A_TOTAL_DIM0(IN_DATA_TENSOR_SIZE_DIM_1),
      .A_TOTAL_DIM1(IN_DATA_TENSOR_SIZE_DIM_1),

      .B_TOTAL_DIM0(IN_DATA_TENSOR_SIZE_DIM_0),
      .B_TOTAL_DIM1(IN_DATA_TENSOR_SIZE_DIM_1),

      .A_COMPUTE_DIM0(IN_DATA_PARALLELISM_DIM_1),
      .A_COMPUTE_DIM1(IN_DATA_PARALLELISM_DIM_1),
      .B_COMPUTE_DIM0(IN_DATA_PARALLELISM_DIM_0),
      .B_COMPUTE_DIM1(IN_DATA_PARALLELISM_DIM_1),

      .A_MAN_WIDTH     (SOFTMAX_OUT_DATA_PRECISION_0),
      .A_EXP_WIDTH(SOFTMAX_OUT_DATA_PRECISION_1),

      .B_MAN_WIDTH     (IN_DATA_PRECISION_0),
      .B_EXP_WIDTH(IN_DATA_PRECISION_1),

      .OUT_MAN_WIDTH     (OUT_DATA_PRECISION_0),
      .OUT_EXP_WIDTH(OUT_DATA_PRECISION_1)

  ) attention_scores_values_matmul_i (
      .clk,
      .rst,

      .ma_data (attention_scores),
      .ea_data(attention_scores_exp),
      .a_valid(attention_scores_valid),
      .a_ready(attention_scores_ready),

      .mb_data (value),
      .eb_data(value_exp),
      .b_valid(value_valid),
      .b_ready(value_ready),

      .mout_data (out),
      .eout_data(out_exp),
      .out_valid(out_valid),
      .out_ready(out_ready)
  );


    fifo_for_autogen #(
        .DATA_IN_0_PRECISION_0(OUT_DATA_PRECISION_0), // = 8
        .DATA_IN_0_PRECISION_1(OUT_DATA_PRECISION_1), // = 4
        .DATA_IN_0_TENSOR_SIZE_DIM_0(OUT_DATA_TENSOR_SIZE_DIM_0), // = 20
        .DATA_IN_0_PARALLELISM_DIM_0(OUT_DATA_PARALLELISM_DIM_0), // = 2
        .DATA_IN_0_TENSOR_SIZE_DIM_1(OUT_DATA_TENSOR_SIZE_DIM_1), // = 4
        .DATA_IN_0_PARALLELISM_DIM_1(OUT_DATA_PARALLELISM_DIM_1), // = 2
        .DEPTH(OUT_DATA_TENSOR_SIZE_DIM_0/OUT_DATA_PARALLELISM_DIM_0), 
        .DATA_OUT_0_PRECISION_0(OUT_DATA_PRECISION_0), 
        .DATA_OUT_0_PRECISION_1(OUT_DATA_PRECISION_1),
        .DATA_OUT_0_TENSOR_SIZE_DIM_0(OUT_DATA_TENSOR_SIZE_DIM_0), 
        .DATA_OUT_0_PARALLELISM_DIM_0(OUT_DATA_PARALLELISM_DIM_0), 
        .DATA_OUT_0_TENSOR_SIZE_DIM_1(OUT_DATA_TENSOR_SIZE_DIM_1), 
        .DATA_OUT_0_PARALLELISM_DIM_1(OUT_DATA_PARALLELISM_DIM_1)
    ) fifo_1_inst (
        .clk(clk),
        .rst(rst),

        .data_in_0(out_casted),
        .data_in_0_valid(out_cast_valid),
        .data_in_0_ready(out_cast_ready),
        .data_out_0(out),
        .data_out_0_valid(out_valid),
        .data_out_0_ready(out_ready)
    );
endmodule
