`timescale 1ns / 1ps
module fixed_swin_attention_head_programmable #(

    // * Queries, keys and values are assumed to have the same
    // * precision, dimensions and parallelism
    parameter IN_DATA_MAX_TENSOR_SIZE_DIM_0 = 64,
    parameter IN_DATA_MAX_TENSOR_SIZE_DIM_1 = 32,
    parameter IN_DATA_PARALLELISM_DIM_0 = 4,
    parameter IN_DATA_PARALLELISM_DIM_1 = 4,
    parameter IN_DATA_PRECISION_0 = 16,
    parameter IN_DATA_PRECISION_1 = 3,
    localparam IN_DATA_MAX_DEPTH_0 = IN_DATA_MAX_TENSOR_SIZE_DIM_0/IN_DATA_PARALLELISM_DIM_0,
    localparam IN_DATA_MAX_DEPTH_1 = IN_DATA_MAX_TENSOR_SIZE_DIM_1/IN_DATA_PARALLELISM_DIM_1,
    localparam IN_DATA_MAX_DEPTH_0_WIDTH = $clog2(IN_DATA_MAX_DEPTH_0),
    localparam IN_DATA_MAX_DEPTH_1_WIDTH = $clog2(IN_DATA_MAX_DEPTH_1),
    localparam IN_DATA_MAX_DEPTH_MULT_WIDTH = $clog2(IN_DATA_MAX_DEPTH_0) + $clog2(IN_DATA_MAX_DEPTH_1),

    // * Output tokens are casted to requested precision
    parameter OUT_DATA_MAX_TENSOR_SIZE_DIM_0 = 64,
    parameter OUT_DATA_MAX_TENSOR_SIZE_DIM_1 = 32,
    parameter OUT_DATA_PARALLELISM_DIM_0 = IN_DATA_PARALLELISM_DIM_0,
    parameter OUT_DATA_PARALLELISM_DIM_1 = IN_DATA_PARALLELISM_DIM_1,
    parameter OUT_DATA_PRECISION_0 = 16,
    parameter OUT_DATA_PRECISION_1 = 3

) (
    input logic clk,
    input logic rst,

    input logic [IN_DATA_MAX_DEPTH_0_WIDTH:0] in_depth_dim_0,
    input logic [IN_DATA_MAX_DEPTH_1_WIDTH:0] in_depth_dim_1,
    input logic [IN_DATA_MAX_DEPTH_MULT_WIDTH:0] in_depth_mult,

    input logic [IN_DATA_PRECISION_0-1:0] query_con [IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_1-1:0],
    input logic query_con_valid,
    output logic query_con_ready,

    input logic [IN_DATA_PRECISION_0-1:0] query_pos [IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_1-1:0],
    input logic query_pos_valid,
    output logic query_pos_ready,

    input logic [IN_DATA_PRECISION_0-1:0] key [IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_1-1:0],
    input logic key_valid,
    output logic key_ready,

    input logic [IN_DATA_PRECISION_0-1:0] value [IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_1-1:0],
    input logic value_valid,
    output logic value_ready,

    input logic [IN_DATA_PRECISION_0-1:0] pos_embed [IN_DATA_PARALLELISM_DIM_1*IN_DATA_PARALLELISM_DIM_0**2-1:0],
    input logic pos_embed_valid,
    output logic pos_embed_ready,

    output logic [OUT_DATA_PRECISION_0-1:0] out [OUT_DATA_PARALLELISM_DIM_0*OUT_DATA_PARALLELISM_DIM_1-1:0],
    output logic out_valid,
    input logic out_ready
);

  initial begin
    // assert (OUT_DATA_TENSOR_SIZE_DIM_0 == IN_DATA_TENSOR_SIZE_DIM_0)
    // else
    //   $fatal(
    //       "Module incorrectly parametrized. OUT_DATA_TENSOR_SIZE_DIM_0 != IN_DATA_TENSOR_SIZE_DIM_0"
    //   );

    // assert (OUT_DATA_TENSOR_SIZE_DIM_1 == IN_DATA_TENSOR_SIZE_DIM_1)
    // else
    //   $fatal(
    //       "Module incorrectly parametrized. OUT_DATA_TENSOR_SIZE_DIM_1 != IN_DATA_TENSOR_SIZE_DIM_1"
    //   );

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

//   parameter IN_DATA_DEPTH_0 = IN_DATA_TENSOR_SIZE_DIM_0 / IN_DATA_PARALLELISM_DIM_0;
//   parameter IN_DATA_DEPTH_1 = IN_DATA_TENSOR_SIZE_DIM_1 / IN_DATA_PARALLELISM_DIM_1;

//   // Query key transpose
//   parameter QUERY_TRANSPOSE_PRECISION_0 = 2 * IN_DATA_PRECISION_0 + $clog2(
//       IN_DATA_PARALLELISM_DIM_0
//   ) + $clog2(
//       IN_DATA_DEPTH_1
//   );
//   parameter QUERY_TRANSPOSE_PRECISION_1 = 2 * IN_DATA_PRECISION_1;

  // Attention scores
  // ! TO DO: check precision transformation post softmax
//   parameter ATTENTION_SCORES_PRECISION_0 = QUERY_TRANSPOSE_PRECISION_0;
//   parameter ATTENTION_SCORES_PRECISION_1 = QUERY_TRANSPOSE_PRECISION_1;

//   parameter OUT_PRE_CAST_PRECISION_0 = IN_DATA_PRECISION_0 + ATTENTION_SCORES_PRECISION_0 + $clog2(
//       IN_DATA_PARALLELISM_DIM_1
//   ) + $clog2(
//       IN_DATA_TENSOR_SIZE_DIM_1 / IN_DATA_PARALLELISM_DIM_1
//   );
//   parameter OUT_PRE_CAST_PRECISION_1 = IN_DATA_PRECISION_1 + ATTENTION_SCORES_PRECISION_1;

  // * Declarations
  // * =================================================================

  logic [IN_DATA_PRECISION_0-1:0] key_transpose [IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_1-1:0];
  logic key_transpose_valid;
  logic key_transpose_ready;

  logic [OUT_DATA_PRECISION_0-1:0] content_att [IN_DATA_PARALLELISM_DIM_0 * IN_DATA_PARALLELISM_DIM_1-1:0];
  logic content_att_valid;
  logic content_att_ready;

  logic [OUT_DATA_PRECISION_0-1:0] positional_att [IN_DATA_PARALLELISM_DIM_0 *IN_DATA_PARALLELISM_DIM_1-1:0];
  logic positional_att_valid;
  logic positional_att_ready;

  logic [OUT_DATA_PRECISION_0-1:0] sum_att [IN_DATA_PARALLELISM_DIM_1 * IN_DATA_PARALLELISM_DIM_0-1:0];
  logic sum_att_valid;
  logic sum_att_ready;

  logic [OUT_DATA_PRECISION_0-1:0] attention_scores [IN_DATA_PARALLELISM_DIM_1 * IN_DATA_PARALLELISM_DIM_0-1:0];
  logic attention_scores_valid;
  logic attention_scores_ready;

  logic [OUT_DATA_PRECISION_0-1:0] out_pre_cast [OUT_DATA_PARALLELISM_DIM_0*OUT_DATA_PARALLELISM_DIM_1-1:0];
  logic [OUT_DATA_PRECISION_0-1:0] out_casted [OUT_DATA_PARALLELISM_DIM_0*OUT_DATA_PARALLELISM_DIM_1-1:0];
  logic out_cast_valid;
  logic out_cast_ready;

  logic [OUT_DATA_PRECISION_0-1:0] query_pos_rows [IN_DATA_PARALLELISM_DIM_0][IN_DATA_PARALLELISM_DIM_1]; 
  logic [OUT_DATA_PRECISION_0-1:0] pos_embed_rows [IN_DATA_PARALLELISM_DIM_0][IN_DATA_PARALLELISM_DIM_1*IN_DATA_PARALLELISM_DIM_0];
  logic [OUT_DATA_PRECISION_0-1:0] positional_att_rows [IN_DATA_PARALLELISM_DIM_0][IN_DATA_PARALLELISM_DIM_1];


  // * Instances
  // * =================================================================

  // * Transpose projected keys



  matrix_stream_transpose_programmable #(
      .TOTAL_MAX_DIM0  (IN_DATA_MAX_TENSOR_SIZE_DIM_0),
      .TOTAL_MAX_DIM1  (IN_DATA_MAX_TENSOR_SIZE_DIM_1),
      .COMPUTE_DIM0(IN_DATA_PARALLELISM_DIM_0),
      .COMPUTE_DIM1(IN_DATA_PARALLELISM_DIM_1),

      .DATA_WIDTH(IN_DATA_PRECISION_0)
  ) key_transpose_i (
      .clk,
      .rst,

      .in_depth_dim_0(in_depth_dim_0),
      .in_depth_dim_1(in_depth_dim_1),

      // In Matrix
      .in_data (key),
      .in_valid(key_valid),
      .in_ready(key_ready),

      // Out Matrix
      .out_data (key_transpose),
      .out_valid(key_transpose_valid),
      .out_ready(key_transpose_ready)
  );

  // * Content Query x Key^T

  matmul_programmable #(
      .A_MAX_DIM0(IN_DATA_MAX_TENSOR_SIZE_DIM_0),
      .A_MAX_DIM1(IN_DATA_MAX_TENSOR_SIZE_DIM_1),

      .B_MAX_DIM0(IN_DATA_MAX_TENSOR_SIZE_DIM_1),
      .B_MAX_DIM1(IN_DATA_MAX_TENSOR_SIZE_DIM_0),

      .A_COMPUTE_DIM0(IN_DATA_PARALLELISM_DIM_0),
      .A_COMPUTE_DIM1(IN_DATA_PARALLELISM_DIM_1),
      .B_COMPUTE_DIM0(IN_DATA_PARALLELISM_DIM_1),
      .B_COMPUTE_DIM1(IN_DATA_PARALLELISM_DIM_0),

      .A_WIDTH     (IN_DATA_PRECISION_0),
      .A_FRAC_WIDTH(IN_DATA_PRECISION_1),

      .B_WIDTH     (IN_DATA_PRECISION_0),
      .B_FRAC_WIDTH(IN_DATA_PRECISION_1),

      .OUT_WIDTH     (OUT_DATA_PRECISION_0),
      .OUT_FRAC_WIDTH(OUT_DATA_PRECISION_1)

  ) content_att_matmul_i (
      .clk,
      .rst,

      .a_depth_dim1(in_depth_dim_1),
      .b_depth_dim0(in_depth_dim_0),
      .b_depth_dim1(in_depth_dim_1),
      .b_depth_mult(in_depth_mult),

      .a_data (query_con),
      .a_valid(query_con_valid),
      .a_ready(query_con_ready),

      .b_data (key_transpose),
      .b_valid(key_transpose_valid),
      .b_ready(key_transpose_ready),

      .out_data (content_att),
      .out_valid(content_att_valid),
      .out_ready(content_att_ready)
  );


  //   //Positonal Query * Positional Embedding (perhaps transpose rel_k?)
  //   matmul_programmable #(
  //     .A_MAX_DIM0(1),
  //     .A_MAX_DIM1(IN_DATA_MAX_TENSOR_SIZE_DIM_0),

  //     .B_MAX_DIM0(IN_DATA_MAX_TENSOR_SIZE_DIM_1),
  //     .B_MAX_DIM1(IN_DATA_MAX_TENSOR_SIZE_DIM_0),

  //     .A_COMPUTE_DIM0(1),
  //     .A_COMPUTE_DIM1(IN_DATA_PARALLELISM_DIM_1),
  //     .B_COMPUTE_DIM0(IN_DATA_PARALLELISM_DIM_1),
  //     .B_COMPUTE_DIM1(IN_DATA_PARALLELISM_DIM_0),

  //     .A_WIDTH     (IN_DATA_PRECISION_0),
  //     .A_FRAC_WIDTH(IN_DATA_PRECISION_1),

  //     .B_WIDTH     (IN_DATA_PRECISION_0),
  //     .B_FRAC_WIDTH(IN_DATA_PRECISION_1),

  //     .OUT_WIDTH     (OUT_DATA_PRECISION_0),
  //     .OUT_FRAC_WIDTH(OUT_DATA_PRECISION_1)

  //   ) positional_att_matmul_i (
  //     .clk,
  //     .rst,

  //     .a_data (query_pos),
  //     .a_valid(query_pos_valid),
  //     .a_ready(query_pos_ready),

  //     .b_data (pos_embed),
  //     .b_valid(pos_embed_valid),
  //     .b_ready(pos_embed_ready),

  //     .out_data (positional_att),
  //     .out_valid(positional_att_valid),
  //     .out_ready(positional_att_ready)
  // );

  always_comb begin 
    for (int i = 0; i < IN_DATA_PARALLELISM_DIM_1; i++) begin
      for (int j = 0; j < IN_DATA_PARALLELISM_DIM_0; j ++) begin
        query_pos_rows [i][j] = query_pos[i * IN_DATA_PARALLELISM_DIM_0 + j];
      end
    end   
  end

  always_comb begin 
    for (int i = 0; i < IN_DATA_PARALLELISM_DIM_1 ; i++) begin
      for (int j = 0; j < IN_DATA_PARALLELISM_DIM_0* IN_DATA_PARALLELISM_DIM_0; j ++) begin
        pos_embed_rows [i][j] = pos_embed[i * IN_DATA_PARALLELISM_DIM_0* IN_DATA_PARALLELISM_DIM_0 + j];
      end
    end   
  end
  
  always_comb begin
    for (int i = 0; i < IN_DATA_PARALLELISM_DIM_1; i++) begin
      for (int j = 0; j < IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_0; j++) begin
        positional_att[i * IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_0 + j] = positional_att_rows[i][j];
      end
    end 
  end

  for (genvar i = 0; i < IN_DATA_PARALLELISM_DIM_1; i++) begin
    matmul_programmable #(
      .A_MAX_DIM0(1),
      .A_MAX_DIM1(IN_DATA_MAX_TENSOR_SIZE_DIM_0),

      .B_MAX_DIM0(IN_DATA_MAX_TENSOR_SIZE_DIM_1),
      .B_MAX_DIM1(IN_DATA_MAX_TENSOR_SIZE_DIM_0),

      .A_COMPUTE_DIM0(IN_DATA_PARALLELISM_DIM_0),
      .A_COMPUTE_DIM1(1),
      .B_COMPUTE_DIM0(IN_DATA_PARALLELISM_DIM_0),
      .B_COMPUTE_DIM1(IN_DATA_PARALLELISM_DIM_0),

      .A_WIDTH     (IN_DATA_PRECISION_0),
      .A_FRAC_WIDTH(IN_DATA_PRECISION_1),

      .B_WIDTH     (IN_DATA_PRECISION_0),
      .B_FRAC_WIDTH(IN_DATA_PRECISION_1),

      .OUT_WIDTH     (OUT_DATA_PRECISION_0),
      .OUT_FRAC_WIDTH(OUT_DATA_PRECISION_1)
    ) pos_attn_matmul
    (
      .clk(clk),
      .rst(rst),

      .a_depth_dim1(1),
      .b_depth_dim0(in_depth_dim_0),
      .b_depth_dim1(in_depth_dim_0),
      .b_depth_mult(in_depth_dim_0 * in_depth_dim_0),

      .a_data (query_pos_rows[i]),
      .a_valid(query_pos_valid),
      .a_ready(query_pos_ready),

      .b_data (pos_embed_rows[i]),
      .b_valid(pos_embed_valid),
      .b_ready(pos_embed_ready),
      .out_data (positional_att_rows[i]),
      .out_valid(positional_att_valid),
      .out_ready(positional_att_ready)
      // 
    );
  end


  //WRONG TENSOR_SIZE TO BE FIXED
  fixed_adder #(
      .DATA_IN_0_PRECISION_0 (IN_DATA_PRECISION_0),
      .DATA_IN_0_PRECISION_1 (IN_DATA_PRECISION_1),
      .DATA_IN_0_TENSOR_SIZE_DIM_0 (IN_DATA_MAX_TENSOR_SIZE_DIM_0),
      .DATA_IN_0_TENSOR_SIZE_DIM_1 (IN_DATA_MAX_TENSOR_SIZE_DIM_1),
      .DATA_IN_0_TENSOR_SIZE_DIM_2 (0),
      .DATA_IN_0_PARALLELISM_DIM_0 (IN_DATA_PARALLELISM_DIM_0),
      .DATA_IN_0_PARALLELISM_DIM_1 (IN_DATA_PARALLELISM_DIM_1),
      .DATA_IN_0_PARALLELISM_DIM_2 (0),
  
      .DATA_IN_1_PRECISION_0 (IN_DATA_PRECISION_0),
      .DATA_IN_1_PRECISION_1 (IN_DATA_PRECISION_1),
      .DATA_IN_1_TENSOR_SIZE_DIM_0 (IN_DATA_MAX_TENSOR_SIZE_DIM_0),
      .DATA_IN_1_TENSOR_SIZE_DIM_1 (IN_DATA_MAX_TENSOR_SIZE_DIM_1),
      .DATA_IN_1_TENSOR_SIZE_DIM_2 (0),
      .DATA_IN_1_PARALLELISM_DIM_0 (IN_DATA_PARALLELISM_DIM_0),
      .DATA_IN_1_PARALLELISM_DIM_1 (IN_DATA_PARALLELISM_DIM_1),
      .DATA_IN_1_PARALLELISM_DIM_2 (0),

      .DATA_OUT_0_PRECISION_0 (IN_DATA_PRECISION_0),
      .DATA_OUT_0_PRECISION_1 (IN_DATA_PRECISION_1)
  ) adder_sum_attention(
      .clk(clk),
      .rst(rst),

      .data_in_0(positional_att),
      .data_in_0_valid(positional_att_valid),
      .data_in_0_ready(positional_att_ready),

      .data_in_1(content_att),
      .data_in_1_valid(content_att_valid),
      .data_in_1_ready(content_att_ready),

      .data_out_0(sum_att),
      .data_out_0_valid(sum_att_valid),
      .data_out_0_ready(sum_att_ready)
  );

  // ! TO DO: normalize query_key_transpose

  // * Attention scores: softmax(Query x Key^T)

  fixed_softermax #(
      .DATA_IN_0_PRECISION_0      (OUT_DATA_PRECISION_0),
      .DATA_IN_0_PRECISION_1      (OUT_DATA_PRECISION_1),
      .DATA_IN_0_TENSOR_SIZE_DIM_0(IN_DATA_MAX_TENSOR_SIZE_DIM_1),
      .DATA_IN_0_TENSOR_SIZE_DIM_1(IN_DATA_MAX_TENSOR_SIZE_DIM_1),
      .DATA_IN_0_PARALLELISM_DIM_0(IN_DATA_PARALLELISM_DIM_0),
      .DATA_IN_0_PARALLELISM_DIM_1(IN_DATA_PARALLELISM_DIM_1),

      .DATA_OUT_0_PRECISION_0      (OUT_DATA_PRECISION_0),
      .DATA_OUT_0_PRECISION_1      (OUT_DATA_PRECISION_1),
      .DATA_OUT_0_TENSOR_SIZE_DIM_0(IN_DATA_MAX_TENSOR_SIZE_DIM_1),
      .DATA_OUT_0_TENSOR_SIZE_DIM_1(IN_DATA_MAX_TENSOR_SIZE_DIM_1),
      .DATA_OUT_0_PARALLELISM_DIM_0(IN_DATA_PARALLELISM_DIM_0),
      .DATA_OUT_0_PARALLELISM_DIM_1(IN_DATA_PARALLELISM_DIM_1)

  ) fixed_softermax_i (
      .clk,
      .rst,

      .data_in_0      (sum_att),
      .data_in_0_valid(sum_att_valid),
      .data_in_0_ready(sum_att_ready),

      .data_out_0      (attention_scores),
      .data_out_0_valid(attention_scores_valid),
      .data_out_0_ready(attention_scores_ready)
  );

  //Output: Attention scores x Value

  matmul_programmable #(
      .A_MAX_DIM0(IN_DATA_MAX_TENSOR_SIZE_DIM_0),
      .A_MAX_DIM1(IN_DATA_MAX_TENSOR_SIZE_DIM_1),

      .B_MAX_DIM0(IN_DATA_MAX_TENSOR_SIZE_DIM_1),
      .B_MAX_DIM1(IN_DATA_MAX_TENSOR_SIZE_DIM_0),

      .A_COMPUTE_DIM0(IN_DATA_PARALLELISM_DIM_0),
      .A_COMPUTE_DIM1(IN_DATA_PARALLELISM_DIM_1),
      .B_COMPUTE_DIM0(IN_DATA_PARALLELISM_DIM_1),
      .B_COMPUTE_DIM1(IN_DATA_PARALLELISM_DIM_0),

      .A_WIDTH     (IN_DATA_PRECISION_0),
      .A_FRAC_WIDTH(IN_DATA_PRECISION_1),

      .B_WIDTH     (IN_DATA_PRECISION_0),
      .B_FRAC_WIDTH(IN_DATA_PRECISION_1),

      .OUT_WIDTH     (OUT_DATA_PRECISION_0),
      .OUT_FRAC_WIDTH(OUT_DATA_PRECISION_1)

  ) attention_scores_values_matmul_i (
      .clk,
      .rst,

      .a_depth_dim1(in_depth_dim_1),
      .b_depth_dim0(in_depth_dim_0),
      .b_depth_dim1(in_depth_dim_1),
      .b_depth_mult(in_depth_mult),

      .a_data (attention_scores),
      .a_valid(attention_scores_valid),
      .a_ready(attention_scores_ready),

      .b_data (value),
      .b_valid(value_valid),
      .b_ready(value_ready),

      .out_data (out_pre_cast),
      .out_valid(out_cast_valid),
      .out_ready(out_cast_ready)
  );

  // * Output cast

  fixed_rounding #(
      .IN_SIZE(OUT_DATA_PARALLELISM_DIM_0 * OUT_DATA_PARALLELISM_DIM_1),

      .IN_WIDTH     (OUT_DATA_PRECISION_0),
      .IN_FRAC_WIDTH(OUT_DATA_PRECISION_1),

      .OUT_WIDTH     (OUT_DATA_PRECISION_0),
      .OUT_FRAC_WIDTH(OUT_DATA_PRECISION_1)
  ) data_out_cast (
      .data_in (out_pre_cast),
      .data_out(out_casted)
  );

  unpacked_register_slice #(
      .DATA_WIDTH(OUT_DATA_PRECISION_0),
      .IN_SIZE   (OUT_DATA_PARALLELISM_DIM_0 * OUT_DATA_PARALLELISM_DIM_1)
  ) out_cast_register_slice_i (
      .clk(clk),
      .rst(rst),

      .data_in (out_casted),
      .data_in_valid(out_cast_valid),
      .data_in_ready(out_cast_ready),

      .data_out (out),
      .data_out_valid(out_valid),
      .data_out_ready(out_ready)
  );

endmodule
