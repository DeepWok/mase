
// =====================================
//     Mase Hardware
//     Model: top
//     23/02/2025 14:08:22
// =====================================
`timescale 1ns/1ps
module top #(
    parameter fork2_DATA_IN_0_PRECISION_0 = 8,
    parameter fork2_DATA_IN_0_PRECISION_1 = 8,
    parameter fork2_DATA_IN_0_TENSOR_SIZE_DIM_0 = 192,
    parameter fork2_DATA_IN_0_PARALLELISM_DIM_0 = 16,
    parameter fork2_DATA_IN_0_TENSOR_SIZE_DIM_1 = 196,
    parameter fork2_DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter fork2_DATA_OUT_0_PRECISION_0 = 8,
    parameter fork2_DATA_OUT_0_PRECISION_1 = 8,
    parameter fork2_DATA_OUT_0_TENSOR_SIZE_DIM_0 = 192,
    parameter fork2_DATA_OUT_0_PARALLELISM_DIM_0 = 16,
    parameter fork2_DATA_OUT_0_TENSOR_SIZE_DIM_1 = 196,
    parameter fork2_DATA_OUT_0_PARALLELISM_DIM_1 = 1,
    parameter fork2_DATA_OUT_1_PRECISION_0 = 8,
    parameter fork2_DATA_OUT_1_PRECISION_1 = 8,
    parameter fork2_DATA_OUT_1_TENSOR_SIZE_DIM_0 = 192,
    parameter fork2_DATA_OUT_1_PARALLELISM_DIM_0 = 16,
    parameter fork2_DATA_OUT_1_TENSOR_SIZE_DIM_1 = 196,
    parameter fork2_DATA_OUT_1_PARALLELISM_DIM_1 = 1,
    parameter linear1_DATA_IN_0_PRECISION_0 = 8,
    parameter linear1_DATA_IN_0_PRECISION_1 = 8,
    parameter linear1_DATA_IN_0_TENSOR_SIZE_DIM_0 = 192,
    parameter linear1_DATA_IN_0_PARALLELISM_DIM_0 = 16,
    parameter linear1_DATA_IN_0_TENSOR_SIZE_DIM_1 = 196,
    parameter linear1_DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter linear1_WEIGHT_PRECISION_0 = 6,
    parameter linear1_WEIGHT_PRECISION_1 = 8,
    parameter linear1_WEIGHT_TENSOR_SIZE_DIM_0 = 192,
    parameter linear1_WEIGHT_PARALLELISM_DIM_0 = 16,
    parameter linear1_WEIGHT_TENSOR_SIZE_DIM_1 = 768,
    parameter linear1_WEIGHT_PARALLELISM_DIM_1 = 16,
    parameter linear1_BIAS_PRECISION_0 = 6,
    parameter linear1_BIAS_PRECISION_1 = 8,
    parameter linear1_BIAS_TENSOR_SIZE_DIM_0 = 768,
    parameter linear1_BIAS_PARALLELISM_DIM_0 = 16,
    parameter linear1_BIAS_TENSOR_SIZE_DIM_1 = 1,
    parameter linear1_BIAS_PARALLELISM_DIM_1 = 1,
    parameter linear1_DATA_OUT_0_PRECISION_0 = 8,
    parameter linear1_DATA_OUT_0_PRECISION_1 = 8,
    parameter linear1_DATA_OUT_0_TENSOR_SIZE_DIM_0 = 768,
    parameter linear1_DATA_OUT_0_PARALLELISM_DIM_0 = 16,
    parameter linear1_DATA_OUT_0_TENSOR_SIZE_DIM_1 = 196,
    parameter linear1_DATA_OUT_0_PARALLELISM_DIM_1 = 1,
    parameter act_DATA_IN_0_PRECISION_0 = 8,
    parameter act_DATA_IN_0_PRECISION_1 = 8,
    parameter act_DATA_IN_0_TENSOR_SIZE_DIM_0 = 768,
    parameter act_DATA_IN_0_PARALLELISM_DIM_0 = 16,
    parameter act_DATA_IN_0_TENSOR_SIZE_DIM_1 = 196,
    parameter act_DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter act_DATA_OUT_0_PRECISION_0 = 8,
    parameter act_DATA_OUT_0_PRECISION_1 = 8,
    parameter act_DATA_OUT_0_TENSOR_SIZE_DIM_0 = 768,
    parameter act_DATA_OUT_0_PARALLELISM_DIM_0 = 16,
    parameter act_DATA_OUT_0_TENSOR_SIZE_DIM_1 = 196,
    parameter act_DATA_OUT_0_PARALLELISM_DIM_1 = 1,
    parameter linear2_DATA_IN_0_PRECISION_0 = 8,
    parameter linear2_DATA_IN_0_PRECISION_1 = 8,
    parameter linear2_DATA_IN_0_TENSOR_SIZE_DIM_0 = 768,
    parameter linear2_DATA_IN_0_PARALLELISM_DIM_0 = 16,
    parameter linear2_DATA_IN_0_TENSOR_SIZE_DIM_1 = 196,
    parameter linear2_DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter linear2_WEIGHT_PRECISION_0 = 6,
    parameter linear2_WEIGHT_PRECISION_1 = 8,
    parameter linear2_WEIGHT_TENSOR_SIZE_DIM_0 = 768,
    parameter linear2_WEIGHT_PARALLELISM_DIM_0 = 16,
    parameter linear2_WEIGHT_TENSOR_SIZE_DIM_1 = 192,
    parameter linear2_WEIGHT_PARALLELISM_DIM_1 = 16,
    parameter linear2_BIAS_PRECISION_0 = 6,
    parameter linear2_BIAS_PRECISION_1 = 8,
    parameter linear2_BIAS_TENSOR_SIZE_DIM_0 = 192,
    parameter linear2_BIAS_PARALLELISM_DIM_0 = 16,
    parameter linear2_BIAS_TENSOR_SIZE_DIM_1 = 1,
    parameter linear2_BIAS_PARALLELISM_DIM_1 = 1,
    parameter linear2_DATA_OUT_0_PRECISION_0 = 8,
    parameter linear2_DATA_OUT_0_PRECISION_1 = 8,
    parameter linear2_DATA_OUT_0_TENSOR_SIZE_DIM_0 = 192,
    parameter linear2_DATA_OUT_0_PARALLELISM_DIM_0 = 16,
    parameter linear2_DATA_OUT_0_TENSOR_SIZE_DIM_1 = 196,
    parameter linear2_DATA_OUT_0_PARALLELISM_DIM_1 = 1,
    parameter norm1_DATA_IN_0_PRECISION_0 = 8,
    parameter norm1_DATA_IN_0_PRECISION_1 = 8,
    parameter norm1_DATA_IN_0_TENSOR_SIZE_DIM_0 = 192,
    parameter norm1_DATA_IN_0_PARALLELISM_DIM_0 = 16,
    parameter norm1_DATA_IN_0_TENSOR_SIZE_DIM_1 = 196,
    parameter norm1_DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter norm1_WEIGHT_PRECISION_0 = 6,
    parameter norm1_WEIGHT_PRECISION_1 = 8,
    parameter norm1_WEIGHT_TENSOR_SIZE_DIM_0 = 192,
    parameter norm1_WEIGHT_PARALLELISM_DIM_0 = 16,
    parameter norm1_WEIGHT_TENSOR_SIZE_DIM_1 = 1,
    parameter norm1_WEIGHT_PARALLELISM_DIM_1 = 1,
    parameter norm1_BIAS_PRECISION_0 = 6,
    parameter norm1_BIAS_PRECISION_1 = 8,
    parameter norm1_BIAS_TENSOR_SIZE_DIM_0 = 192,
    parameter norm1_BIAS_PARALLELISM_DIM_0 = 16,
    parameter norm1_BIAS_TENSOR_SIZE_DIM_1 = 1,
    parameter norm1_BIAS_PARALLELISM_DIM_1 = 1,
    parameter norm1_ELEMENTWISE_AFFINE = 1,
    parameter norm1_HAS_BIAS = 1,
    parameter norm1_DATA_OUT_0_PRECISION_0 = 8,
    parameter norm1_DATA_OUT_0_PRECISION_1 = 8,
    parameter norm1_DATA_OUT_0_TENSOR_SIZE_DIM_0 = 192,
    parameter norm1_DATA_OUT_0_PARALLELISM_DIM_0 = 16,
    parameter norm1_DATA_OUT_0_TENSOR_SIZE_DIM_1 = 196,
    parameter norm1_DATA_OUT_0_PARALLELISM_DIM_1 = 1,
    parameter add_DATA_IN_0_PRECISION_0 = 8,
    parameter add_DATA_IN_0_PRECISION_1 = 8,
    parameter add_DATA_IN_0_TENSOR_SIZE_DIM_0 = 192,
    parameter add_DATA_IN_0_PARALLELISM_DIM_0 = 16,
    parameter add_DATA_IN_0_TENSOR_SIZE_DIM_1 = 196,
    parameter add_DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter add_DATA_IN_1_PRECISION_0 = 8,
    parameter add_DATA_IN_1_PRECISION_1 = 8,
    parameter add_DATA_IN_1_TENSOR_SIZE_DIM_0 = 192,
    parameter add_DATA_IN_1_PARALLELISM_DIM_0 = 16,
    parameter add_DATA_IN_1_TENSOR_SIZE_DIM_1 = 196,
    parameter add_DATA_IN_1_PARALLELISM_DIM_1 = 1,
    parameter add_DATA_OUT_0_PRECISION_0 = 8,
    parameter add_DATA_OUT_0_PRECISION_1 = 8,
    parameter add_DATA_OUT_0_TENSOR_SIZE_DIM_0 = 192,
    parameter add_DATA_OUT_0_PARALLELISM_DIM_0 = 16,
    parameter add_DATA_OUT_0_TENSOR_SIZE_DIM_1 = 196,
    parameter add_DATA_OUT_0_PARALLELISM_DIM_1 = 1,
    parameter fork2_1_DATA_IN_0_PRECISION_0 = 8,
    parameter fork2_1_DATA_IN_0_PRECISION_1 = 8,
    parameter fork2_1_DATA_IN_0_TENSOR_SIZE_DIM_0 = 192,
    parameter fork2_1_DATA_IN_0_PARALLELISM_DIM_0 = 16,
    parameter fork2_1_DATA_IN_0_TENSOR_SIZE_DIM_1 = 196,
    parameter fork2_1_DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter fork2_1_DATA_OUT_0_PRECISION_0 = 8,
    parameter fork2_1_DATA_OUT_0_PRECISION_1 = 8,
    parameter fork2_1_DATA_OUT_0_TENSOR_SIZE_DIM_0 = 192,
    parameter fork2_1_DATA_OUT_0_PARALLELISM_DIM_0 = 16,
    parameter fork2_1_DATA_OUT_0_TENSOR_SIZE_DIM_1 = 196,
    parameter fork2_1_DATA_OUT_0_PARALLELISM_DIM_1 = 1,
    parameter fork2_1_DATA_OUT_1_PRECISION_0 = 8,
    parameter fork2_1_DATA_OUT_1_PRECISION_1 = 8,
    parameter fork2_1_DATA_OUT_1_TENSOR_SIZE_DIM_0 = 192,
    parameter fork2_1_DATA_OUT_1_PARALLELISM_DIM_0 = 16,
    parameter fork2_1_DATA_OUT_1_TENSOR_SIZE_DIM_1 = 196,
    parameter fork2_1_DATA_OUT_1_PARALLELISM_DIM_1 = 1,
    parameter attention_DATA_IN_0_PRECISION_0 = 8,
    parameter attention_DATA_IN_0_PRECISION_1 = 8,
    parameter attention_DATA_IN_0_TENSOR_SIZE_DIM_0 = 192,
    parameter attention_DATA_IN_0_PARALLELISM_DIM_0 = 16,
    parameter attention_DATA_IN_0_TENSOR_SIZE_DIM_1 = 196,
    parameter attention_DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter attention_QUERY_WEIGHT_PRECISION_0 = 6,
    parameter attention_QUERY_WEIGHT_PRECISION_1 = 8,
    parameter attention_QUERY_WEIGHT_TENSOR_SIZE_DIM_0 = 192,
    parameter attention_QUERY_WEIGHT_PARALLELISM_DIM_0 = 16,
    parameter attention_QUERY_WEIGHT_TENSOR_SIZE_DIM_1 = 192,
    parameter attention_QUERY_WEIGHT_PARALLELISM_DIM_1 = 16,
    parameter attention_QUERY_BIAS_PRECISION_0 = 6,
    parameter attention_QUERY_BIAS_PRECISION_1 = 8,
    parameter attention_QUERY_BIAS_TENSOR_SIZE_DIM_0 = 192,
    parameter attention_QUERY_BIAS_PARALLELISM_DIM_0 = 16,
    parameter attention_QUERY_BIAS_TENSOR_SIZE_DIM_1 = 1,
    parameter attention_QUERY_BIAS_PARALLELISM_DIM_1 = 1,
    parameter attention_KEY_WEIGHT_PRECISION_0 = 6,
    parameter attention_KEY_WEIGHT_PRECISION_1 = 8,
    parameter attention_KEY_WEIGHT_TENSOR_SIZE_DIM_0 = 192,
    parameter attention_KEY_WEIGHT_PARALLELISM_DIM_0 = 16,
    parameter attention_KEY_WEIGHT_TENSOR_SIZE_DIM_1 = 192,
    parameter attention_KEY_WEIGHT_PARALLELISM_DIM_1 = 16,
    parameter attention_KEY_BIAS_PRECISION_0 = 6,
    parameter attention_KEY_BIAS_PRECISION_1 = 8,
    parameter attention_KEY_BIAS_TENSOR_SIZE_DIM_0 = 192,
    parameter attention_KEY_BIAS_PARALLELISM_DIM_0 = 16,
    parameter attention_KEY_BIAS_TENSOR_SIZE_DIM_1 = 1,
    parameter attention_KEY_BIAS_PARALLELISM_DIM_1 = 1,
    parameter attention_VALUE_WEIGHT_PRECISION_0 = 6,
    parameter attention_VALUE_WEIGHT_PRECISION_1 = 8,
    parameter attention_VALUE_WEIGHT_TENSOR_SIZE_DIM_0 = 192,
    parameter attention_VALUE_WEIGHT_PARALLELISM_DIM_0 = 16,
    parameter attention_VALUE_WEIGHT_TENSOR_SIZE_DIM_1 = 192,
    parameter attention_VALUE_WEIGHT_PARALLELISM_DIM_1 = 16,
    parameter attention_VALUE_BIAS_PRECISION_0 = 6,
    parameter attention_VALUE_BIAS_PRECISION_1 = 8,
    parameter attention_VALUE_BIAS_TENSOR_SIZE_DIM_0 = 192,
    parameter attention_VALUE_BIAS_PARALLELISM_DIM_0 = 16,
    parameter attention_VALUE_BIAS_TENSOR_SIZE_DIM_1 = 1,
    parameter attention_VALUE_BIAS_PARALLELISM_DIM_1 = 1,
    parameter attention_PROJ_WEIGHT_PRECISION_0 = 6,
    parameter attention_PROJ_WEIGHT_PRECISION_1 = 8,
    parameter attention_PROJ_WEIGHT_TENSOR_SIZE_DIM_0 = 192,
    parameter attention_PROJ_WEIGHT_PARALLELISM_DIM_0 = 16,
    parameter attention_PROJ_WEIGHT_TENSOR_SIZE_DIM_1 = 192,
    parameter attention_PROJ_WEIGHT_PARALLELISM_DIM_1 = 16,
    parameter attention_PROJ_BIAS_PRECISION_0 = 6,
    parameter attention_PROJ_BIAS_PRECISION_1 = 8,
    parameter attention_PROJ_BIAS_TENSOR_SIZE_DIM_0 = 192,
    parameter attention_PROJ_BIAS_PARALLELISM_DIM_0 = 16,
    parameter attention_PROJ_BIAS_TENSOR_SIZE_DIM_1 = 1,
    parameter attention_PROJ_BIAS_PARALLELISM_DIM_1 = 1,
    parameter attention_DATA_OUT_0_PRECISION_0 = 8,
    parameter attention_DATA_OUT_0_PRECISION_1 = 8,
    parameter attention_DATA_OUT_0_TENSOR_SIZE_DIM_0 = 192,
    parameter attention_DATA_OUT_0_PARALLELISM_DIM_0 = 16,
    parameter attention_DATA_OUT_0_TENSOR_SIZE_DIM_1 = 196,
    parameter attention_DATA_OUT_0_PARALLELISM_DIM_1 = 1,
    parameter norm2_DATA_IN_0_PRECISION_0 = 8,
    parameter norm2_DATA_IN_0_PRECISION_1 = 8,
    parameter norm2_DATA_IN_0_TENSOR_SIZE_DIM_0 = 192,
    parameter norm2_DATA_IN_0_PARALLELISM_DIM_0 = 16,
    parameter norm2_DATA_IN_0_TENSOR_SIZE_DIM_1 = 196,
    parameter norm2_DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter norm2_WEIGHT_PRECISION_0 = 6,
    parameter norm2_WEIGHT_PRECISION_1 = 8,
    parameter norm2_WEIGHT_TENSOR_SIZE_DIM_0 = 192,
    parameter norm2_WEIGHT_PARALLELISM_DIM_0 = 16,
    parameter norm2_WEIGHT_TENSOR_SIZE_DIM_1 = 1,
    parameter norm2_WEIGHT_PARALLELISM_DIM_1 = 1,
    parameter norm2_BIAS_PRECISION_0 = 6,
    parameter norm2_BIAS_PRECISION_1 = 8,
    parameter norm2_BIAS_TENSOR_SIZE_DIM_0 = 192,
    parameter norm2_BIAS_PARALLELISM_DIM_0 = 16,
    parameter norm2_BIAS_TENSOR_SIZE_DIM_1 = 1,
    parameter norm2_BIAS_PARALLELISM_DIM_1 = 1,
    parameter norm2_ELEMENTWISE_AFFINE = 1,
    parameter norm2_HAS_BIAS = 1,
    parameter norm2_DATA_OUT_0_PRECISION_0 = 8,
    parameter norm2_DATA_OUT_0_PRECISION_1 = 8,
    parameter norm2_DATA_OUT_0_TENSOR_SIZE_DIM_0 = 192,
    parameter norm2_DATA_OUT_0_PARALLELISM_DIM_0 = 16,
    parameter norm2_DATA_OUT_0_TENSOR_SIZE_DIM_1 = 196,
    parameter norm2_DATA_OUT_0_PARALLELISM_DIM_1 = 1,
    parameter add_1_DATA_IN_0_PRECISION_0 = 8,
    parameter add_1_DATA_IN_0_PRECISION_1 = 8,
    parameter add_1_DATA_IN_0_TENSOR_SIZE_DIM_0 = 192,
    parameter add_1_DATA_IN_0_PARALLELISM_DIM_0 = 16,
    parameter add_1_DATA_IN_0_TENSOR_SIZE_DIM_1 = 196,
    parameter add_1_DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter add_1_DATA_IN_1_PRECISION_0 = 8,
    parameter add_1_DATA_IN_1_PRECISION_1 = 8,
    parameter add_1_DATA_IN_1_TENSOR_SIZE_DIM_0 = 192,
    parameter add_1_DATA_IN_1_PARALLELISM_DIM_0 = 16,
    parameter add_1_DATA_IN_1_TENSOR_SIZE_DIM_1 = 196,
    parameter add_1_DATA_IN_1_PARALLELISM_DIM_1 = 1,
    parameter add_1_DATA_OUT_0_PRECISION_0 = 8,
    parameter add_1_DATA_OUT_0_PRECISION_1 = 8,
    parameter add_1_DATA_OUT_0_TENSOR_SIZE_DIM_0 = 192,
    parameter add_1_DATA_OUT_0_PARALLELISM_DIM_0 = 16,
    parameter add_1_DATA_OUT_0_TENSOR_SIZE_DIM_1 = 196,
    parameter add_1_DATA_OUT_0_PARALLELISM_DIM_1 = 1,
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 8,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 192,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 16,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 196,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 8,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 192,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 16,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 196,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1
) (
    input clk,
    input rst,

    input  [DATA_IN_0_PRECISION_0-1:0] mdata_in_0 [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input  [DATA_IN_0_PRECISION_1-1:0] edata_in_0,
    input  data_in_0_valid,
    output data_in_0_ready,
    output  [DATA_OUT_0_PRECISION_0-1:0] mdata_out_0 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output  [DATA_OUT_0_PRECISION_1-1:0] edata_out_0,
    output  data_out_0_valid,
    input data_out_0_ready
);

// --------------------------
//   fork2 signals
// --------------------------
logic [fork2_DATA_IN_0_PRECISION_0-1:0]  fork2_mdata_in_0        [fork2_DATA_IN_0_PARALLELISM_DIM_0*fork2_DATA_IN_0_PARALLELISM_DIM_1-1:0];
logic [fork2_DATA_IN_0_PRECISION_1-1:0]  fork2_edata_in_0;
logic                             fork2_data_in_0_valid;
logic                             fork2_data_in_0_ready;
logic [fork2_DATA_OUT_0_PRECISION_0-1:0]  fork2_mdata_out_0        [fork2_DATA_OUT_0_PARALLELISM_DIM_0*fork2_DATA_OUT_0_PARALLELISM_DIM_1-1:0];
logic [fork2_DATA_OUT_0_PRECISION_1-1:0]  fork2_edata_out_0;
logic                             fork2_data_out_0_valid;
logic                             fork2_data_out_0_ready;
logic [fork2_DATA_OUT_1_PRECISION_0-1:0]  fork2_mdata_out_1        [fork2_DATA_OUT_1_PARALLELISM_DIM_0*fork2_DATA_OUT_1_PARALLELISM_DIM_1-1:0];
logic [fork2_DATA_OUT_1_PRECISION_1-1:0]  fork2_edata_out_1;
logic                             fork2_data_out_1_valid;
logic                             fork2_data_out_1_ready;
// --------------------------
//   linear1 signals
// --------------------------
logic [linear1_DATA_IN_0_PRECISION_0-1:0]  linear1_mdata_in_0        [linear1_DATA_IN_0_PARALLELISM_DIM_0*linear1_DATA_IN_0_PARALLELISM_DIM_1-1:0];
logic [linear1_DATA_IN_0_PRECISION_1-1:0]  linear1_edata_in_0;
logic                             linear1_data_in_0_valid;
logic                             linear1_data_in_0_ready;
logic [linear1_WEIGHT_PRECISION_0-1:0]  linear1_mweight        [linear1_WEIGHT_PARALLELISM_DIM_0*linear1_WEIGHT_PARALLELISM_DIM_1-1:0];
logic [linear1_WEIGHT_PRECISION_1-1:0]  linear1_eweight;
logic                             linear1_weight_valid;
logic                             linear1_weight_ready;
logic [linear1_BIAS_PRECISION_0-1:0]  linear1_mbias        [linear1_BIAS_PARALLELISM_DIM_0*linear1_BIAS_PARALLELISM_DIM_1-1:0];
logic [linear1_BIAS_PRECISION_1-1:0]  linear1_ebias;
logic                             linear1_bias_valid;
logic                             linear1_bias_ready;
logic [linear1_DATA_OUT_0_PRECISION_0-1:0]  linear1_mdata_out_0        [linear1_DATA_OUT_0_PARALLELISM_DIM_0*linear1_DATA_OUT_0_PARALLELISM_DIM_1-1:0];
logic [linear1_DATA_OUT_0_PRECISION_1-1:0]  linear1_edata_out_0;
logic                             linear1_data_out_0_valid;
logic                             linear1_data_out_0_ready;
// --------------------------
//   act signals
// --------------------------
logic [act_DATA_IN_0_PRECISION_0-1:0]  act_mdata_in_0        [act_DATA_IN_0_PARALLELISM_DIM_0*act_DATA_IN_0_PARALLELISM_DIM_1-1:0];
logic [act_DATA_IN_0_PRECISION_1-1:0]  act_edata_in_0;
logic                             act_data_in_0_valid;
logic                             act_data_in_0_ready;
logic [act_DATA_OUT_0_PRECISION_0-1:0]  act_mdata_out_0        [act_DATA_OUT_0_PARALLELISM_DIM_0*act_DATA_OUT_0_PARALLELISM_DIM_1-1:0];
logic [act_DATA_OUT_0_PRECISION_1-1:0]  act_edata_out_0;
logic                             act_data_out_0_valid;
logic                             act_data_out_0_ready;
// --------------------------
//   linear2 signals
// --------------------------
logic [linear2_DATA_IN_0_PRECISION_0-1:0]  linear2_mdata_in_0        [linear2_DATA_IN_0_PARALLELISM_DIM_0*linear2_DATA_IN_0_PARALLELISM_DIM_1-1:0];
logic [linear2_DATA_IN_0_PRECISION_1-1:0]  linear2_edata_in_0;
logic                             linear2_data_in_0_valid;
logic                             linear2_data_in_0_ready;
logic [linear2_WEIGHT_PRECISION_0-1:0]  linear2_mweight        [linear2_WEIGHT_PARALLELISM_DIM_0*linear2_WEIGHT_PARALLELISM_DIM_1-1:0];
logic [linear2_WEIGHT_PRECISION_1-1:0]  linear2_eweight;
logic                             linear2_weight_valid;
logic                             linear2_weight_ready;
logic [linear2_BIAS_PRECISION_0-1:0]  linear2_mbias        [linear2_BIAS_PARALLELISM_DIM_0*linear2_BIAS_PARALLELISM_DIM_1-1:0];
logic [linear2_BIAS_PRECISION_1-1:0]  linear2_ebias;
logic                             linear2_bias_valid;
logic                             linear2_bias_ready;
logic [linear2_DATA_OUT_0_PRECISION_0-1:0]  linear2_mdata_out_0        [linear2_DATA_OUT_0_PARALLELISM_DIM_0*linear2_DATA_OUT_0_PARALLELISM_DIM_1-1:0];
logic [linear2_DATA_OUT_0_PRECISION_1-1:0]  linear2_edata_out_0;
logic                             linear2_data_out_0_valid;
logic                             linear2_data_out_0_ready;
// --------------------------
//   norm1 signals
// --------------------------
logic [norm1_DATA_IN_0_PRECISION_0-1:0]  norm1_mdata_in_0        [norm1_DATA_IN_0_PARALLELISM_DIM_0*norm1_DATA_IN_0_PARALLELISM_DIM_1-1:0];
logic [norm1_DATA_IN_0_PRECISION_1-1:0]  norm1_edata_in_0;
logic                             norm1_data_in_0_valid;
logic                             norm1_data_in_0_ready;
logic [norm1_WEIGHT_PRECISION_0-1:0]  norm1_mweight        [norm1_WEIGHT_PARALLELISM_DIM_0*norm1_WEIGHT_PARALLELISM_DIM_1-1:0];
logic [norm1_WEIGHT_PRECISION_1-1:0]  norm1_eweight;
logic                             norm1_weight_valid;
logic                             norm1_weight_ready;
logic [norm1_BIAS_PRECISION_0-1:0]  norm1_mbias        [norm1_BIAS_PARALLELISM_DIM_0*norm1_BIAS_PARALLELISM_DIM_1-1:0];
logic [norm1_BIAS_PRECISION_1-1:0]  norm1_ebias;
logic                             norm1_bias_valid;
logic                             norm1_bias_ready;
logic [norm1_DATA_OUT_0_PRECISION_0-1:0]  norm1_mdata_out_0        [norm1_DATA_OUT_0_PARALLELISM_DIM_0*norm1_DATA_OUT_0_PARALLELISM_DIM_1-1:0];
logic [norm1_DATA_OUT_0_PRECISION_1-1:0]  norm1_edata_out_0;
logic                             norm1_data_out_0_valid;
logic                             norm1_data_out_0_ready;
// --------------------------
//   add signals
// --------------------------
logic [add_DATA_IN_0_PRECISION_0-1:0]  add_mdata_in_0        [add_DATA_IN_0_PARALLELISM_DIM_0*add_DATA_IN_0_PARALLELISM_DIM_1-1:0];
logic [add_DATA_IN_0_PRECISION_1-1:0]  add_edata_in_0;
logic                             add_data_in_0_valid;
logic                             add_data_in_0_ready;
logic [add_DATA_IN_1_PRECISION_0-1:0]  add_mdata_in_1        [add_DATA_IN_1_PARALLELISM_DIM_0*add_DATA_IN_1_PARALLELISM_DIM_1-1:0];
logic [add_DATA_IN_1_PRECISION_1-1:0]  add_edata_in_1;
logic                             add_data_in_1_valid;
logic                             add_data_in_1_ready;
logic [add_DATA_OUT_0_PRECISION_0-1:0]  add_mdata_out_0        [add_DATA_OUT_0_PARALLELISM_DIM_0*add_DATA_OUT_0_PARALLELISM_DIM_1-1:0];
logic [add_DATA_OUT_0_PRECISION_1-1:0]  add_edata_out_0;
logic                             add_data_out_0_valid;
logic                             add_data_out_0_ready;
// --------------------------
//   fork2_1 signals
// --------------------------
logic [fork2_1_DATA_IN_0_PRECISION_0-1:0]  fork2_1_mdata_in_0        [fork2_1_DATA_IN_0_PARALLELISM_DIM_0*fork2_1_DATA_IN_0_PARALLELISM_DIM_1-1:0];
logic [fork2_1_DATA_IN_0_PRECISION_1-1:0]  fork2_1_edata_in_0;
logic                             fork2_1_data_in_0_valid;
logic                             fork2_1_data_in_0_ready;
logic [fork2_1_DATA_OUT_0_PRECISION_0-1:0]  fork2_1_mdata_out_0        [fork2_1_DATA_OUT_0_PARALLELISM_DIM_0*fork2_1_DATA_OUT_0_PARALLELISM_DIM_1-1:0];
logic [fork2_1_DATA_OUT_0_PRECISION_1-1:0]  fork2_1_edata_out_0;
logic                             fork2_1_data_out_0_valid;
logic                             fork2_1_data_out_0_ready;
logic [fork2_1_DATA_OUT_1_PRECISION_0-1:0]  fork2_1_mdata_out_1        [fork2_1_DATA_OUT_1_PARALLELISM_DIM_0*fork2_1_DATA_OUT_1_PARALLELISM_DIM_1-1:0];
logic [fork2_1_DATA_OUT_1_PRECISION_1-1:0]  fork2_1_edata_out_1;
logic                             fork2_1_data_out_1_valid;
logic                             fork2_1_data_out_1_ready;
// --------------------------
//   attention signals
// --------------------------
logic [attention_DATA_IN_0_PRECISION_0-1:0]  attention_mdata_in_0        [attention_DATA_IN_0_PARALLELISM_DIM_0*attention_DATA_IN_0_PARALLELISM_DIM_1-1:0];
logic [attention_DATA_IN_0_PRECISION_1-1:0]  attention_edata_in_0;
logic                             attention_data_in_0_valid;
logic                             attention_data_in_0_ready;
logic [attention_QUERY_WEIGHT_PRECISION_0-1:0]  attention_mquery_weight        [attention_QUERY_WEIGHT_PARALLELISM_DIM_0*attention_QUERY_WEIGHT_PARALLELISM_DIM_1-1:0];
logic [attention_QUERY_WEIGHT_PRECISION_1-1:0]  attention_equery_weight;
logic                             attention_query_weight_valid;
logic                             attention_query_weight_ready;
logic [attention_QUERY_BIAS_PRECISION_0-1:0]  attention_mquery_bias        [attention_QUERY_BIAS_PARALLELISM_DIM_0*attention_QUERY_BIAS_PARALLELISM_DIM_1-1:0];
logic [attention_QUERY_BIAS_PRECISION_1-1:0]  attention_equery_bias;
logic                             attention_query_bias_valid;
logic                             attention_query_bias_ready;
logic [attention_KEY_WEIGHT_PRECISION_0-1:0]  attention_mkey_weight        [attention_KEY_WEIGHT_PARALLELISM_DIM_0*attention_KEY_WEIGHT_PARALLELISM_DIM_1-1:0];
logic [attention_KEY_WEIGHT_PRECISION_1-1:0]  attention_ekey_weight;
logic                             attention_key_weight_valid;
logic                             attention_key_weight_ready;
logic [attention_KEY_BIAS_PRECISION_0-1:0]  attention_mkey_bias        [attention_KEY_BIAS_PARALLELISM_DIM_0*attention_KEY_BIAS_PARALLELISM_DIM_1-1:0];
logic [attention_KEY_BIAS_PRECISION_1-1:0]  attention_ekey_bias;
logic                             attention_key_bias_valid;
logic                             attention_key_bias_ready;
logic [attention_VALUE_WEIGHT_PRECISION_0-1:0]  attention_mvalue_weight        [attention_VALUE_WEIGHT_PARALLELISM_DIM_0*attention_VALUE_WEIGHT_PARALLELISM_DIM_1-1:0];
logic [attention_VALUE_WEIGHT_PRECISION_1-1:0]  attention_evalue_weight;
logic                             attention_value_weight_valid;
logic                             attention_value_weight_ready;
logic [attention_VALUE_BIAS_PRECISION_0-1:0]  attention_mvalue_bias        [attention_VALUE_BIAS_PARALLELISM_DIM_0*attention_VALUE_BIAS_PARALLELISM_DIM_1-1:0];
logic [attention_VALUE_BIAS_PRECISION_1-1:0]  attention_evalue_bias;
logic                             attention_value_bias_valid;
logic                             attention_value_bias_ready;
logic [attention_PROJ_WEIGHT_PRECISION_0-1:0]  attention_mproj_weight        [attention_PROJ_WEIGHT_PARALLELISM_DIM_0*attention_PROJ_WEIGHT_PARALLELISM_DIM_1-1:0];
logic [attention_PROJ_WEIGHT_PRECISION_1-1:0]  attention_eproj_weight;
logic                             attention_proj_weight_valid;
logic                             attention_proj_weight_ready;
logic [attention_PROJ_BIAS_PRECISION_0-1:0]  attention_mproj_bias        [attention_PROJ_BIAS_PARALLELISM_DIM_0*attention_PROJ_BIAS_PARALLELISM_DIM_1-1:0];
logic [attention_PROJ_BIAS_PRECISION_1-1:0]  attention_eproj_bias;
logic                             attention_proj_bias_valid;
logic                             attention_proj_bias_ready;
logic [attention_DATA_OUT_0_PRECISION_0-1:0]  attention_mdata_out_0        [attention_DATA_OUT_0_PARALLELISM_DIM_0*attention_DATA_OUT_0_PARALLELISM_DIM_1-1:0];
logic [attention_DATA_OUT_0_PRECISION_1-1:0]  attention_edata_out_0;
logic                             attention_data_out_0_valid;
logic                             attention_data_out_0_ready;
// --------------------------
//   norm2 signals
// --------------------------
logic [norm2_DATA_IN_0_PRECISION_0-1:0]  norm2_mdata_in_0        [norm2_DATA_IN_0_PARALLELISM_DIM_0*norm2_DATA_IN_0_PARALLELISM_DIM_1-1:0];
logic [norm2_DATA_IN_0_PRECISION_1-1:0]  norm2_edata_in_0;
logic                             norm2_data_in_0_valid;
logic                             norm2_data_in_0_ready;
logic [norm2_WEIGHT_PRECISION_0-1:0]  norm2_mweight        [norm2_WEIGHT_PARALLELISM_DIM_0*norm2_WEIGHT_PARALLELISM_DIM_1-1:0];
logic [norm2_WEIGHT_PRECISION_1-1:0]  norm2_eweight;
logic                             norm2_weight_valid;
logic                             norm2_weight_ready;
logic [norm2_BIAS_PRECISION_0-1:0]  norm2_mbias        [norm2_BIAS_PARALLELISM_DIM_0*norm2_BIAS_PARALLELISM_DIM_1-1:0];
logic [norm2_BIAS_PRECISION_1-1:0]  norm2_ebias;
logic                             norm2_bias_valid;
logic                             norm2_bias_ready;
logic [norm2_DATA_OUT_0_PRECISION_0-1:0]  norm2_mdata_out_0        [norm2_DATA_OUT_0_PARALLELISM_DIM_0*norm2_DATA_OUT_0_PARALLELISM_DIM_1-1:0];
logic [norm2_DATA_OUT_0_PRECISION_1-1:0]  norm2_edata_out_0;
logic                             norm2_data_out_0_valid;
logic                             norm2_data_out_0_ready;
// --------------------------
//   add_1 signals
// --------------------------
logic [add_1_DATA_IN_0_PRECISION_0-1:0]  add_1_mdata_in_0        [add_1_DATA_IN_0_PARALLELISM_DIM_0*add_1_DATA_IN_0_PARALLELISM_DIM_1-1:0];
logic [add_1_DATA_IN_0_PRECISION_1-1:0]  add_1_edata_in_0;
logic                             add_1_data_in_0_valid;
logic                             add_1_data_in_0_ready;
logic [add_1_DATA_IN_1_PRECISION_0-1:0]  add_1_mdata_in_1        [add_1_DATA_IN_1_PARALLELISM_DIM_0*add_1_DATA_IN_1_PARALLELISM_DIM_1-1:0];
logic [add_1_DATA_IN_1_PRECISION_1-1:0]  add_1_edata_in_1;
logic                             add_1_data_in_1_valid;
logic                             add_1_data_in_1_ready;
logic [add_1_DATA_OUT_0_PRECISION_0-1:0]  add_1_mdata_out_0        [add_1_DATA_OUT_0_PARALLELISM_DIM_0*add_1_DATA_OUT_0_PARALLELISM_DIM_1-1:0];
logic [add_1_DATA_OUT_0_PRECISION_1-1:0]  add_1_edata_out_0;
logic                             add_1_data_out_0_valid;
logic                             add_1_data_out_0_ready;

// --------------------------
//   Component instantiation
// --------------------------

// fork2
mxint_fork2 #(
    .DATA_IN_0_PRECISION_0(fork2_DATA_IN_0_PRECISION_0), // = 8
    .DATA_IN_0_PRECISION_1(fork2_DATA_IN_0_PRECISION_1), // = 8
    .DATA_IN_0_TENSOR_SIZE_DIM_0(fork2_DATA_IN_0_TENSOR_SIZE_DIM_0), // = 192
    .DATA_IN_0_PARALLELISM_DIM_0(fork2_DATA_IN_0_PARALLELISM_DIM_0), // = 16
    .DATA_IN_0_TENSOR_SIZE_DIM_1(fork2_DATA_IN_0_TENSOR_SIZE_DIM_1), // = 196
    .DATA_IN_0_PARALLELISM_DIM_1(fork2_DATA_IN_0_PARALLELISM_DIM_1), // = 1
    .DATA_OUT_0_PRECISION_0(fork2_DATA_OUT_0_PRECISION_0), // = 8
    .DATA_OUT_0_PRECISION_1(fork2_DATA_OUT_0_PRECISION_1), // = 8
    .DATA_OUT_0_TENSOR_SIZE_DIM_0(fork2_DATA_OUT_0_TENSOR_SIZE_DIM_0), // = 192
    .DATA_OUT_0_PARALLELISM_DIM_0(fork2_DATA_OUT_0_PARALLELISM_DIM_0), // = 16
    .DATA_OUT_0_TENSOR_SIZE_DIM_1(fork2_DATA_OUT_0_TENSOR_SIZE_DIM_1), // = 196
    .DATA_OUT_0_PARALLELISM_DIM_1(fork2_DATA_OUT_0_PARALLELISM_DIM_1), // = 1
    .DATA_OUT_1_PRECISION_0(fork2_DATA_OUT_1_PRECISION_0), // = 8
    .DATA_OUT_1_PRECISION_1(fork2_DATA_OUT_1_PRECISION_1), // = 8
    .DATA_OUT_1_TENSOR_SIZE_DIM_0(fork2_DATA_OUT_1_TENSOR_SIZE_DIM_0), // = 192
    .DATA_OUT_1_PARALLELISM_DIM_0(fork2_DATA_OUT_1_PARALLELISM_DIM_0), // = 16
    .DATA_OUT_1_TENSOR_SIZE_DIM_1(fork2_DATA_OUT_1_TENSOR_SIZE_DIM_1), // = 196
    .DATA_OUT_1_PARALLELISM_DIM_1(fork2_DATA_OUT_1_PARALLELISM_DIM_1)
) fork2_inst (
    .clk(clk),
    .rst(rst),

    .mdata_in_0(fork2_mdata_in_0),
    .edata_in_0(fork2_edata_in_0),
    .data_in_0_valid(fork2_data_in_0_valid),
    .data_in_0_ready(fork2_data_in_0_ready),
        
    .mdata_out_0(fork2_mdata_out_0),
    .edata_out_0(fork2_edata_out_0),
    .data_out_0_valid(fork2_data_out_0_valid),
    .data_out_0_ready(fork2_data_out_0_ready),
        
    .mdata_out_1(fork2_mdata_out_1),
    .edata_out_1(fork2_edata_out_1),
    .data_out_1_valid(fork2_data_out_1_valid),
    .data_out_1_ready(fork2_data_out_1_ready)
);

// linear1
mxint_linear #(
    .DATA_IN_0_PRECISION_0(linear1_DATA_IN_0_PRECISION_0), // = 8
    .DATA_IN_0_PRECISION_1(linear1_DATA_IN_0_PRECISION_1), // = 8
    .DATA_IN_0_TENSOR_SIZE_DIM_0(linear1_DATA_IN_0_TENSOR_SIZE_DIM_0), // = 192
    .DATA_IN_0_PARALLELISM_DIM_0(linear1_DATA_IN_0_PARALLELISM_DIM_0), // = 16
    .DATA_IN_0_TENSOR_SIZE_DIM_1(linear1_DATA_IN_0_TENSOR_SIZE_DIM_1), // = 196
    .DATA_IN_0_PARALLELISM_DIM_1(linear1_DATA_IN_0_PARALLELISM_DIM_1), // = 1
    .WEIGHT_PRECISION_0(linear1_WEIGHT_PRECISION_0), // = 6
    .WEIGHT_PRECISION_1(linear1_WEIGHT_PRECISION_1), // = 8
    .WEIGHT_TENSOR_SIZE_DIM_0(linear1_WEIGHT_TENSOR_SIZE_DIM_0), // = 192
    .WEIGHT_PARALLELISM_DIM_0(linear1_WEIGHT_PARALLELISM_DIM_0), // = 16
    .WEIGHT_TENSOR_SIZE_DIM_1(linear1_WEIGHT_TENSOR_SIZE_DIM_1), // = 768
    .WEIGHT_PARALLELISM_DIM_1(linear1_WEIGHT_PARALLELISM_DIM_1), // = 16
    .BIAS_PRECISION_0(linear1_BIAS_PRECISION_0), // = 6
    .BIAS_PRECISION_1(linear1_BIAS_PRECISION_1), // = 8
    .BIAS_TENSOR_SIZE_DIM_0(linear1_BIAS_TENSOR_SIZE_DIM_0), // = 768
    .BIAS_PARALLELISM_DIM_0(linear1_BIAS_PARALLELISM_DIM_0), // = 16
    .BIAS_TENSOR_SIZE_DIM_1(linear1_BIAS_TENSOR_SIZE_DIM_1), // = 1
    .BIAS_PARALLELISM_DIM_1(linear1_BIAS_PARALLELISM_DIM_1), // = 1
    .DATA_OUT_0_PRECISION_0(linear1_DATA_OUT_0_PRECISION_0), // = 8
    .DATA_OUT_0_PRECISION_1(linear1_DATA_OUT_0_PRECISION_1), // = 8
    .DATA_OUT_0_TENSOR_SIZE_DIM_0(linear1_DATA_OUT_0_TENSOR_SIZE_DIM_0), // = 768
    .DATA_OUT_0_PARALLELISM_DIM_0(linear1_DATA_OUT_0_PARALLELISM_DIM_0), // = 16
    .DATA_OUT_0_TENSOR_SIZE_DIM_1(linear1_DATA_OUT_0_TENSOR_SIZE_DIM_1), // = 196
    .DATA_OUT_0_PARALLELISM_DIM_1(linear1_DATA_OUT_0_PARALLELISM_DIM_1)
) linear1_inst (
    .clk(clk),
    .rst(rst),

    .mdata_in_0(linear1_mdata_in_0),
    .edata_in_0(linear1_edata_in_0),
    .data_in_0_valid(linear1_data_in_0_valid),
    .data_in_0_ready(linear1_data_in_0_ready),
        
    .mweight(linear1_mweight),
    .eweight(linear1_eweight),
    .weight_valid(linear1_weight_valid),
    .weight_ready(linear1_weight_ready),
        
    .mbias(linear1_mbias),
    .ebias(linear1_ebias),
    .bias_valid(linear1_bias_valid),
    .bias_ready(linear1_bias_ready),
        
    .mdata_out_0(linear1_mdata_out_0),
    .edata_out_0(linear1_edata_out_0),
    .data_out_0_valid(linear1_data_out_0_valid),
    .data_out_0_ready(linear1_data_out_0_ready)
);

linear1_weight_source #(
    .WEIGHT_PRECISION_0(linear1_WEIGHT_PRECISION_0),
    .WEIGHT_PRECISION_1(linear1_WEIGHT_PRECISION_1),
    .WEIGHT_TENSOR_SIZE_DIM_0(linear1_WEIGHT_TENSOR_SIZE_DIM_0),
    .WEIGHT_PARALLELISM_DIM_0(linear1_WEIGHT_PARALLELISM_DIM_0),
    .WEIGHT_TENSOR_SIZE_DIM_1(linear1_WEIGHT_TENSOR_SIZE_DIM_1),
    .WEIGHT_PARALLELISM_DIM_1(linear1_WEIGHT_PARALLELISM_DIM_1)
) linear1_weight_source_0 (
    .clk(clk),
    .rst(rst),
    .mdata_out(linear1_mweight),
    .edata_out(linear1_eweight),
    .data_out_ready(linear1_weight_ready),
    .data_out_valid(linear1_weight_valid)
);

linear1_bias_source #(
    .BIAS_PRECISION_0(linear1_BIAS_PRECISION_0),
    .BIAS_PRECISION_1(linear1_BIAS_PRECISION_1),
    .BIAS_TENSOR_SIZE_DIM_0(linear1_BIAS_TENSOR_SIZE_DIM_0),
    .BIAS_PARALLELISM_DIM_0(linear1_BIAS_PARALLELISM_DIM_0),
    .BIAS_TENSOR_SIZE_DIM_1(linear1_BIAS_TENSOR_SIZE_DIM_1),
    .BIAS_PARALLELISM_DIM_1(linear1_BIAS_PARALLELISM_DIM_1)
) linear1_bias_source_0 (
    .clk(clk),
    .rst(rst),
    .mdata_out(linear1_mbias),
    .edata_out(linear1_ebias),
    .data_out_ready(linear1_bias_ready),
    .data_out_valid(linear1_bias_valid)
);

// act
mxint_gelu #(
    .DATA_IN_0_PRECISION_0(act_DATA_IN_0_PRECISION_0), // = 8
    .DATA_IN_0_PRECISION_1(act_DATA_IN_0_PRECISION_1), // = 8
    .DATA_IN_0_TENSOR_SIZE_DIM_0(act_DATA_IN_0_TENSOR_SIZE_DIM_0), // = 768
    .DATA_IN_0_PARALLELISM_DIM_0(act_DATA_IN_0_PARALLELISM_DIM_0), // = 16
    .DATA_IN_0_TENSOR_SIZE_DIM_1(act_DATA_IN_0_TENSOR_SIZE_DIM_1), // = 196
    .DATA_IN_0_PARALLELISM_DIM_1(act_DATA_IN_0_PARALLELISM_DIM_1), // = 1
    .DATA_OUT_0_PRECISION_0(act_DATA_OUT_0_PRECISION_0), // = 8
    .DATA_OUT_0_PRECISION_1(act_DATA_OUT_0_PRECISION_1), // = 8
    .DATA_OUT_0_TENSOR_SIZE_DIM_0(act_DATA_OUT_0_TENSOR_SIZE_DIM_0), // = 768
    .DATA_OUT_0_PARALLELISM_DIM_0(act_DATA_OUT_0_PARALLELISM_DIM_0), // = 16
    .DATA_OUT_0_TENSOR_SIZE_DIM_1(act_DATA_OUT_0_TENSOR_SIZE_DIM_1), // = 196
    .DATA_OUT_0_PARALLELISM_DIM_1(act_DATA_OUT_0_PARALLELISM_DIM_1)
) act_inst (
    .clk(clk),
    .rst(rst),

    .mdata_in_0(act_mdata_in_0),
    .edata_in_0(act_edata_in_0),
    .data_in_0_valid(act_data_in_0_valid),
    .data_in_0_ready(act_data_in_0_ready),
        
    .mdata_out_0(act_mdata_out_0),
    .edata_out_0(act_edata_out_0),
    .data_out_0_valid(act_data_out_0_valid),
    .data_out_0_ready(act_data_out_0_ready)
);

// linear2
mxint_linear #(
    .DATA_IN_0_PRECISION_0(linear2_DATA_IN_0_PRECISION_0), // = 8
    .DATA_IN_0_PRECISION_1(linear2_DATA_IN_0_PRECISION_1), // = 8
    .DATA_IN_0_TENSOR_SIZE_DIM_0(linear2_DATA_IN_0_TENSOR_SIZE_DIM_0), // = 768
    .DATA_IN_0_PARALLELISM_DIM_0(linear2_DATA_IN_0_PARALLELISM_DIM_0), // = 16
    .DATA_IN_0_TENSOR_SIZE_DIM_1(linear2_DATA_IN_0_TENSOR_SIZE_DIM_1), // = 196
    .DATA_IN_0_PARALLELISM_DIM_1(linear2_DATA_IN_0_PARALLELISM_DIM_1), // = 1
    .WEIGHT_PRECISION_0(linear2_WEIGHT_PRECISION_0), // = 6
    .WEIGHT_PRECISION_1(linear2_WEIGHT_PRECISION_1), // = 8
    .WEIGHT_TENSOR_SIZE_DIM_0(linear2_WEIGHT_TENSOR_SIZE_DIM_0), // = 768
    .WEIGHT_PARALLELISM_DIM_0(linear2_WEIGHT_PARALLELISM_DIM_0), // = 16
    .WEIGHT_TENSOR_SIZE_DIM_1(linear2_WEIGHT_TENSOR_SIZE_DIM_1), // = 192
    .WEIGHT_PARALLELISM_DIM_1(linear2_WEIGHT_PARALLELISM_DIM_1), // = 16
    .BIAS_PRECISION_0(linear2_BIAS_PRECISION_0), // = 6
    .BIAS_PRECISION_1(linear2_BIAS_PRECISION_1), // = 8
    .BIAS_TENSOR_SIZE_DIM_0(linear2_BIAS_TENSOR_SIZE_DIM_0), // = 192
    .BIAS_PARALLELISM_DIM_0(linear2_BIAS_PARALLELISM_DIM_0), // = 16
    .BIAS_TENSOR_SIZE_DIM_1(linear2_BIAS_TENSOR_SIZE_DIM_1), // = 1
    .BIAS_PARALLELISM_DIM_1(linear2_BIAS_PARALLELISM_DIM_1), // = 1
    .DATA_OUT_0_PRECISION_0(linear2_DATA_OUT_0_PRECISION_0), // = 8
    .DATA_OUT_0_PRECISION_1(linear2_DATA_OUT_0_PRECISION_1), // = 8
    .DATA_OUT_0_TENSOR_SIZE_DIM_0(linear2_DATA_OUT_0_TENSOR_SIZE_DIM_0), // = 192
    .DATA_OUT_0_PARALLELISM_DIM_0(linear2_DATA_OUT_0_PARALLELISM_DIM_0), // = 16
    .DATA_OUT_0_TENSOR_SIZE_DIM_1(linear2_DATA_OUT_0_TENSOR_SIZE_DIM_1), // = 196
    .DATA_OUT_0_PARALLELISM_DIM_1(linear2_DATA_OUT_0_PARALLELISM_DIM_1)
) linear2_inst (
    .clk(clk),
    .rst(rst),

    .mdata_in_0(linear2_mdata_in_0),
    .edata_in_0(linear2_edata_in_0),
    .data_in_0_valid(linear2_data_in_0_valid),
    .data_in_0_ready(linear2_data_in_0_ready),
        
    .mweight(linear2_mweight),
    .eweight(linear2_eweight),
    .weight_valid(linear2_weight_valid),
    .weight_ready(linear2_weight_ready),
        
    .mbias(linear2_mbias),
    .ebias(linear2_ebias),
    .bias_valid(linear2_bias_valid),
    .bias_ready(linear2_bias_ready),
        
    .mdata_out_0(linear2_mdata_out_0),
    .edata_out_0(linear2_edata_out_0),
    .data_out_0_valid(linear2_data_out_0_valid),
    .data_out_0_ready(linear2_data_out_0_ready)
);

linear2_weight_source #(
    .WEIGHT_PRECISION_0(linear2_WEIGHT_PRECISION_0),
    .WEIGHT_PRECISION_1(linear2_WEIGHT_PRECISION_1),
    .WEIGHT_TENSOR_SIZE_DIM_0(linear2_WEIGHT_TENSOR_SIZE_DIM_0),
    .WEIGHT_PARALLELISM_DIM_0(linear2_WEIGHT_PARALLELISM_DIM_0),
    .WEIGHT_TENSOR_SIZE_DIM_1(linear2_WEIGHT_TENSOR_SIZE_DIM_1),
    .WEIGHT_PARALLELISM_DIM_1(linear2_WEIGHT_PARALLELISM_DIM_1)
) linear2_weight_source_0 (
    .clk(clk),
    .rst(rst),
    .mdata_out(linear2_mweight),
    .edata_out(linear2_eweight),
    .data_out_ready(linear2_weight_ready),
    .data_out_valid(linear2_weight_valid)
);

linear2_bias_source #(
    .BIAS_PRECISION_0(linear2_BIAS_PRECISION_0),
    .BIAS_PRECISION_1(linear2_BIAS_PRECISION_1),
    .BIAS_TENSOR_SIZE_DIM_0(linear2_BIAS_TENSOR_SIZE_DIM_0),
    .BIAS_PARALLELISM_DIM_0(linear2_BIAS_PARALLELISM_DIM_0),
    .BIAS_TENSOR_SIZE_DIM_1(linear2_BIAS_TENSOR_SIZE_DIM_1),
    .BIAS_PARALLELISM_DIM_1(linear2_BIAS_PARALLELISM_DIM_1)
) linear2_bias_source_0 (
    .clk(clk),
    .rst(rst),
    .mdata_out(linear2_mbias),
    .edata_out(linear2_ebias),
    .data_out_ready(linear2_bias_ready),
    .data_out_valid(linear2_bias_valid)
);

// norm1
mxint_layernorm #(
    .DATA_IN_0_PRECISION_0(norm1_DATA_IN_0_PRECISION_0), // = 8
    .DATA_IN_0_PRECISION_1(norm1_DATA_IN_0_PRECISION_1), // = 8
    .DATA_IN_0_TENSOR_SIZE_DIM_0(norm1_DATA_IN_0_TENSOR_SIZE_DIM_0), // = 192
    .DATA_IN_0_PARALLELISM_DIM_0(norm1_DATA_IN_0_PARALLELISM_DIM_0), // = 16
    .DATA_IN_0_TENSOR_SIZE_DIM_1(norm1_DATA_IN_0_TENSOR_SIZE_DIM_1), // = 196
    .DATA_IN_0_PARALLELISM_DIM_1(norm1_DATA_IN_0_PARALLELISM_DIM_1), // = 1
    .WEIGHT_PRECISION_0(norm1_WEIGHT_PRECISION_0), // = 6
    .WEIGHT_PRECISION_1(norm1_WEIGHT_PRECISION_1), // = 8
    .WEIGHT_TENSOR_SIZE_DIM_0(norm1_WEIGHT_TENSOR_SIZE_DIM_0), // = 192
    .WEIGHT_PARALLELISM_DIM_0(norm1_WEIGHT_PARALLELISM_DIM_0), // = 16
    .WEIGHT_TENSOR_SIZE_DIM_1(norm1_WEIGHT_TENSOR_SIZE_DIM_1), // = 1
    .WEIGHT_PARALLELISM_DIM_1(norm1_WEIGHT_PARALLELISM_DIM_1), // = 1
    .BIAS_PRECISION_0(norm1_BIAS_PRECISION_0), // = 6
    .BIAS_PRECISION_1(norm1_BIAS_PRECISION_1), // = 8
    .BIAS_TENSOR_SIZE_DIM_0(norm1_BIAS_TENSOR_SIZE_DIM_0), // = 192
    .BIAS_PARALLELISM_DIM_0(norm1_BIAS_PARALLELISM_DIM_0), // = 16
    .BIAS_TENSOR_SIZE_DIM_1(norm1_BIAS_TENSOR_SIZE_DIM_1), // = 1
    .BIAS_PARALLELISM_DIM_1(norm1_BIAS_PARALLELISM_DIM_1), // = 1
    .ELEMENTWISE_AFFINE(norm1_ELEMENTWISE_AFFINE), // = 1
    .HAS_BIAS(norm1_HAS_BIAS), // = 1
    .DATA_OUT_0_PRECISION_0(norm1_DATA_OUT_0_PRECISION_0), // = 8
    .DATA_OUT_0_PRECISION_1(norm1_DATA_OUT_0_PRECISION_1), // = 8
    .DATA_OUT_0_TENSOR_SIZE_DIM_0(norm1_DATA_OUT_0_TENSOR_SIZE_DIM_0), // = 192
    .DATA_OUT_0_PARALLELISM_DIM_0(norm1_DATA_OUT_0_PARALLELISM_DIM_0), // = 16
    .DATA_OUT_0_TENSOR_SIZE_DIM_1(norm1_DATA_OUT_0_TENSOR_SIZE_DIM_1), // = 196
    .DATA_OUT_0_PARALLELISM_DIM_1(norm1_DATA_OUT_0_PARALLELISM_DIM_1)
) norm1_inst (
    .clk(clk),
    .rst(rst),

    .mdata_in_0(norm1_mdata_in_0),
    .edata_in_0(norm1_edata_in_0),
    .data_in_0_valid(norm1_data_in_0_valid),
    .data_in_0_ready(norm1_data_in_0_ready),
        
    .mweight(norm1_mweight),
    .eweight(norm1_eweight),
    .weight_valid(norm1_weight_valid),
    .weight_ready(norm1_weight_ready),
        
    .mbias(norm1_mbias),
    .ebias(norm1_ebias),
    .bias_valid(norm1_bias_valid),
    .bias_ready(norm1_bias_ready),
        
    .mdata_out_0(norm1_mdata_out_0),
    .edata_out_0(norm1_edata_out_0),
    .data_out_0_valid(norm1_data_out_0_valid),
    .data_out_0_ready(norm1_data_out_0_ready)
);

norm1_weight_source #(
    .WEIGHT_PRECISION_0(norm1_WEIGHT_PRECISION_0),
    .WEIGHT_PRECISION_1(norm1_WEIGHT_PRECISION_1),
    .WEIGHT_TENSOR_SIZE_DIM_0(norm1_WEIGHT_TENSOR_SIZE_DIM_0),
    .WEIGHT_PARALLELISM_DIM_0(norm1_WEIGHT_PARALLELISM_DIM_0),
    .WEIGHT_TENSOR_SIZE_DIM_1(norm1_WEIGHT_TENSOR_SIZE_DIM_1),
    .WEIGHT_PARALLELISM_DIM_1(norm1_WEIGHT_PARALLELISM_DIM_1)
) norm1_weight_source_0 (
    .clk(clk),
    .rst(rst),
    .mdata_out(norm1_mweight),
    .edata_out(norm1_eweight),
    .data_out_ready(norm1_weight_ready),
    .data_out_valid(norm1_weight_valid)
);

norm1_bias_source #(
    .BIAS_PRECISION_0(norm1_BIAS_PRECISION_0),
    .BIAS_PRECISION_1(norm1_BIAS_PRECISION_1),
    .BIAS_TENSOR_SIZE_DIM_0(norm1_BIAS_TENSOR_SIZE_DIM_0),
    .BIAS_PARALLELISM_DIM_0(norm1_BIAS_PARALLELISM_DIM_0),
    .BIAS_TENSOR_SIZE_DIM_1(norm1_BIAS_TENSOR_SIZE_DIM_1),
    .BIAS_PARALLELISM_DIM_1(norm1_BIAS_PARALLELISM_DIM_1)
) norm1_bias_source_0 (
    .clk(clk),
    .rst(rst),
    .mdata_out(norm1_mbias),
    .edata_out(norm1_ebias),
    .data_out_ready(norm1_bias_ready),
    .data_out_valid(norm1_bias_valid)
);

// add
mxint_addition #(
    .DATA_IN_0_PRECISION_0(add_DATA_IN_0_PRECISION_0), // = 8
    .DATA_IN_0_PRECISION_1(add_DATA_IN_0_PRECISION_1), // = 8
    .DATA_IN_0_TENSOR_SIZE_DIM_0(add_DATA_IN_0_TENSOR_SIZE_DIM_0), // = 192
    .DATA_IN_0_PARALLELISM_DIM_0(add_DATA_IN_0_PARALLELISM_DIM_0), // = 16
    .DATA_IN_0_TENSOR_SIZE_DIM_1(add_DATA_IN_0_TENSOR_SIZE_DIM_1), // = 196
    .DATA_IN_0_PARALLELISM_DIM_1(add_DATA_IN_0_PARALLELISM_DIM_1), // = 1
    .DATA_IN_1_PRECISION_0(add_DATA_IN_1_PRECISION_0), // = 8
    .DATA_IN_1_PRECISION_1(add_DATA_IN_1_PRECISION_1), // = 8
    .DATA_IN_1_TENSOR_SIZE_DIM_0(add_DATA_IN_1_TENSOR_SIZE_DIM_0), // = 192
    .DATA_IN_1_PARALLELISM_DIM_0(add_DATA_IN_1_PARALLELISM_DIM_0), // = 16
    .DATA_IN_1_TENSOR_SIZE_DIM_1(add_DATA_IN_1_TENSOR_SIZE_DIM_1), // = 196
    .DATA_IN_1_PARALLELISM_DIM_1(add_DATA_IN_1_PARALLELISM_DIM_1), // = 1
    .DATA_OUT_0_PRECISION_0(add_DATA_OUT_0_PRECISION_0), // = 8
    .DATA_OUT_0_PRECISION_1(add_DATA_OUT_0_PRECISION_1), // = 8
    .DATA_OUT_0_TENSOR_SIZE_DIM_0(add_DATA_OUT_0_TENSOR_SIZE_DIM_0), // = 192
    .DATA_OUT_0_PARALLELISM_DIM_0(add_DATA_OUT_0_PARALLELISM_DIM_0), // = 16
    .DATA_OUT_0_TENSOR_SIZE_DIM_1(add_DATA_OUT_0_TENSOR_SIZE_DIM_1), // = 196
    .DATA_OUT_0_PARALLELISM_DIM_1(add_DATA_OUT_0_PARALLELISM_DIM_1)
) add_inst (
    .clk(clk),
    .rst(rst),

    .mdata_in_0(add_mdata_in_0),
    .edata_in_0(add_edata_in_0),
    .data_in_0_valid(add_data_in_0_valid),
    .data_in_0_ready(add_data_in_0_ready),
        
    .mdata_in_1(add_mdata_in_1),
    .edata_in_1(add_edata_in_1),
    .data_in_1_valid(add_data_in_1_valid),
    .data_in_1_ready(add_data_in_1_ready),
        
    .mdata_out_0(add_mdata_out_0),
    .edata_out_0(add_edata_out_0),
    .data_out_0_valid(add_data_out_0_valid),
    .data_out_0_ready(add_data_out_0_ready)
);

// fork2_1
mxint_fork2 #(
    .DATA_IN_0_PRECISION_0(fork2_1_DATA_IN_0_PRECISION_0), // = 8
    .DATA_IN_0_PRECISION_1(fork2_1_DATA_IN_0_PRECISION_1), // = 8
    .DATA_IN_0_TENSOR_SIZE_DIM_0(fork2_1_DATA_IN_0_TENSOR_SIZE_DIM_0), // = 192
    .DATA_IN_0_PARALLELISM_DIM_0(fork2_1_DATA_IN_0_PARALLELISM_DIM_0), // = 16
    .DATA_IN_0_TENSOR_SIZE_DIM_1(fork2_1_DATA_IN_0_TENSOR_SIZE_DIM_1), // = 196
    .DATA_IN_0_PARALLELISM_DIM_1(fork2_1_DATA_IN_0_PARALLELISM_DIM_1), // = 1
    .DATA_OUT_0_PRECISION_0(fork2_1_DATA_OUT_0_PRECISION_0), // = 8
    .DATA_OUT_0_PRECISION_1(fork2_1_DATA_OUT_0_PRECISION_1), // = 8
    .DATA_OUT_0_TENSOR_SIZE_DIM_0(fork2_1_DATA_OUT_0_TENSOR_SIZE_DIM_0), // = 192
    .DATA_OUT_0_PARALLELISM_DIM_0(fork2_1_DATA_OUT_0_PARALLELISM_DIM_0), // = 16
    .DATA_OUT_0_TENSOR_SIZE_DIM_1(fork2_1_DATA_OUT_0_TENSOR_SIZE_DIM_1), // = 196
    .DATA_OUT_0_PARALLELISM_DIM_1(fork2_1_DATA_OUT_0_PARALLELISM_DIM_1), // = 1
    .DATA_OUT_1_PRECISION_0(fork2_1_DATA_OUT_1_PRECISION_0), // = 8
    .DATA_OUT_1_PRECISION_1(fork2_1_DATA_OUT_1_PRECISION_1), // = 8
    .DATA_OUT_1_TENSOR_SIZE_DIM_0(fork2_1_DATA_OUT_1_TENSOR_SIZE_DIM_0), // = 192
    .DATA_OUT_1_PARALLELISM_DIM_0(fork2_1_DATA_OUT_1_PARALLELISM_DIM_0), // = 16
    .DATA_OUT_1_TENSOR_SIZE_DIM_1(fork2_1_DATA_OUT_1_TENSOR_SIZE_DIM_1), // = 196
    .DATA_OUT_1_PARALLELISM_DIM_1(fork2_1_DATA_OUT_1_PARALLELISM_DIM_1)
) fork2_1_inst (
    .clk(clk),
    .rst(rst),

    .mdata_in_0(fork2_1_mdata_in_0),
    .edata_in_0(fork2_1_edata_in_0),
    .data_in_0_valid(fork2_1_data_in_0_valid),
    .data_in_0_ready(fork2_1_data_in_0_ready),
        
    .mdata_out_0(fork2_1_mdata_out_0),
    .edata_out_0(fork2_1_edata_out_0),
    .data_out_0_valid(fork2_1_data_out_0_valid),
    .data_out_0_ready(fork2_1_data_out_0_ready),
        
    .mdata_out_1(fork2_1_mdata_out_1),
    .edata_out_1(fork2_1_edata_out_1),
    .data_out_1_valid(fork2_1_data_out_1_valid),
    .data_out_1_ready(fork2_1_data_out_1_ready)
);

// attention
mxint_vit_attention_wrap #(
    .DATA_IN_0_PRECISION_0(attention_DATA_IN_0_PRECISION_0), // = 8
    .DATA_IN_0_PRECISION_1(attention_DATA_IN_0_PRECISION_1), // = 8
    .DATA_IN_0_TENSOR_SIZE_DIM_0(attention_DATA_IN_0_TENSOR_SIZE_DIM_0), // = 192
    .DATA_IN_0_PARALLELISM_DIM_0(attention_DATA_IN_0_PARALLELISM_DIM_0), // = 16
    .DATA_IN_0_TENSOR_SIZE_DIM_1(attention_DATA_IN_0_TENSOR_SIZE_DIM_1), // = 196
    .DATA_IN_0_PARALLELISM_DIM_1(attention_DATA_IN_0_PARALLELISM_DIM_1), // = 1
    .QUERY_WEIGHT_PRECISION_0(attention_QUERY_WEIGHT_PRECISION_0), // = 6
    .QUERY_WEIGHT_PRECISION_1(attention_QUERY_WEIGHT_PRECISION_1), // = 8
    .QUERY_WEIGHT_TENSOR_SIZE_DIM_0(attention_QUERY_WEIGHT_TENSOR_SIZE_DIM_0), // = 192
    .QUERY_WEIGHT_PARALLELISM_DIM_0(attention_QUERY_WEIGHT_PARALLELISM_DIM_0), // = 16
    .QUERY_WEIGHT_TENSOR_SIZE_DIM_1(attention_QUERY_WEIGHT_TENSOR_SIZE_DIM_1), // = 192
    .QUERY_WEIGHT_PARALLELISM_DIM_1(attention_QUERY_WEIGHT_PARALLELISM_DIM_1), // = 16
    .QUERY_BIAS_PRECISION_0(attention_QUERY_BIAS_PRECISION_0), // = 6
    .QUERY_BIAS_PRECISION_1(attention_QUERY_BIAS_PRECISION_1), // = 8
    .QUERY_BIAS_TENSOR_SIZE_DIM_0(attention_QUERY_BIAS_TENSOR_SIZE_DIM_0), // = 192
    .QUERY_BIAS_PARALLELISM_DIM_0(attention_QUERY_BIAS_PARALLELISM_DIM_0), // = 16
    .QUERY_BIAS_TENSOR_SIZE_DIM_1(attention_QUERY_BIAS_TENSOR_SIZE_DIM_1), // = 1
    .QUERY_BIAS_PARALLELISM_DIM_1(attention_QUERY_BIAS_PARALLELISM_DIM_1), // = 1
    .KEY_WEIGHT_PRECISION_0(attention_KEY_WEIGHT_PRECISION_0), // = 6
    .KEY_WEIGHT_PRECISION_1(attention_KEY_WEIGHT_PRECISION_1), // = 8
    .KEY_WEIGHT_TENSOR_SIZE_DIM_0(attention_KEY_WEIGHT_TENSOR_SIZE_DIM_0), // = 192
    .KEY_WEIGHT_PARALLELISM_DIM_0(attention_KEY_WEIGHT_PARALLELISM_DIM_0), // = 16
    .KEY_WEIGHT_TENSOR_SIZE_DIM_1(attention_KEY_WEIGHT_TENSOR_SIZE_DIM_1), // = 192
    .KEY_WEIGHT_PARALLELISM_DIM_1(attention_KEY_WEIGHT_PARALLELISM_DIM_1), // = 16
    .KEY_BIAS_PRECISION_0(attention_KEY_BIAS_PRECISION_0), // = 6
    .KEY_BIAS_PRECISION_1(attention_KEY_BIAS_PRECISION_1), // = 8
    .KEY_BIAS_TENSOR_SIZE_DIM_0(attention_KEY_BIAS_TENSOR_SIZE_DIM_0), // = 192
    .KEY_BIAS_PARALLELISM_DIM_0(attention_KEY_BIAS_PARALLELISM_DIM_0), // = 16
    .KEY_BIAS_TENSOR_SIZE_DIM_1(attention_KEY_BIAS_TENSOR_SIZE_DIM_1), // = 1
    .KEY_BIAS_PARALLELISM_DIM_1(attention_KEY_BIAS_PARALLELISM_DIM_1), // = 1
    .VALUE_WEIGHT_PRECISION_0(attention_VALUE_WEIGHT_PRECISION_0), // = 6
    .VALUE_WEIGHT_PRECISION_1(attention_VALUE_WEIGHT_PRECISION_1), // = 8
    .VALUE_WEIGHT_TENSOR_SIZE_DIM_0(attention_VALUE_WEIGHT_TENSOR_SIZE_DIM_0), // = 192
    .VALUE_WEIGHT_PARALLELISM_DIM_0(attention_VALUE_WEIGHT_PARALLELISM_DIM_0), // = 16
    .VALUE_WEIGHT_TENSOR_SIZE_DIM_1(attention_VALUE_WEIGHT_TENSOR_SIZE_DIM_1), // = 192
    .VALUE_WEIGHT_PARALLELISM_DIM_1(attention_VALUE_WEIGHT_PARALLELISM_DIM_1), // = 16
    .VALUE_BIAS_PRECISION_0(attention_VALUE_BIAS_PRECISION_0), // = 6
    .VALUE_BIAS_PRECISION_1(attention_VALUE_BIAS_PRECISION_1), // = 8
    .VALUE_BIAS_TENSOR_SIZE_DIM_0(attention_VALUE_BIAS_TENSOR_SIZE_DIM_0), // = 192
    .VALUE_BIAS_PARALLELISM_DIM_0(attention_VALUE_BIAS_PARALLELISM_DIM_0), // = 16
    .VALUE_BIAS_TENSOR_SIZE_DIM_1(attention_VALUE_BIAS_TENSOR_SIZE_DIM_1), // = 1
    .VALUE_BIAS_PARALLELISM_DIM_1(attention_VALUE_BIAS_PARALLELISM_DIM_1), // = 1
    .PROJ_WEIGHT_PRECISION_0(attention_PROJ_WEIGHT_PRECISION_0), // = 6
    .PROJ_WEIGHT_PRECISION_1(attention_PROJ_WEIGHT_PRECISION_1), // = 8
    .PROJ_WEIGHT_TENSOR_SIZE_DIM_0(attention_PROJ_WEIGHT_TENSOR_SIZE_DIM_0), // = 192
    .PROJ_WEIGHT_PARALLELISM_DIM_0(attention_PROJ_WEIGHT_PARALLELISM_DIM_0), // = 16
    .PROJ_WEIGHT_TENSOR_SIZE_DIM_1(attention_PROJ_WEIGHT_TENSOR_SIZE_DIM_1), // = 192
    .PROJ_WEIGHT_PARALLELISM_DIM_1(attention_PROJ_WEIGHT_PARALLELISM_DIM_1), // = 16
    .PROJ_BIAS_PRECISION_0(attention_PROJ_BIAS_PRECISION_0), // = 6
    .PROJ_BIAS_PRECISION_1(attention_PROJ_BIAS_PRECISION_1), // = 8
    .PROJ_BIAS_TENSOR_SIZE_DIM_0(attention_PROJ_BIAS_TENSOR_SIZE_DIM_0), // = 192
    .PROJ_BIAS_PARALLELISM_DIM_0(attention_PROJ_BIAS_PARALLELISM_DIM_0), // = 16
    .PROJ_BIAS_TENSOR_SIZE_DIM_1(attention_PROJ_BIAS_TENSOR_SIZE_DIM_1), // = 1
    .PROJ_BIAS_PARALLELISM_DIM_1(attention_PROJ_BIAS_PARALLELISM_DIM_1), // = 1
    .DATA_OUT_0_PRECISION_0(attention_DATA_OUT_0_PRECISION_0), // = 8
    .DATA_OUT_0_PRECISION_1(attention_DATA_OUT_0_PRECISION_1), // = 8
    .DATA_OUT_0_TENSOR_SIZE_DIM_0(attention_DATA_OUT_0_TENSOR_SIZE_DIM_0), // = 192
    .DATA_OUT_0_PARALLELISM_DIM_0(attention_DATA_OUT_0_PARALLELISM_DIM_0), // = 16
    .DATA_OUT_0_TENSOR_SIZE_DIM_1(attention_DATA_OUT_0_TENSOR_SIZE_DIM_1), // = 196
    .DATA_OUT_0_PARALLELISM_DIM_1(attention_DATA_OUT_0_PARALLELISM_DIM_1)
) attention_inst (
    .clk(clk),
    .rst(rst),

    .mdata_in_0(attention_mdata_in_0),
    .edata_in_0(attention_edata_in_0),
    .data_in_0_valid(attention_data_in_0_valid),
    .data_in_0_ready(attention_data_in_0_ready),
        
    .mquery_weight(attention_mquery_weight),
    .equery_weight(attention_equery_weight),
    .query_weight_valid(attention_query_weight_valid),
    .query_weight_ready(attention_query_weight_ready),
        
    .mquery_bias(attention_mquery_bias),
    .equery_bias(attention_equery_bias),
    .query_bias_valid(attention_query_bias_valid),
    .query_bias_ready(attention_query_bias_ready),
        
    .mkey_weight(attention_mkey_weight),
    .ekey_weight(attention_ekey_weight),
    .key_weight_valid(attention_key_weight_valid),
    .key_weight_ready(attention_key_weight_ready),
        
    .mkey_bias(attention_mkey_bias),
    .ekey_bias(attention_ekey_bias),
    .key_bias_valid(attention_key_bias_valid),
    .key_bias_ready(attention_key_bias_ready),
        
    .mvalue_weight(attention_mvalue_weight),
    .evalue_weight(attention_evalue_weight),
    .value_weight_valid(attention_value_weight_valid),
    .value_weight_ready(attention_value_weight_ready),
        
    .mvalue_bias(attention_mvalue_bias),
    .evalue_bias(attention_evalue_bias),
    .value_bias_valid(attention_value_bias_valid),
    .value_bias_ready(attention_value_bias_ready),
        
    .mproj_weight(attention_mproj_weight),
    .eproj_weight(attention_eproj_weight),
    .proj_weight_valid(attention_proj_weight_valid),
    .proj_weight_ready(attention_proj_weight_ready),
        
    .mproj_bias(attention_mproj_bias),
    .eproj_bias(attention_eproj_bias),
    .proj_bias_valid(attention_proj_bias_valid),
    .proj_bias_ready(attention_proj_bias_ready),
        
    .mdata_out_0(attention_mdata_out_0),
    .edata_out_0(attention_edata_out_0),
    .data_out_0_valid(attention_data_out_0_valid),
    .data_out_0_ready(attention_data_out_0_ready)
);

attention_query_weight_source #(
    .QUERY_WEIGHT_PRECISION_0(attention_QUERY_WEIGHT_PRECISION_0),
    .QUERY_WEIGHT_PRECISION_1(attention_QUERY_WEIGHT_PRECISION_1),
    .QUERY_WEIGHT_TENSOR_SIZE_DIM_0(attention_QUERY_WEIGHT_TENSOR_SIZE_DIM_0),
    .QUERY_WEIGHT_PARALLELISM_DIM_0(attention_QUERY_WEIGHT_PARALLELISM_DIM_0),
    .QUERY_WEIGHT_TENSOR_SIZE_DIM_1(attention_QUERY_WEIGHT_TENSOR_SIZE_DIM_1),
    .QUERY_WEIGHT_PARALLELISM_DIM_1(attention_QUERY_WEIGHT_PARALLELISM_DIM_1)
) attention_query_weight_source_0 (
    .clk(clk),
    .rst(rst),
    .mdata_out(attention_mquery_weight),
    .edata_out(attention_equery_weight),
    .data_out_ready(attention_query_weight_ready),
    .data_out_valid(attention_query_weight_valid)
);

attention_query_bias_source #(
    .QUERY_BIAS_PRECISION_0(attention_QUERY_BIAS_PRECISION_0),
    .QUERY_BIAS_PRECISION_1(attention_QUERY_BIAS_PRECISION_1),
    .QUERY_BIAS_TENSOR_SIZE_DIM_0(attention_QUERY_BIAS_TENSOR_SIZE_DIM_0),
    .QUERY_BIAS_PARALLELISM_DIM_0(attention_QUERY_BIAS_PARALLELISM_DIM_0),
    .QUERY_BIAS_TENSOR_SIZE_DIM_1(attention_QUERY_BIAS_TENSOR_SIZE_DIM_1),
    .QUERY_BIAS_PARALLELISM_DIM_1(attention_QUERY_BIAS_PARALLELISM_DIM_1)
) attention_query_bias_source_0 (
    .clk(clk),
    .rst(rst),
    .mdata_out(attention_mquery_bias),
    .edata_out(attention_equery_bias),
    .data_out_ready(attention_query_bias_ready),
    .data_out_valid(attention_query_bias_valid)
);

attention_key_weight_source #(
    .KEY_WEIGHT_PRECISION_0(attention_KEY_WEIGHT_PRECISION_0),
    .KEY_WEIGHT_PRECISION_1(attention_KEY_WEIGHT_PRECISION_1),
    .KEY_WEIGHT_TENSOR_SIZE_DIM_0(attention_KEY_WEIGHT_TENSOR_SIZE_DIM_0),
    .KEY_WEIGHT_PARALLELISM_DIM_0(attention_KEY_WEIGHT_PARALLELISM_DIM_0),
    .KEY_WEIGHT_TENSOR_SIZE_DIM_1(attention_KEY_WEIGHT_TENSOR_SIZE_DIM_1),
    .KEY_WEIGHT_PARALLELISM_DIM_1(attention_KEY_WEIGHT_PARALLELISM_DIM_1)
) attention_key_weight_source_0 (
    .clk(clk),
    .rst(rst),
    .mdata_out(attention_mkey_weight),
    .edata_out(attention_ekey_weight),
    .data_out_ready(attention_key_weight_ready),
    .data_out_valid(attention_key_weight_valid)
);

attention_key_bias_source #(
    .KEY_BIAS_PRECISION_0(attention_KEY_BIAS_PRECISION_0),
    .KEY_BIAS_PRECISION_1(attention_KEY_BIAS_PRECISION_1),
    .KEY_BIAS_TENSOR_SIZE_DIM_0(attention_KEY_BIAS_TENSOR_SIZE_DIM_0),
    .KEY_BIAS_PARALLELISM_DIM_0(attention_KEY_BIAS_PARALLELISM_DIM_0),
    .KEY_BIAS_TENSOR_SIZE_DIM_1(attention_KEY_BIAS_TENSOR_SIZE_DIM_1),
    .KEY_BIAS_PARALLELISM_DIM_1(attention_KEY_BIAS_PARALLELISM_DIM_1)
) attention_key_bias_source_0 (
    .clk(clk),
    .rst(rst),
    .mdata_out(attention_mkey_bias),
    .edata_out(attention_ekey_bias),
    .data_out_ready(attention_key_bias_ready),
    .data_out_valid(attention_key_bias_valid)
);

attention_value_weight_source #(
    .VALUE_WEIGHT_PRECISION_0(attention_VALUE_WEIGHT_PRECISION_0),
    .VALUE_WEIGHT_PRECISION_1(attention_VALUE_WEIGHT_PRECISION_1),
    .VALUE_WEIGHT_TENSOR_SIZE_DIM_0(attention_VALUE_WEIGHT_TENSOR_SIZE_DIM_0),
    .VALUE_WEIGHT_PARALLELISM_DIM_0(attention_VALUE_WEIGHT_PARALLELISM_DIM_0),
    .VALUE_WEIGHT_TENSOR_SIZE_DIM_1(attention_VALUE_WEIGHT_TENSOR_SIZE_DIM_1),
    .VALUE_WEIGHT_PARALLELISM_DIM_1(attention_VALUE_WEIGHT_PARALLELISM_DIM_1)
) attention_value_weight_source_0 (
    .clk(clk),
    .rst(rst),
    .mdata_out(attention_mvalue_weight),
    .edata_out(attention_evalue_weight),
    .data_out_ready(attention_value_weight_ready),
    .data_out_valid(attention_value_weight_valid)
);

attention_value_bias_source #(
    .VALUE_BIAS_PRECISION_0(attention_VALUE_BIAS_PRECISION_0),
    .VALUE_BIAS_PRECISION_1(attention_VALUE_BIAS_PRECISION_1),
    .VALUE_BIAS_TENSOR_SIZE_DIM_0(attention_VALUE_BIAS_TENSOR_SIZE_DIM_0),
    .VALUE_BIAS_PARALLELISM_DIM_0(attention_VALUE_BIAS_PARALLELISM_DIM_0),
    .VALUE_BIAS_TENSOR_SIZE_DIM_1(attention_VALUE_BIAS_TENSOR_SIZE_DIM_1),
    .VALUE_BIAS_PARALLELISM_DIM_1(attention_VALUE_BIAS_PARALLELISM_DIM_1)
) attention_value_bias_source_0 (
    .clk(clk),
    .rst(rst),
    .mdata_out(attention_mvalue_bias),
    .edata_out(attention_evalue_bias),
    .data_out_ready(attention_value_bias_ready),
    .data_out_valid(attention_value_bias_valid)
);

attention_proj_weight_source #(
    .PROJ_WEIGHT_PRECISION_0(attention_PROJ_WEIGHT_PRECISION_0),
    .PROJ_WEIGHT_PRECISION_1(attention_PROJ_WEIGHT_PRECISION_1),
    .PROJ_WEIGHT_TENSOR_SIZE_DIM_0(attention_PROJ_WEIGHT_TENSOR_SIZE_DIM_0),
    .PROJ_WEIGHT_PARALLELISM_DIM_0(attention_PROJ_WEIGHT_PARALLELISM_DIM_0),
    .PROJ_WEIGHT_TENSOR_SIZE_DIM_1(attention_PROJ_WEIGHT_TENSOR_SIZE_DIM_1),
    .PROJ_WEIGHT_PARALLELISM_DIM_1(attention_PROJ_WEIGHT_PARALLELISM_DIM_1)
) attention_proj_weight_source_0 (
    .clk(clk),
    .rst(rst),
    .mdata_out(attention_mproj_weight),
    .edata_out(attention_eproj_weight),
    .data_out_ready(attention_proj_weight_ready),
    .data_out_valid(attention_proj_weight_valid)
);

attention_proj_bias_source #(
    .PROJ_BIAS_PRECISION_0(attention_PROJ_BIAS_PRECISION_0),
    .PROJ_BIAS_PRECISION_1(attention_PROJ_BIAS_PRECISION_1),
    .PROJ_BIAS_TENSOR_SIZE_DIM_0(attention_PROJ_BIAS_TENSOR_SIZE_DIM_0),
    .PROJ_BIAS_PARALLELISM_DIM_0(attention_PROJ_BIAS_PARALLELISM_DIM_0),
    .PROJ_BIAS_TENSOR_SIZE_DIM_1(attention_PROJ_BIAS_TENSOR_SIZE_DIM_1),
    .PROJ_BIAS_PARALLELISM_DIM_1(attention_PROJ_BIAS_PARALLELISM_DIM_1)
) attention_proj_bias_source_0 (
    .clk(clk),
    .rst(rst),
    .mdata_out(attention_mproj_bias),
    .edata_out(attention_eproj_bias),
    .data_out_ready(attention_proj_bias_ready),
    .data_out_valid(attention_proj_bias_valid)
);

// norm2
mxint_layernorm #(
    .DATA_IN_0_PRECISION_0(norm2_DATA_IN_0_PRECISION_0), // = 8
    .DATA_IN_0_PRECISION_1(norm2_DATA_IN_0_PRECISION_1), // = 8
    .DATA_IN_0_TENSOR_SIZE_DIM_0(norm2_DATA_IN_0_TENSOR_SIZE_DIM_0), // = 192
    .DATA_IN_0_PARALLELISM_DIM_0(norm2_DATA_IN_0_PARALLELISM_DIM_0), // = 16
    .DATA_IN_0_TENSOR_SIZE_DIM_1(norm2_DATA_IN_0_TENSOR_SIZE_DIM_1), // = 196
    .DATA_IN_0_PARALLELISM_DIM_1(norm2_DATA_IN_0_PARALLELISM_DIM_1), // = 1
    .WEIGHT_PRECISION_0(norm2_WEIGHT_PRECISION_0), // = 6
    .WEIGHT_PRECISION_1(norm2_WEIGHT_PRECISION_1), // = 8
    .WEIGHT_TENSOR_SIZE_DIM_0(norm2_WEIGHT_TENSOR_SIZE_DIM_0), // = 192
    .WEIGHT_PARALLELISM_DIM_0(norm2_WEIGHT_PARALLELISM_DIM_0), // = 16
    .WEIGHT_TENSOR_SIZE_DIM_1(norm2_WEIGHT_TENSOR_SIZE_DIM_1), // = 1
    .WEIGHT_PARALLELISM_DIM_1(norm2_WEIGHT_PARALLELISM_DIM_1), // = 1
    .BIAS_PRECISION_0(norm2_BIAS_PRECISION_0), // = 6
    .BIAS_PRECISION_1(norm2_BIAS_PRECISION_1), // = 8
    .BIAS_TENSOR_SIZE_DIM_0(norm2_BIAS_TENSOR_SIZE_DIM_0), // = 192
    .BIAS_PARALLELISM_DIM_0(norm2_BIAS_PARALLELISM_DIM_0), // = 16
    .BIAS_TENSOR_SIZE_DIM_1(norm2_BIAS_TENSOR_SIZE_DIM_1), // = 1
    .BIAS_PARALLELISM_DIM_1(norm2_BIAS_PARALLELISM_DIM_1), // = 1
    .ELEMENTWISE_AFFINE(norm2_ELEMENTWISE_AFFINE), // = 1
    .HAS_BIAS(norm2_HAS_BIAS), // = 1
    .DATA_OUT_0_PRECISION_0(norm2_DATA_OUT_0_PRECISION_0), // = 8
    .DATA_OUT_0_PRECISION_1(norm2_DATA_OUT_0_PRECISION_1), // = 8
    .DATA_OUT_0_TENSOR_SIZE_DIM_0(norm2_DATA_OUT_0_TENSOR_SIZE_DIM_0), // = 192
    .DATA_OUT_0_PARALLELISM_DIM_0(norm2_DATA_OUT_0_PARALLELISM_DIM_0), // = 16
    .DATA_OUT_0_TENSOR_SIZE_DIM_1(norm2_DATA_OUT_0_TENSOR_SIZE_DIM_1), // = 196
    .DATA_OUT_0_PARALLELISM_DIM_1(norm2_DATA_OUT_0_PARALLELISM_DIM_1)
) norm2_inst (
    .clk(clk),
    .rst(rst),

    .mdata_in_0(norm2_mdata_in_0),
    .edata_in_0(norm2_edata_in_0),
    .data_in_0_valid(norm2_data_in_0_valid),
    .data_in_0_ready(norm2_data_in_0_ready),
        
    .mweight(norm2_mweight),
    .eweight(norm2_eweight),
    .weight_valid(norm2_weight_valid),
    .weight_ready(norm2_weight_ready),
        
    .mbias(norm2_mbias),
    .ebias(norm2_ebias),
    .bias_valid(norm2_bias_valid),
    .bias_ready(norm2_bias_ready),
        
    .mdata_out_0(norm2_mdata_out_0),
    .edata_out_0(norm2_edata_out_0),
    .data_out_0_valid(norm2_data_out_0_valid),
    .data_out_0_ready(norm2_data_out_0_ready)
);

norm2_weight_source #(
    .WEIGHT_PRECISION_0(norm2_WEIGHT_PRECISION_0),
    .WEIGHT_PRECISION_1(norm2_WEIGHT_PRECISION_1),
    .WEIGHT_TENSOR_SIZE_DIM_0(norm2_WEIGHT_TENSOR_SIZE_DIM_0),
    .WEIGHT_PARALLELISM_DIM_0(norm2_WEIGHT_PARALLELISM_DIM_0),
    .WEIGHT_TENSOR_SIZE_DIM_1(norm2_WEIGHT_TENSOR_SIZE_DIM_1),
    .WEIGHT_PARALLELISM_DIM_1(norm2_WEIGHT_PARALLELISM_DIM_1)
) norm2_weight_source_0 (
    .clk(clk),
    .rst(rst),
    .mdata_out(norm2_mweight),
    .edata_out(norm2_eweight),
    .data_out_ready(norm2_weight_ready),
    .data_out_valid(norm2_weight_valid)
);

norm2_bias_source #(
    .BIAS_PRECISION_0(norm2_BIAS_PRECISION_0),
    .BIAS_PRECISION_1(norm2_BIAS_PRECISION_1),
    .BIAS_TENSOR_SIZE_DIM_0(norm2_BIAS_TENSOR_SIZE_DIM_0),
    .BIAS_PARALLELISM_DIM_0(norm2_BIAS_PARALLELISM_DIM_0),
    .BIAS_TENSOR_SIZE_DIM_1(norm2_BIAS_TENSOR_SIZE_DIM_1),
    .BIAS_PARALLELISM_DIM_1(norm2_BIAS_PARALLELISM_DIM_1)
) norm2_bias_source_0 (
    .clk(clk),
    .rst(rst),
    .mdata_out(norm2_mbias),
    .edata_out(norm2_ebias),
    .data_out_ready(norm2_bias_ready),
    .data_out_valid(norm2_bias_valid)
);

// add_1
mxint_addition #(
    .DATA_IN_0_PRECISION_0(add_1_DATA_IN_0_PRECISION_0), // = 8
    .DATA_IN_0_PRECISION_1(add_1_DATA_IN_0_PRECISION_1), // = 8
    .DATA_IN_0_TENSOR_SIZE_DIM_0(add_1_DATA_IN_0_TENSOR_SIZE_DIM_0), // = 192
    .DATA_IN_0_PARALLELISM_DIM_0(add_1_DATA_IN_0_PARALLELISM_DIM_0), // = 16
    .DATA_IN_0_TENSOR_SIZE_DIM_1(add_1_DATA_IN_0_TENSOR_SIZE_DIM_1), // = 196
    .DATA_IN_0_PARALLELISM_DIM_1(add_1_DATA_IN_0_PARALLELISM_DIM_1), // = 1
    .DATA_IN_1_PRECISION_0(add_1_DATA_IN_1_PRECISION_0), // = 8
    .DATA_IN_1_PRECISION_1(add_1_DATA_IN_1_PRECISION_1), // = 8
    .DATA_IN_1_TENSOR_SIZE_DIM_0(add_1_DATA_IN_1_TENSOR_SIZE_DIM_0), // = 192
    .DATA_IN_1_PARALLELISM_DIM_0(add_1_DATA_IN_1_PARALLELISM_DIM_0), // = 16
    .DATA_IN_1_TENSOR_SIZE_DIM_1(add_1_DATA_IN_1_TENSOR_SIZE_DIM_1), // = 196
    .DATA_IN_1_PARALLELISM_DIM_1(add_1_DATA_IN_1_PARALLELISM_DIM_1), // = 1
    .DATA_OUT_0_PRECISION_0(add_1_DATA_OUT_0_PRECISION_0), // = 8
    .DATA_OUT_0_PRECISION_1(add_1_DATA_OUT_0_PRECISION_1), // = 8
    .DATA_OUT_0_TENSOR_SIZE_DIM_0(add_1_DATA_OUT_0_TENSOR_SIZE_DIM_0), // = 192
    .DATA_OUT_0_PARALLELISM_DIM_0(add_1_DATA_OUT_0_PARALLELISM_DIM_0), // = 16
    .DATA_OUT_0_TENSOR_SIZE_DIM_1(add_1_DATA_OUT_0_TENSOR_SIZE_DIM_1), // = 196
    .DATA_OUT_0_PARALLELISM_DIM_1(add_1_DATA_OUT_0_PARALLELISM_DIM_1)
) add_1_inst (
    .clk(clk),
    .rst(rst),

    .mdata_in_0(add_1_mdata_in_0),
    .edata_in_0(add_1_edata_in_0),
    .data_in_0_valid(add_1_data_in_0_valid),
    .data_in_0_ready(add_1_data_in_0_ready),
        
    .mdata_in_1(add_1_mdata_in_1),
    .edata_in_1(add_1_edata_in_1),
    .data_in_1_valid(add_1_data_in_1_valid),
    .data_in_1_ready(add_1_data_in_1_ready),
        
    .mdata_out_0(add_1_mdata_out_0),
    .edata_out_0(add_1_edata_out_0),
    .data_out_0_valid(add_1_data_out_0_valid),
    .data_out_0_ready(add_1_data_out_0_ready)
);


// --------------------------
//   Interconnections
// --------------------------
    
assign data_in_0_ready = fork2_data_in_0_ready;
assign fork2_data_in_0_valid    = data_in_0_valid;
assign fork2_mdata_in_0    = mdata_in_0;
assign fork2_edata_in_0    = edata_in_0;

assign data_out_0_valid = add_1_data_out_0_valid;
assign add_1_data_out_0_ready    = data_out_0_ready;
assign mdata_out_0 = add_1_mdata_out_0;
assign edata_out_0 = add_1_edata_out_0;

assign fork2_data_out_0_ready  = linear1_data_in_0_ready;
assign linear1_data_in_0_valid    = fork2_data_out_0_valid;
assign linear1_mdata_in_0 = fork2_mdata_out_0;
assign linear1_edata_in_0 = fork2_edata_out_0;

assign linear1_data_out_0_ready  = act_data_in_0_ready;
assign act_data_in_0_valid    = linear1_data_out_0_valid;
assign act_mdata_in_0 = linear1_mdata_out_0;
assign act_edata_in_0 = linear1_edata_out_0;

assign act_data_out_0_ready  = linear2_data_in_0_ready;
assign linear2_data_in_0_valid    = act_data_out_0_valid;
assign linear2_mdata_in_0 = act_mdata_out_0;
assign linear2_edata_in_0 = act_edata_out_0;

assign linear2_data_out_0_ready  = norm1_data_in_0_ready;
assign norm1_data_in_0_valid    = linear2_data_out_0_valid;
assign norm1_mdata_in_0 = linear2_mdata_out_0;
assign norm1_edata_in_0 = linear2_edata_out_0;

assign norm1_data_out_0_ready  = add_data_in_0_ready;
assign add_data_in_0_valid    = norm1_data_out_0_valid;
assign add_mdata_in_0 = norm1_mdata_out_0;
assign add_edata_in_0 = norm1_edata_out_0;

assign fork2_data_out_1_ready  = add_data_in_1_ready;
assign add_data_in_1_valid    = fork2_data_out_1_valid;
assign add_mdata_in_1 = fork2_mdata_out_1;
assign add_edata_in_1 = fork2_edata_out_1;

assign add_data_out_0_ready  = fork2_1_data_in_0_ready;
assign fork2_1_data_in_0_valid    = add_data_out_0_valid;
assign fork2_1_mdata_in_0 = add_mdata_out_0;
assign fork2_1_edata_in_0 = add_edata_out_0;

assign fork2_1_data_out_0_ready  = attention_data_in_0_ready;
assign attention_data_in_0_valid    = fork2_1_data_out_0_valid;
assign attention_mdata_in_0 = fork2_1_mdata_out_0;
assign attention_edata_in_0 = fork2_1_edata_out_0;

assign attention_data_out_0_ready  = norm2_data_in_0_ready;
assign norm2_data_in_0_valid    = attention_data_out_0_valid;
assign norm2_mdata_in_0 = attention_mdata_out_0;
assign norm2_edata_in_0 = attention_edata_out_0;

assign norm2_data_out_0_ready  = add_1_data_in_0_ready;
assign add_1_data_in_0_valid    = norm2_data_out_0_valid;
assign add_1_mdata_in_0 = norm2_mdata_out_0;
assign add_1_edata_in_0 = norm2_edata_out_0;

assign fork2_1_data_out_1_ready  = add_1_data_in_1_ready;
assign add_1_data_in_1_valid    = fork2_1_data_out_1_valid;
assign add_1_mdata_in_1 = fork2_1_mdata_out_1;
assign add_1_edata_in_1 = fork2_1_edata_out_1;

endmodule
    