`timescale 1ns / 1ps
module swintop_simple #(
    parameter PRECISION_0 = 16,
    parameter PRECISION_1 = 8,

    parameter PARALLELISM_DIM0 = 2,
    parameter PARALLELISM_DIM1 = 2,

    parameter ADDER_FIFO_PL0_0_DATA_WIDTH = PRECISION_0,
    parameter ADDER_FIFO_PL0_0_DIM0 = PARALLELISM_DIM0,
    parameter ADDER_FIFO_PL0_0_DIM1 = PARALLELISM_DIM1,
    parameter ADDER_FIFO_PL0_0_FIFO_SIZE = 16,
    parameter LAYER_NORM_PL0_0_TOTAL_MAX_DIM0 = 16,
    parameter LAYER_NORM_PL0_0_TOTAL_MAX_DIM1 = 16,
    parameter LAYER_NORM_PL0_0_PARALLELISM_DIM0 = PARALLELISM_DIM0,
    parameter LAYER_NORM_PL0_0_PARALLELISM_DIM1 = PARALLELISM_DIM1,
    parameter LAYER_NORM_PL0_0_PRECISION_0 = PRECISION_0,
    parameter LAYER_NORM_PL0_0_PRECISION_1 = PRECISION_1,
    parameter MHA_PL0_0_DATA_IN_0_MAX_TENSOR_SIZE_DIM_0 = 16,
    parameter MHA_PL0_0_DATA_IN_0_MAX_TENSOR_SIZE_DIM_1 = 16,
    parameter MHA_PL0_0_DATA_IN_0_PARALLELISM_DIM_0 = PARALLELISM_DIM0,
    parameter MHA_PL0_0_DATA_IN_0_PARALLELISM_DIM_1 = PARALLELISM_DIM1,
    parameter MHA_PL0_0_DATA_IN_0_PRECISION_0 = PRECISION_0,
    parameter MHA_PL0_0_DATA_IN_0_PRECISION_1 = PRECISION_1,
    parameter MHA_PL0_0_WEIGHT_MAX_TENSOR_SIZE_DIM_0 = 16,
    parameter MHA_PL0_0_WEIGHT_MAX_TENSOR_SIZE_DIM_1 = 16,
    parameter MHA_PL0_0_WEIGHT_PARALLELISM_DIM_0 = PARALLELISM_DIM0,
    parameter MHA_PL0_0_WEIGHT_PARALLELISM_DIM_1 = PARALLELISM_DIM1,
    parameter MHA_PL0_0_WEIGHT_PRECISION_0 = PRECISION_0,
    parameter MHA_PL0_0_WEIGHT_PRECISION_1 = PRECISION_1,
    parameter MHA_PL0_0_BIAS_MAX_TENSOR_SIZE_DIM_0 = 16,
    parameter MHA_PL0_0_BIAS_MAX_TENSOR_SIZE_DIM_1 = 16,
    parameter MHA_PL0_0_BIAS_PARALLELISM_DIM_0 = PARALLELISM_DIM0,
    parameter MHA_PL0_0_BIAS_PARALLELISM_DIM_1 = PARALLELISM_DIM1,
    parameter MHA_PL0_0_BIAS_PRECISION_0 = PRECISION_0,
    parameter MHA_PL0_0_BIAS_PRECISION_1 = PRECISION_1,
    parameter MHA_PL0_0_DATA_OUT_0_MAX_TENSOR_SIZE_DIM_0 = 16,
    parameter MHA_PL0_0_DATA_OUT_0_MAX_TENSOR_SIZE_DIM_1 = 16,
    parameter MHA_PL0_0_DATA_OUT_0_PARALLELISM_DIM_0 = PARALLELISM_DIM0,
    parameter MHA_PL0_0_DATA_OUT_0_PARALLELISM_DIM_1 = PARALLELISM_DIM1,
    parameter MHA_PL0_0_DATA_OUT_0_PRECISION_0 = PRECISION_0,
    parameter MHA_PL0_0_DATA_OUT_0_PRECISION_1 = PRECISION_1,
    parameter RESIDUAL_PL0_0_DATA_IN_0_PRECISION_0 = PRECISION_0,
    parameter RESIDUAL_PL0_0_DATA_IN_0_PRECISION_1 = PRECISION_1,
    parameter RESIDUAL_PL0_0_DATA_IN_0_TENSOR_SIZE_DIM_0 = 16,
    parameter RESIDUAL_PL0_0_DATA_IN_0_TENSOR_SIZE_DIM_1 = 16,
    parameter RESIDUAL_PL0_0_DATA_IN_0_TENSOR_SIZE_DIM_2 = 16,
    parameter RESIDUAL_PL0_0_DATA_IN_0_PARALLELISM_DIM_0 = PARALLELISM_DIM0,
    parameter RESIDUAL_PL0_0_DATA_IN_0_PARALLELISM_DIM_1 = PARALLELISM_DIM1,
    parameter RESIDUAL_PL0_0_DATA_IN_0_PARALLELISM_DIM_2 = 16,
    parameter RESIDUAL_PL0_0_DATA_IN_1_PRECISION_0 = PRECISION_0,
    parameter RESIDUAL_PL0_0_DATA_IN_1_PRECISION_1 = PRECISION_1,
    parameter RESIDUAL_PL0_0_DATA_IN_1_TENSOR_SIZE_DIM_0 = 16,
    parameter RESIDUAL_PL0_0_DATA_IN_1_TENSOR_SIZE_DIM_1 = 16,
    parameter RESIDUAL_PL0_0_DATA_IN_1_TENSOR_SIZE_DIM_2 = 16,
    parameter RESIDUAL_PL0_0_DATA_IN_1_PARALLELISM_DIM_0 = PARALLELISM_DIM0,
    parameter RESIDUAL_PL0_0_DATA_IN_1_PARALLELISM_DIM_1 = PARALLELISM_DIM1,
    parameter RESIDUAL_PL0_0_DATA_IN_1_PARALLELISM_DIM_2 = 16,
    parameter ADDER_FIFO_PL0_1_DATA_WIDTH = PRECISION_0,
    parameter ADDER_FIFO_PL0_1_DIM0 = PARALLELISM_DIM0,
    parameter ADDER_FIFO_PL0_1_DIM1 = PARALLELISM_DIM1,
    parameter ADDER_FIFO_PL0_1_FIFO_SIZE = 16,
    parameter LAYER_NORM_PL0_1_TOTAL_MAX_DIM0 = 16,
    parameter LAYER_NORM_PL0_1_TOTAL_MAX_DIM1 = 16,
    parameter LAYER_NORM_PL0_1_PARALLELISM_DIM0 = PARALLELISM_DIM0,
    parameter LAYER_NORM_PL0_1_PARALLELISM_DIM1 = PARALLELISM_DIM1,
    parameter LAYER_NORM_PL0_1_PRECISION_0 = PRECISION_0,
    parameter LAYER_NORM_PL0_1_PRECISION_1 = PRECISION_1,
    parameter LINEAR_PL0_0_WEIGHTS_PRE_TRANSPOSED = 16,
    parameter LINEAR_PL0_0_DATA_IN_0_PRECISION_0 = PRECISION_0,
    parameter LINEAR_PL0_0_DATA_IN_0_PRECISION_1 = PRECISION_1,
    parameter LINEAR_PL0_0_DATA_IN_0_MAX_TENSOR_SIZE_DIM_0 = 16,
    parameter LINEAR_PL0_0_DATA_IN_0_MAX_TENSOR_SIZE_DIM_1 = 16,
    parameter LINEAR_PL0_0_DATA_IN_0_TENSOR_SIZE_DIM_2 = 16,
    parameter LINEAR_PL0_0_DATA_IN_0_PARALLELISM_DIM_0 = PARALLELISM_DIM0,
    parameter LINEAR_PL0_0_DATA_IN_0_PARALLELISM_DIM_1 = PARALLELISM_DIM1,
    parameter LINEAR_PL0_0_DATA_IN_0_PARALLELISM_DIM_2 = 16,
    parameter LINEAR_PL0_0_WEIGHT_PRECISION_0 = PRECISION_0,
    parameter LINEAR_PL0_0_WEIGHT_PRECISION_1 = PRECISION_1,
    parameter LINEAR_PL0_0_WEIGHT_MAX_TENSOR_SIZE_DIM_0 = 16,
    parameter LINEAR_PL0_0_WEIGHT_MAX_TENSOR_SIZE_DIM_1 = 16,
    parameter LINEAR_PL0_0_WEIGHT_PARALLELISM_DIM_0 = PARALLELISM_DIM0,
    parameter LINEAR_PL0_0_WEIGHT_PARALLELISM_DIM_1 = PARALLELISM_DIM1,
    parameter LINEAR_PL0_0_DATA_OUT_0_PRECISION_0 = PRECISION_0,
    parameter LINEAR_PL0_0_DATA_OUT_0_PRECISION_1 = PRECISION_1,
    parameter LINEAR_PL0_0_DATA_OUT_0_MAX_TENSOR_SIZE_DIM_0 = 16,
    parameter LINEAR_PL0_0_DATA_OUT_0_MAX_TENSOR_SIZE_DIM_1 = 16,
    parameter LINEAR_PL0_0_DATA_OUT_0_MAX_TENSOR_SIZE_DIM_2 = 16,
    parameter LINEAR_PL0_0_DATA_OUT_0_PARALLELISM_DIM_0 = PARALLELISM_DIM0,
    parameter LINEAR_PL0_0_DATA_OUT_0_PARALLELISM_DIM_1 = PARALLELISM_DIM1,
    parameter LINEAR_PL0_0_DATA_OUT_0_PARALLELISM_DIM_2 = 16,
    parameter LINEAR_PL0_0_BIAS_PRECISION_0 = PRECISION_0,
    parameter LINEAR_PL0_0_BIAS_PRECISION_1 = PRECISION_1,
    parameter LINEAR_PL0_0_BIAS_MAX_TENSOR_SIZE_DIM_0 = 16,
    parameter LINEAR_PL0_0_BIAS_MAX_TENSOR_SIZE_DIM_1 = 16,
    parameter LINEAR_PL0_0_BIAS_PARALLELISM_DIM_0 = PARALLELISM_DIM0,
    parameter LINEAR_PL0_0_BIAS_PARALLELISM_DIM_1 = PARALLELISM_DIM1,
    parameter RELU_PL0_0_DATA_IN_0_PRECISION_0 = PRECISION_0,
    parameter RELU_PL0_0_DATA_IN_0_PRECISION_1 = PRECISION_1,
    parameter RELU_PL0_0_DATA_IN_0_TENSOR_SIZE_DIM_0 = 16,
    parameter RELU_PL0_0_DATA_IN_0_TENSOR_SIZE_DIM_1 = 16,
    parameter RELU_PL0_0_DATA_IN_0_PARALLELISM_DIM_0 = PARALLELISM_DIM0,
    parameter RELU_PL0_0_DATA_IN_0_PARALLELISM_DIM_1 = PARALLELISM_DIM1,
    parameter RELU_PL0_0_DATA_OUT_0_PRECISION_0 = PRECISION_0,
    parameter RELU_PL0_0_DATA_OUT_0_PRECISION_1 = PRECISION_1,
    parameter RELU_PL0_0_DATA_OUT_0_TENSOR_SIZE_DIM_0 = 16,
    parameter RELU_PL0_0_DATA_OUT_0_TENSOR_SIZE_DIM_1 = 16,
    parameter RELU_PL0_0_DATA_OUT_0_PARALLELISM_DIM_0 = PARALLELISM_DIM0,
    parameter RELU_PL0_0_DATA_OUT_0_PARALLELISM_DIM_1 = PARALLELISM_DIM1,
    parameter LINEAR_PL0_1_WEIGHTS_PRE_TRANSPOSED = 16,
    parameter LINEAR_PL0_1_DATA_IN_0_PRECISION_0 = PRECISION_0,
    parameter LINEAR_PL0_1_DATA_IN_0_PRECISION_1 = PRECISION_1,
    parameter LINEAR_PL0_1_DATA_IN_0_MAX_TENSOR_SIZE_DIM_0 = 16,
    parameter LINEAR_PL0_1_DATA_IN_0_MAX_TENSOR_SIZE_DIM_1 = 16,
    parameter LINEAR_PL0_1_DATA_IN_0_TENSOR_SIZE_DIM_2 = 16,
    parameter LINEAR_PL0_1_DATA_IN_0_PARALLELISM_DIM_0 = PARALLELISM_DIM0,
    parameter LINEAR_PL0_1_DATA_IN_0_PARALLELISM_DIM_1 = PARALLELISM_DIM1,
    parameter LINEAR_PL0_1_DATA_IN_0_PARALLELISM_DIM_2 = 16,
    parameter LINEAR_PL0_1_WEIGHT_PRECISION_0 = PRECISION_0,
    parameter LINEAR_PL0_1_WEIGHT_PRECISION_1 = PRECISION_1,
    parameter LINEAR_PL0_1_WEIGHT_MAX_TENSOR_SIZE_DIM_0 = 16,
    parameter LINEAR_PL0_1_WEIGHT_MAX_TENSOR_SIZE_DIM_1 = 16,
    parameter LINEAR_PL0_1_WEIGHT_PARALLELISM_DIM_0 = PARALLELISM_DIM0,
    parameter LINEAR_PL0_1_WEIGHT_PARALLELISM_DIM_1 = PARALLELISM_DIM1,
    parameter LINEAR_PL0_1_DATA_OUT_0_PRECISION_0 = PRECISION_0,
    parameter LINEAR_PL0_1_DATA_OUT_0_PRECISION_1 = PRECISION_1,
    parameter LINEAR_PL0_1_DATA_OUT_0_MAX_TENSOR_SIZE_DIM_0 = 16,
    parameter LINEAR_PL0_1_DATA_OUT_0_MAX_TENSOR_SIZE_DIM_1 = 16,
    parameter LINEAR_PL0_1_DATA_OUT_0_MAX_TENSOR_SIZE_DIM_2 = 16,
    parameter LINEAR_PL0_1_DATA_OUT_0_PARALLELISM_DIM_0 = PARALLELISM_DIM0,
    parameter LINEAR_PL0_1_DATA_OUT_0_PARALLELISM_DIM_1 = PARALLELISM_DIM1,
    parameter LINEAR_PL0_1_DATA_OUT_0_PARALLELISM_DIM_2 = 16,
    parameter LINEAR_PL0_1_BIAS_PRECISION_0 = PRECISION_0,
    parameter LINEAR_PL0_1_BIAS_PRECISION_1 = PRECISION_1,
    parameter LINEAR_PL0_1_BIAS_MAX_TENSOR_SIZE_DIM_0 = 16,
    parameter LINEAR_PL0_1_BIAS_MAX_TENSOR_SIZE_DIM_1 = 16,
    parameter LINEAR_PL0_1_BIAS_PARALLELISM_DIM_0 = PARALLELISM_DIM0,
    parameter LINEAR_PL0_1_BIAS_PARALLELISM_DIM_1 = PARALLELISM_DIM1,
    parameter RESIDUAL_PL0_1_DATA_IN_0_PRECISION_0 = PRECISION_0,
    parameter RESIDUAL_PL0_1_DATA_IN_0_PRECISION_1 = PRECISION_1,
    parameter RESIDUAL_PL0_1_DATA_IN_0_TENSOR_SIZE_DIM_0 = 16,
    parameter RESIDUAL_PL0_1_DATA_IN_0_TENSOR_SIZE_DIM_1 = 16,
    parameter RESIDUAL_PL0_1_DATA_IN_0_TENSOR_SIZE_DIM_2 = 16,
    parameter RESIDUAL_PL0_1_DATA_IN_0_PARALLELISM_DIM_0 = PARALLELISM_DIM0,
    parameter RESIDUAL_PL0_1_DATA_IN_0_PARALLELISM_DIM_1 = PARALLELISM_DIM1,
    parameter RESIDUAL_PL0_1_DATA_IN_0_PARALLELISM_DIM_2 = 16,
    parameter RESIDUAL_PL0_1_DATA_IN_1_PRECISION_0 = PRECISION_0,
    parameter RESIDUAL_PL0_1_DATA_IN_1_PRECISION_1 = PRECISION_1,
    parameter RESIDUAL_PL0_1_DATA_IN_1_TENSOR_SIZE_DIM_0 = 16,
    parameter RESIDUAL_PL0_1_DATA_IN_1_TENSOR_SIZE_DIM_1 = 16,
    parameter RESIDUAL_PL0_1_DATA_IN_1_TENSOR_SIZE_DIM_2 = 16,
    parameter RESIDUAL_PL0_1_DATA_IN_1_PARALLELISM_DIM_0 = PARALLELISM_DIM0,
    parameter RESIDUAL_PL0_1_DATA_IN_1_PARALLELISM_DIM_1 = PARALLELISM_DIM1,
    parameter RESIDUAL_PL0_1_DATA_IN_1_PARALLELISM_DIM_2 = 16,
    parameter ROLL_PL0_0_ROLL_MAX_DISTANCE = 16,
    parameter ROLL_PL0_0_MAX_BUFFER_SIZE = 16,
    parameter ROLL_PL0_0_DATA_IN_0_PARALLELISM_DIM_0 = PARALLELISM_DIM0,
    parameter ROLL_PL0_0_DATA_IN_0_PARALLELISM_DIM_1 = PARALLELISM_DIM1,
    parameter ROLL_PL0_0_DATA_IN_0_PRECISION_0 = PRECISION_0,
    parameter ROLL_PL0_0_DATA_IN_0_PRECISION_1 = PRECISION_1,
    parameter ROLL_PL0_0_DATA_OUT_0_PRECISION_0 = PRECISION_0,
    parameter ROLL_PL0_0_DATA_OUT_0_PARALLELISM_DIM_0 = PARALLELISM_DIM0,
    parameter ROLL_PL0_0_DATA_OUT_0_PARALLELISM_DIM_1 = PARALLELISM_DIM1,

    parameter ISQRT_LUT_MEMFILE = "/workspace/src/mase_components/attention/test/build/swintop_simple/mem/lutmem-default.mem",
    //parameter ISQRT_LUT_MEMFILE = 0,
    parameter ISQRT_LUT_POW = 4

 )
(
    input logic clk,
    input logic rst,
    //add weight buffers as they are needed

    input logic [PRECISION_0-1:0] input_buffer_data_in_0 [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic input_buffer_data_in_0_valid,
    input logic input_buffer_data_in_0_ready,

    input logic [PRECISION_0-1:0] mha_pl0_0_weight_query [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic mha_pl0_0_weight_query_valid,
    output logic mha_pl0_0_weight_query_ready,
    input logic [PRECISION_0-1:0] mha_pl0_0_bias_con [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic mha_pl0_0_bias_con_valid,
    output logic mha_pl0_0_bias_con_ready,
    input logic [PRECISION_0-1:0] mha_pl0_0_bias_pos [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic mha_pl0_0_bias_pos_valid,
    output logic mha_pl0_0_bias_pos_ready,
    input logic [PRECISION_0-1:0] mha_pl0_0_weight_key [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic mha_pl0_0_weight_key_valid,
    output logic mha_pl0_0_weight_key_ready,
    input logic [PRECISION_0-1:0] mha_pl0_0_weight_value [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic mha_pl0_0_weight_value_valid,
    output logic mha_pl0_0_weight_value_ready,
    input logic [PRECISION_0-1:0] mha_pl0_0_pos_embed [PARALLELISM_DIM0*PARALLELISM_DIM1*PARALLELISM_DIM1-1:0],
    input logic mha_pl0_0_pos_embed_valid,
    output logic mha_pl0_0_pos_embed_ready,
    input logic [PRECISION_0-1:0] mha_pl0_0_weight_out [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic mha_pl0_0_weight_out_valid,
    output logic mha_pl0_0_weight_out_ready,
    input logic [PRECISION_0-1:0] mha_pl0_0_bias_out [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic mha_pl0_0_bias_out_valid,
    output logic mha_pl0_0_bias_out_ready,
    input logic [PRECISION_0-1:0] linear_pl0_0_weight [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic linear_pl0_0_weight_valid,
    output logic linear_pl0_0_weight_ready,
    input logic [PRECISION_0-1:0] linear_pl0_0_bias [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic linear_pl0_0_bias_valid,
    output logic linear_pl0_0_bias_ready,
    input logic [PRECISION_0-1:0] linear_pl0_1_weight [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic linear_pl0_1_weight_valid,
    output logic linear_pl0_1_weight_ready,
    input logic [PRECISION_0-1:0] linear_pl0_1_bias [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic linear_pl0_1_bias_valid,
    output logic linear_pl0_1_bias_ready,
    input logic [PRECISION_0-1:0] mha_pl1_0_weight_query [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic mha_pl1_0_weight_query_valid,
    output logic mha_pl1_0_weight_query_ready,
    input logic [PRECISION_0-1:0] mha_pl1_0_bias_con [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic mha_pl1_0_bias_con_valid,
    output logic mha_pl1_0_bias_con_ready,
    input logic [PRECISION_0-1:0] mha_pl1_0_bias_pos [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic mha_pl1_0_bias_pos_valid,
    output logic mha_pl1_0_bias_pos_ready,
    input logic [PRECISION_0-1:0] mha_pl1_0_weight_key [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic mha_pl1_0_weight_key_valid,
    output logic mha_pl1_0_weight_key_ready,
    input logic [PRECISION_0-1:0] mha_pl1_0_weight_value [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic mha_pl1_0_weight_value_valid,
    output logic mha_pl1_0_weight_value_ready,
    input logic [PRECISION_0-1:0] mha_pl1_0_pos_embed [PARALLELISM_DIM0*PARALLELISM_DIM1*PARALLELISM_DIM1-1:0],
    input logic mha_pl1_0_pos_embed_valid,
    output logic mha_pl1_0_pos_embed_ready,
    input logic [PRECISION_0-1:0] mha_pl1_0_weight_out [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic mha_pl1_0_weight_out_valid,
    output logic mha_pl1_0_weight_out_ready,
    input logic [PRECISION_0-1:0] mha_pl1_0_bias_out [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic mha_pl1_0_bias_out_valid,
    output logic mha_pl1_0_bias_out_ready,
    input logic [PRECISION_0-1:0] linear_pl1_0_weight [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic linear_pl1_0_weight_valid,
    output logic linear_pl1_0_weight_ready,
    input logic [PRECISION_0-1:0] linear_pl1_0_bias [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic linear_pl1_0_bias_valid,
    output logic linear_pl1_0_bias_ready,
    input logic [PRECISION_0-1:0] linear_pl1_1_weight [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic linear_pl1_1_weight_valid,
    output logic linear_pl1_1_weight_ready,
    input logic [PRECISION_0-1:0] linear_pl1_1_bias [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic linear_pl1_1_bias_valid,
    output logic linear_pl1_1_bias_ready,
    input logic [PRECISION_0-1:0] linear_pl1_2_weight [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic linear_pl1_2_weight_valid,
    output logic linear_pl1_2_weight_ready,
    input logic [PRECISION_0-1:0] linear_pl1_2_bias [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0],
    input logic linear_pl1_2_bias_valid,
    output logic linear_pl1_2_bias_ready



);

logic [9:0]layer_norm_pl0_0_n_iters;
logic [39:0]layer_norm_pl0_0_inv_numvalues_0;
logic [39:0]layer_norm_pl0_0_inv_numvalues_1;
logic [PRECISION_0-1:0] layer_norm_pl0_0_in_data [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic layer_norm_pl0_0_in_valid;
logic layer_norm_pl0_0_in_ready;
logic [PRECISION_0-1:0] layer_norm_pl0_0_out_data [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic layer_norm_pl0_0_out_valid;
logic layer_norm_pl0_0_out_ready;
logic [9:0]mha_pl0_0_data_in_0_depth_dim_1;
logic [9:0]mha_pl0_0_weight_tensor_size_dim0;
logic [9:0]mha_pl0_0_weight_depth_dim_0;
logic [9:0]mha_pl0_0_weight_depth_dim_1;
logic [9:0]mha_pl0_0_weight_depth_mult;
logic [9:0]mha_pl0_0_block_per_head;
logic [9:0]mha_pl0_0_q_depth_dim_0;
logic [9:0]mha_pl0_0_q_depth_dim_1;
logic [9:0]mha_pl0_0_q_depth_mult;
logic [9:0]mha_pl0_0_weight_out_depth_dim_1;
logic [PRECISION_0-1:0] mha_pl0_0_data_in_0 [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic mha_pl0_0_data_in_0_valid;
logic mha_pl0_0_data_in_0_ready;
logic [PRECISION_0-1:0] mha_pl0_0_data_out_0 [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic mha_pl0_0_data_out_0_valid;
logic mha_pl0_0_data_out_0_ready;
logic [PRECISION_0-1:0] adder_fifo_pl0_0_in_data [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic adder_fifo_pl0_0_in_valid;
logic adder_fifo_pl0_0_in_ready;
logic [PRECISION_0-1:0] adder_fifo_pl0_0_out_data [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic adder_fifo_pl0_0_out_valid;
logic adder_fifo_pl0_0_out_ready;
logic [PRECISION_0-1:0] residual_pl0_0_data_in_0 [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic residual_pl0_0_data_in_0_valid;
logic residual_pl0_0_data_in_0_ready;
logic [PRECISION_0-1:0] residual_pl0_0_data_in_1 [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic residual_pl0_0_data_in_1_valid;
logic residual_pl0_0_data_in_1_ready;
logic [PRECISION_0-1:0] residual_pl0_0_data_out_0 [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic residual_pl0_0_data_out_0_valid;
logic residual_pl0_0_data_out_0_ready;
logic [9:0]layer_norm_pl0_1_n_iters;
logic [39:0]layer_norm_pl0_1_inv_numvalues_0;
logic [39:0]layer_norm_pl0_1_inv_numvalues_1;
logic [PRECISION_0-1:0] layer_norm_pl0_1_in_data [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic layer_norm_pl0_1_in_valid;
logic layer_norm_pl0_1_in_ready;
logic [PRECISION_0-1:0] layer_norm_pl0_1_out_data [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic layer_norm_pl0_1_out_valid;
logic layer_norm_pl0_1_out_ready;
logic [9:0]linear_pl0_0_data_in_0_depth_dim1;
logic [9:0]linear_pl0_0_weight_tensor_size_dim0;
logic [9:0]linear_pl0_0_weight_depth_dim0;
logic [9:0]linear_pl0_0_weight_depth_dim1;
logic [9:0]linear_pl0_0_weight_depth_mult;
logic [PRECISION_0-1:0] linear_pl0_0_data_in_0 [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic linear_pl0_0_data_in_0_valid;
logic linear_pl0_0_data_in_0_ready;
logic [PRECISION_0-1:0] linear_pl0_0_data_out_0 [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic linear_pl0_0_data_out_0_valid;
logic linear_pl0_0_data_out_0_ready;
logic relu_pl0_0_rst;
logic relu_pl0_0_clk;
logic [PRECISION_0-1:0] relu_pl0_0_data_in_0 [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic [PRECISION_0-1:0] relu_pl0_0_data_out_0 [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic relu_pl0_0_data_in_0_valid;
logic relu_pl0_0_data_in_0_ready;
logic relu_pl0_0_data_out_0_valid;
logic relu_pl0_0_data_out_0_ready;
logic [9:0]linear_pl0_1_data_in_0_depth_dim1;
logic [9:0]linear_pl0_1_weight_tensor_size_dim0;
logic [9:0]linear_pl0_1_weight_depth_dim0;
logic [9:0]linear_pl0_1_weight_depth_dim1;
logic [9:0]linear_pl0_1_weight_depth_mult;
logic [PRECISION_0-1:0] linear_pl0_1_data_in_0 [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic linear_pl0_1_data_in_0_valid;
logic linear_pl0_1_data_in_0_ready;
logic [PRECISION_0-1:0] linear_pl0_1_data_out_0 [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic linear_pl0_1_data_out_0_valid;
logic linear_pl0_1_data_out_0_ready;
logic [9:0] roll_pl0_0_roll_distance;
logic [9:0] roll_pl0_0_buffer_size;
logic [PRECISION_0-1:0] roll_pl0_0_data_in_0 [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic roll_pl0_0_data_in_0_valid;
logic roll_pl0_0_data_in_0_ready;
logic [PRECISION_0-1:0] roll_pl0_0_data_out_0 [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic roll_pl0_0_data_out_0_valid;
logic roll_pl0_0_data_out_0_ready;
logic [9:0] roll_pl0_1_roll_distance;
logic [9:0] roll_pl0_1_buffer_size;
logic [PRECISION_0-1:0] roll_pl0_1_data_in_0 [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic roll_pl0_1_data_in_0_valid;
logic roll_pl0_1_data_in_0_ready;
logic [PRECISION_0-1:0] roll_pl0_1_data_out_0 [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic roll_pl0_1_data_out_0_valid;
logic roll_pl0_1_data_out_0_ready;
logic [PRECISION_0-1:0] adder_fifo_pl0_1_in_data [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic adder_fifo_pl0_1_in_valid;
logic adder_fifo_pl0_1_in_ready;
logic [PRECISION_0-1:0] adder_fifo_pl0_1_out_data [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic adder_fifo_pl0_1_out_valid;
logic adder_fifo_pl0_1_out_ready;
logic residual_pl0_1_clk;
logic residual_pl0_1_rst;
logic [PRECISION_0-1:0] residual_pl0_1_data_in_0 [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic residual_pl0_1_data_in_0_valid;
logic residual_pl0_1_data_in_0_ready;
logic [PRECISION_0-1:0] residual_pl0_1_data_in_1 [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic residual_pl0_1_data_in_1_valid;
logic residual_pl0_1_data_in_1_ready;
logic [PRECISION_0-1:0] residual_pl0_1_data_out_0 [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic residual_pl0_1_data_out_0_valid;
logic residual_pl0_1_data_out_0_ready;

logic load_input_seq;
logic input_mux_ctrl;
logic output_mux_ctrl;

logic [PRECISION_0-1:0] input_mux_data [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic input_mux_ready;
logic input_mux_valid;

logic [PRECISION_0-1:0] output_mux_data [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic output_mux_ready;
logic output_mux_valid;

logic [PRECISION_0-1:0] input_signal [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic input_signal_ready;
logic input_signal_valid;

logic [PRECISION_0-1:0] input_buffer_data_out_0 [PARALLELISM_DIM0*PARALLELISM_DIM1-1:0];
logic input_buffer_data_out_0_ready;
logic input_buffer_data_out_0_valid;

logic [14:0] max_input_counter;
logic [14:0] max_output_counter;

swin_controller control(
    .layer_norm_pl0_0_n_iters(layer_norm_pl0_0_n_iters),
    .layer_norm_pl0_0_inv_numvalues_0(layer_norm_pl0_0_inv_numvalues_0),
    .layer_norm_pl0_0_inv_numvalues_1(layer_norm_pl0_0_inv_numvalues_1),
    .mha_pl0_0_data_in_0_depth_dim_1(mha_pl0_0_data_in_0_depth_dim_1),
    .mha_pl0_0_weight_tensor_size_dim0(mha_pl0_0_weight_tensor_size_dim0),
    .mha_pl0_0_weight_depth_dim_0(mha_pl0_0_weight_depth_dim_0),
    .mha_pl0_0_weight_depth_dim_1(mha_pl0_0_weight_depth_dim_1),
    .mha_pl0_0_weight_depth_mult(mha_pl0_0_weight_depth_mult),
    .mha_pl0_0_block_per_head(mha_pl0_0_block_per_head),
    .mha_pl0_0_q_depth_dim_0(mha_pl0_0_q_depth_dim_0),
    .mha_pl0_0_q_depth_dim_1(mha_pl0_0_q_depth_dim_1),
    .mha_pl0_0_q_depth_mult(mha_pl0_0_q_depth_mult),
    .mha_pl0_0_weight_out_depth_dim_1(mha_pl0_0_weight_out_depth_dim_1),
    .layer_norm_pl0_1_n_iters(layer_norm_pl0_1_n_iters),
    .layer_norm_pl0_1_inv_numvalues_0(layer_norm_pl0_1_inv_numvalues_0),
    .layer_norm_pl0_1_inv_numvalues_1(layer_norm_pl0_1_inv_numvalues_1),
    .linear_pl0_0_data_in_0_depth_dim1(linear_pl0_0_data_in_0_depth_dim1),
    .linear_pl0_0_weight_tensor_size_dim0(linear_pl0_0_weight_tensor_size_dim0),
    .linear_pl0_0_weight_depth_dim0(linear_pl0_0_weight_depth_dim0),
    .linear_pl0_0_weight_depth_dim1(linear_pl0_0_weight_depth_dim1),
    .linear_pl0_0_weight_depth_mult(linear_pl0_0_weight_depth_mult),
    .linear_pl0_1_data_in_0_depth_dim1(linear_pl0_1_data_in_0_depth_dim1),
    .linear_pl0_1_weight_tensor_size_dim0(linear_pl0_1_weight_tensor_size_dim0),
    .linear_pl0_1_weight_depth_dim0(linear_pl0_1_weight_depth_dim0),
    .linear_pl0_1_weight_depth_dim1(linear_pl0_1_weight_depth_dim1),
    .linear_pl0_1_weight_depth_mult(linear_pl0_1_weight_depth_mult),
    .roll_pl0_0_roll_distance(roll_pl0_0_roll_distance),
    .roll_pl0_0_buffer_size(roll_pl0_0_buffer_size),
    .roll_pl0_1_roll_distance(roll_pl0_1_roll_distance),
    .roll_pl0_1_buffer_size(roll_pl0_1_buffer_size),
    .done(done),
    .load_input_seq(load_input_seq),
    .input_mux_ctrl(input_mux_ctrl),
    .output_mux_ctrl(output_mux_ctrl),
    .max_input_counter(max_input_counter),
    .max_output_counter(max_output_counter)
);

split2 split_pl0_0(
    .data_in_valid(input_signal_valid),
    .data_in_ready(input_signal_ready),
    .data_out_valid({layer_norm_pl0_0_in_valid, adder_fifo_pl0_0_in_valid}),
    .data_out_ready({layer_norm_pl0_0_in_ready, adder_fifo_pl0_0_in_ready})
);


matrix_fifo #(
    .DATA_WIDTH(PRECISION_0),
    .DIM0(PARALLELISM_DIM0),
    .DIM1(PARALLELISM_DIM0),
    .FIFO_SIZE(256)
) input_buffer
(
    .clk(clk),
    .rst(rst),
    .in_data(input_buffer_data_in_0),
    .in_valid(input_buffer_data_in_0_valid),
    .in_ready(input_buffer_data_in_0_ready),
    .out_data(input_buffer_data_out_0),
    .out_valid(input_buffer_data_out_0_valid),
    .out_ready(input_buffer_data_out_0_ready)
);



matrix_fifo #(
    .DATA_WIDTH(ADDER_FIFO_PL0_0_DATA_WIDTH),
    .DIM0(ADDER_FIFO_PL0_0_DIM0),
    .DIM1(ADDER_FIFO_PL0_0_DIM1),
    .FIFO_SIZE(ADDER_FIFO_PL0_0_FIFO_SIZE)
) adder_fifo_pl0_0
(
    .clk(clk),
    .rst(rst),
    .in_data(adder_fifo_pl0_0_in_data),
    .in_valid(adder_fifo_pl0_0_in_valid),
    .in_ready(adder_fifo_pl0_0_in_ready),
    .out_data(adder_fifo_pl0_0_out_data),
    .out_valid(adder_fifo_pl0_0_out_valid),
    .out_ready(adder_fifo_pl0_0_out_ready)
);

swin_counter counter_0
(

    .clk(clk),
    .rst(rst),

    .max_input_counter(counter_0_max_input_counter),
    .max_output_counter(counter_0_max_output_counter),

    .data_in_0_valid(layer_norm_pl0_0_in_valid),
    .data_in_0_ready(layer_norm_pl0_0_in_ready),
    .data_out_0_valid(layer_norm_pl0_0_out_valid),
    .data_out_0_ready(layer_norm_pl0_0_out_ready)
);

//pipeline 0 - layer norm 0
group_norm_2d_programmable #(
    .TOTAL_MAX_DIM0  (LAYER_NORM_PL0_0_TOTAL_MAX_DIM0),
    .TOTAL_MAX_DIM1  (LAYER_NORM_PL0_0_TOTAL_MAX_DIM1),
    .COMPUTE_DIM0  (LAYER_NORM_PL0_0_PARALLELISM_DIM0),
    .COMPUTE_DIM1  (LAYER_NORM_PL0_0_PARALLELISM_DIM0),
    .GROUP_CHANNELS  (1),
    .IN_WIDTH  (LAYER_NORM_PL0_0_PRECISION_0),
    .IN_FRAC_WIDTH  (LAYER_NORM_PL0_0_PRECISION_1),
    .OUT_WIDTH  (LAYER_NORM_PL0_0_PRECISION_0),
    .OUT_FRAC_WIDTH  (LAYER_NORM_PL0_0_PRECISION_1),
    .ISQRT_LUT_MEMFILE(ISQRT_LUT_MEMFILE),
    .ISQRT_LUT_POW (ISQRT_LUT_POW)  
) layer_norm_pl0_0
(
    .clk(clk),
    .rst(rst),

    .n_iters (layer_norm_pl0_0_n_iters), 
    .inv_numvalues_0 (layer_norm_pl0_0_inv_numvalues_0),
    .inv_numvalues_1 (layer_norm_pl0_0_inv_numvalues_1),
    .in_data (layer_norm_pl0_0_in_data), 
    .in_valid (layer_norm_pl0_0_in_valid),
    .in_ready (layer_norm_pl0_0_in_ready),
    .out_data (layer_norm_pl0_0_out_data) ,
    .out_valid (layer_norm_pl0_0_out_valid),
    .out_ready (layer_norm_pl0_0_out_ready)

);

//pipeline 0 - mha 0 

fixed_swin_attention_programmable #(
    .NUM_HEADS(1),
    .HAS_BIAS(0),
    .DATA_IN_0_MAX_TENSOR_SIZE_DIM_0(MHA_PL0_0_DATA_IN_0_MAX_TENSOR_SIZE_DIM_0),
    .DATA_IN_0_MAX_TENSOR_SIZE_DIM_1(MHA_PL0_0_DATA_IN_0_MAX_TENSOR_SIZE_DIM_1),
    .DATA_IN_0_PARALLELISM_DIM_0(MHA_PL0_0_DATA_IN_0_PARALLELISM_DIM_0),
    .DATA_IN_0_PARALLELISM_DIM_1(MHA_PL0_0_DATA_IN_0_PARALLELISM_DIM_1),
    .DATA_IN_0_PRECISION_0(MHA_PL0_0_DATA_IN_0_PRECISION_0),
    .DATA_IN_0_PRECISION_1(MHA_PL0_0_DATA_IN_0_PRECISION_1),
    .WEIGHT_MAX_TENSOR_SIZE_DIM_0(MHA_PL0_0_WEIGHT_MAX_TENSOR_SIZE_DIM_0),
    .WEIGHT_MAX_TENSOR_SIZE_DIM_1(MHA_PL0_0_WEIGHT_MAX_TENSOR_SIZE_DIM_1),
    .WEIGHT_PARALLELISM_DIM_0(MHA_PL0_0_WEIGHT_PARALLELISM_DIM_0),
    .WEIGHT_PARALLELISM_DIM_1(MHA_PL0_0_WEIGHT_PARALLELISM_DIM_1),
    .WEIGHT_PRECISION_0(MHA_PL0_0_WEIGHT_PRECISION_0),
    .WEIGHT_PRECISION_1(MHA_PL0_0_WEIGHT_PRECISION_1),
    .BIAS_MAX_TENSOR_SIZE_DIM_0(MHA_PL0_0_BIAS_MAX_TENSOR_SIZE_DIM_0),
    .BIAS_MAX_TENSOR_SIZE_DIM_1(MHA_PL0_0_BIAS_MAX_TENSOR_SIZE_DIM_1),
    .BIAS_PARALLELISM_DIM_0(MHA_PL0_0_BIAS_PARALLELISM_DIM_0),
    .BIAS_PARALLELISM_DIM_1(MHA_PL0_0_BIAS_PARALLELISM_DIM_1),
    .BIAS_PRECISION_0(MHA_PL0_0_BIAS_PRECISION_0),
    .BIAS_PRECISION_1(MHA_PL0_0_BIAS_PRECISION_1),
    .DATA_OUT_0_MAX_TENSOR_SIZE_DIM_0(MHA_PL0_0_DATA_OUT_0_MAX_TENSOR_SIZE_DIM_0),
    .DATA_OUT_0_MAX_TENSOR_SIZE_DIM_1(MHA_PL0_0_DATA_OUT_0_MAX_TENSOR_SIZE_DIM_1),
    .DATA_OUT_0_PARALLELISM_DIM_0(MHA_PL0_0_DATA_OUT_0_PARALLELISM_DIM_0),
    .DATA_OUT_0_PARALLELISM_DIM_1(MHA_PL0_0_DATA_OUT_0_PARALLELISM_DIM_1),
    .DATA_OUT_0_PRECISION_0(MHA_PL0_0_DATA_OUT_0_PRECISION_0),
    .DATA_OUT_0_PRECISION_1(MHA_PL0_0_DATA_OUT_0_PRECISION_1)
) mha_pl0_0(
    .clk(clk),
    .rst(rst),
    .data_in_0_depth_dim_1(mha_pl0_0_data_in_0_depth_dim_1),
    .weight_tensor_size_dim0(mha_pl0_0_weight_tensor_size_dim0),
    .weight_depth_dim_0(mha_pl0_0_weight_depth_dim_0),
    .weight_depth_dim_1(mha_pl0_0_weight_depth_dim_1),  
    .weight_depth_mult(mha_pl0_0_weight_depth_mult),
    .block_per_head(mha_pl0_0_block_per_head),
    .q_depth_dim_0(mha_pl0_0_q_depth_dim_0),
    .q_depth_dim_1(mha_pl0_0_q_depth_dim_1),
    .q_depth_mult(mha_pl0_0_q_depth_mult),
    .weight_out_depth_dim_1(mha_pl0_0_weight_out_depth_dim_1),
    .data_in_0(mha_pl0_0_data_in_0),
    .data_in_0_valid(mha_pl0_0_data_in_0_valid),
    .data_in_0_ready(mha_pl0_0_data_in_0_ready),
    .weight_query(mha_pl0_0_weight_query),
    .weight_query_valid(mha_pl0_0_weight_query_valid),
    .weight_query_ready(mha_pl0_0_weight_query_ready), 
    .bias_con(mha_pl0_0_bias_con),
    .bias_con_valid(mha_pl0_0_bias_con_valid),
    .bias_con_ready(mha_pl0_0_bias_con_ready),
    .bias_pos(mha_pl0_0_bias_pos),
    .bias_pos_valid(mha_pl0_0_bias_pos_valid),
    .bias_pos_ready(mha_pl0_0_bias_pos_ready),
    .weight_key(mha_pl0_0_weight_key),
    .weight_key_valid(mha_pl0_0_weight_key_valid),
    .weight_key_ready(mha_pl0_0_weight_key_ready),
    .weight_value(mha_pl0_0_weight_value),
    .weight_value_valid(mha_pl0_0_weight_value_valid),
    .weight_value_ready(mha_pl0_0_weight_value_ready),
    .pos_embed(mha_pl0_0_pos_embed),
    .pos_embed_valid(mha_pl0_0_pos_embed_valid),
    .pos_embed_ready(mha_pl0_0_pos_embed_ready),
    .weight_out(mha_pl0_0_weight_out),
    .weight_out_valid(mha_pl0_0_weight_out_valid),
    .weight_out_ready(mha_pl0_0_weight_out_ready),
    .bias_out(mha_pl0_0_bias_out),
    .bias_out_valid(mha_pl0_0_bias_out_valid),
    .bias_out_ready(mha_pl0_0_bias_out_ready),
    .data_out_0(mha_pl0_0_data_out_0),
    .data_out_0_valid(mha_pl0_0_data_out_0_valid),
    .data_out_0_ready(mha_pl0_0_data_out_0_ready)
);


fixed_adder #(
    .DATA_IN_0_PRECISION_0(RESIDUAL_PL0_0_DATA_IN_0_PRECISION_0),
    .DATA_IN_0_PRECISION_1(RESIDUAL_PL0_0_DATA_IN_0_PRECISION_1),
    .DATA_IN_0_TENSOR_SIZE_DIM_0(RESIDUAL_PL0_0_DATA_IN_0_TENSOR_SIZE_DIM_0),
    .DATA_IN_0_TENSOR_SIZE_DIM_1(RESIDUAL_PL0_0_DATA_IN_0_TENSOR_SIZE_DIM_1),
    .DATA_IN_0_TENSOR_SIZE_DIM_2(RESIDUAL_PL0_0_DATA_IN_0_TENSOR_SIZE_DIM_2),
    .DATA_IN_0_PARALLELISM_DIM_0(RESIDUAL_PL0_0_DATA_IN_0_PARALLELISM_DIM_0),
    .DATA_IN_0_PARALLELISM_DIM_1(RESIDUAL_PL0_0_DATA_IN_0_PARALLELISM_DIM_1),
    .DATA_IN_0_PARALLELISM_DIM_2(RESIDUAL_PL0_0_DATA_IN_0_PARALLELISM_DIM_2),
    .DATA_IN_1_PRECISION_0(RESIDUAL_PL0_0_DATA_IN_1_PRECISION_0),
    .DATA_IN_1_PRECISION_1(RESIDUAL_PL0_0_DATA_IN_1_PRECISION_1),
    .DATA_IN_1_TENSOR_SIZE_DIM_0(RESIDUAL_PL0_0_DATA_IN_1_TENSOR_SIZE_DIM_0),
    .DATA_IN_1_TENSOR_SIZE_DIM_1(RESIDUAL_PL0_0_DATA_IN_1_TENSOR_SIZE_DIM_1),
    .DATA_IN_1_TENSOR_SIZE_DIM_2(RESIDUAL_PL0_0_DATA_IN_1_TENSOR_SIZE_DIM_2),
    .DATA_IN_1_PARALLELISM_DIM_0(RESIDUAL_PL0_0_DATA_IN_1_PARALLELISM_DIM_0),
    .DATA_IN_1_PARALLELISM_DIM_1(RESIDUAL_PL0_0_DATA_IN_1_PARALLELISM_DIM_1),
    .DATA_IN_1_PARALLELISM_DIM_2(RESIDUAL_PL0_0_DATA_IN_1_PARALLELISM_DIM_2)
) residual_pl0_0
(
    .clk(clk),
    .rst(rst),
    .data_in_0(residual_pl0_0_data_in_0),
    .data_in_0_valid(residual_pl0_0_data_in_0_valid),
    .data_in_0_ready(residual_pl0_0_data_in_0_ready),
    .data_in_1(residual_pl0_0_data_in_1),
    .data_in_1_valid(residual_pl0_0_data_in_1_valid),
    .data_in_1_ready(residual_pl0_0_data_in_1_ready),
    .data_out_0(residual_pl0_0_data_out_0),
    .data_out_0_valid(residual_pl0_0_data_out_0_valid),
    .data_out_0_ready(residual_pl0_0_data_out_0_ready)
);

split2 split_pl0_1(
    .data_in_valid(residual_pl0_0_data_out_0_valid),
    .data_in_ready(residual_pl0_0_data_out_0_ready),
    .data_out_valid({layer_norm_pl0_1_in_valid, adder_fifo_pl0_1_in_valid}),
    .data_out_ready({layer_norm_pl0_1_in_ready, adder_fifo_pl0_1_in_ready})
);

matrix_fifo #(
    .DATA_WIDTH(ADDER_FIFO_PL0_1_DATA_WIDTH),
    .DIM0(ADDER_FIFO_PL0_1_DIM0),
    .DIM1(ADDER_FIFO_PL0_1_DIM1),
    .FIFO_SIZE(ADDER_FIFO_PL0_1_FIFO_SIZE)
) adder_fifo_pl0_1
(
    .clk(clk),
    .rst(rst),
    .in_data(adder_fifo_pl0_1_in_data),
    .in_valid(adder_fifo_pl0_1_in_valid),
    .in_ready(adder_fifo_pl0_1_in_ready),
    .out_data(adder_fifo_pl0_1_out_data),
    .out_valid(adder_fifo_pl0_1_out_valid),
    .out_ready(adder_fifo_pl0_1_out_ready)
);


group_norm_2d_programmable #(
    .TOTAL_MAX_DIM0  (LAYER_NORM_PL0_1_TOTAL_MAX_DIM0),
    .TOTAL_MAX_DIM1  (LAYER_NORM_PL0_1_TOTAL_MAX_DIM1),
    .COMPUTE_DIM0  (LAYER_NORM_PL0_1_PARALLELISM_DIM0),
    .COMPUTE_DIM1  (LAYER_NORM_PL0_1_PARALLELISM_DIM0),
    .GROUP_CHANNELS  (1),
    .IN_WIDTH  (LAYER_NORM_PL0_1_PRECISION_0),
    .IN_FRAC_WIDTH  (LAYER_NORM_PL0_1_PRECISION_1),
    .OUT_WIDTH  (LAYER_NORM_PL0_1_PRECISION_0),
    .OUT_FRAC_WIDTH  (LAYER_NORM_PL0_1_PRECISION_1),
    .ISQRT_LUT_MEMFILE(ISQRT_LUT_MEMFILE),  
    .ISQRT_LUT_POW (ISQRT_LUT_POW)  
) layer_norm_pl0_1
(
    .clk(clk),
    .rst(rst),

    .n_iters (layer_norm_pl0_1_n_iters), 
    .inv_numvalues_0 (layer_norm_pl0_1_inv_numvalues_0),
    .inv_numvalues_1 (layer_norm_pl0_1_inv_numvalues_1),

    .in_data (layer_norm_pl0_1_in_data), 
    .in_valid (layer_norm_pl0_1_in_valid),
    .in_ready (layer_norm_pl0_1_in_ready),
    .out_data (layer_norm_pl0_1_out_data) ,
    .out_valid (layer_norm_pl0_1_out_valid),
    .out_ready (layer_norm_pl0_1_out_ready)

);

fixed_linear_programmable #(
    .WEIGHTS_PRE_TRANSPOSED(LINEAR_PL0_0_WEIGHTS_PRE_TRANSPOSED),
    .DATA_IN_0_PRECISION_0(LINEAR_PL0_0_DATA_IN_0_PRECISION_0),
    .DATA_IN_0_PRECISION_1(LINEAR_PL0_0_DATA_IN_0_PRECISION_1),
    .DATA_IN_0_MAX_TENSOR_SIZE_DIM_0(LINEAR_PL0_0_DATA_IN_0_MAX_TENSOR_SIZE_DIM_0),
    .DATA_IN_0_MAX_TENSOR_SIZE_DIM_1(LINEAR_PL0_0_DATA_IN_0_MAX_TENSOR_SIZE_DIM_1),
    .DATA_IN_0_TENSOR_SIZE_DIM_2(LINEAR_PL0_0_DATA_IN_0_TENSOR_SIZE_DIM_2),
    .DATA_IN_0_PARALLELISM_DIM_0(LINEAR_PL0_0_DATA_IN_0_PARALLELISM_DIM_0),  // must equal WEIGHT_PARALLELISM_DIM_1
    .DATA_IN_0_PARALLELISM_DIM_1(LINEAR_PL0_0_DATA_IN_0_PARALLELISM_DIM_1),
    .DATA_IN_0_PARALLELISM_DIM_2(LINEAR_PL0_0_DATA_IN_0_PARALLELISM_DIM_2),
    .WEIGHT_PRECISION_0(LINEAR_PL0_0_WEIGHT_PRECISION_0),
    .WEIGHT_PRECISION_1(LINEAR_PL0_0_WEIGHT_PRECISION_1),
    .WEIGHT_MAX_TENSOR_SIZE_DIM_0(LINEAR_PL0_0_WEIGHT_MAX_TENSOR_SIZE_DIM_0),
    .WEIGHT_MAX_TENSOR_SIZE_DIM_1(LINEAR_PL0_0_WEIGHT_MAX_TENSOR_SIZE_DIM_1),
    .WEIGHT_PARALLELISM_DIM_0(LINEAR_PL0_0_WEIGHT_PARALLELISM_DIM_0),
    .WEIGHT_PARALLELISM_DIM_1(LINEAR_PL0_0_WEIGHT_PARALLELISM_DIM_1),
    .DATA_OUT_0_PRECISION_0(LINEAR_PL0_0_DATA_OUT_0_PRECISION_0),
    .DATA_OUT_0_PRECISION_1(LINEAR_PL0_0_DATA_OUT_0_PRECISION_1),
    .DATA_OUT_0_MAX_TENSOR_SIZE_DIM_0(LINEAR_PL0_0_DATA_OUT_0_MAX_TENSOR_SIZE_DIM_0),
    .DATA_OUT_0_MAX_TENSOR_SIZE_DIM_1(LINEAR_PL0_0_DATA_OUT_0_MAX_TENSOR_SIZE_DIM_1),
    .DATA_OUT_0_MAX_TENSOR_SIZE_DIM_2(LINEAR_PL0_0_DATA_OUT_0_MAX_TENSOR_SIZE_DIM_2),
    .DATA_OUT_0_PARALLELISM_DIM_0(LINEAR_PL0_0_DATA_OUT_0_PARALLELISM_DIM_0),
    .DATA_OUT_0_PARALLELISM_DIM_1(LINEAR_PL0_0_DATA_OUT_0_PARALLELISM_DIM_1),
    .DATA_OUT_0_PARALLELISM_DIM_2(LINEAR_PL0_0_DATA_OUT_0_PARALLELISM_DIM_2),
    .BIAS_PRECISION_0(LINEAR_PL0_0_BIAS_PRECISION_0),
    .BIAS_PRECISION_1(LINEAR_PL0_0_BIAS_PRECISION_1),
    .BIAS_MAX_TENSOR_SIZE_DIM_0(LINEAR_PL0_0_BIAS_MAX_TENSOR_SIZE_DIM_0),
    .BIAS_MAX_TENSOR_SIZE_DIM_1(LINEAR_PL0_0_BIAS_MAX_TENSOR_SIZE_DIM_1),
    .BIAS_PARALLELISM_DIM_0(LINEAR_PL0_0_BIAS_PARALLELISM_DIM_0),
    .BIAS_PARALLELISM_DIM_1(LINEAR_PL0_0_BIAS_PARALLELISM_DIM_1)
) linear_pl0_0 (
    .clk(clk),
    .rst(rst),
    .data_in_0_depth_dim1(linear_pl0_0_data_in_0_depth_dim1),
    .weight_tensor_size_dim0(linear_pl0_0_weight_tensor_size_dim0),
    .weight_depth_dim0(linear_pl0_0_weight_depth_dim0),
    .weight_depth_dim1(linear_pl0_0_weight_depth_dim1),
    .weight_depth_mult(linear_pl0_0_weight_depth_mult),
    .data_in_0(linear_pl0_0_data_in_0),
    .data_in_0_valid(linear_pl0_0_data_in_0_valid),
    .data_in_0_ready(linear_pl0_0_data_in_0_ready),
    .weight(linear_pl0_0_weight),
    .weight_valid(linear_pl0_0_weight_valid),
    .weight_ready(linear_pl0_0_weight_ready),
    .bias(linear_pl0_0_bias),
    .bias_valid(linear_pl0_0_bias_valid),
    .bias_ready(linear_pl0_0_bias_ready),
    .data_out_0(linear_pl0_0_data_out_0),
    .data_out_0_valid(linear_pl0_0_data_out_0_valid),
    .data_out_0_ready(linear_pl0_0_data_out_0_ready)
);

fixed_relu #(
    .DATA_IN_0_PRECISION_0(RELU_PL0_0_DATA_IN_0_PRECISION_0),
    .DATA_IN_0_PRECISION_1(RELU_PL0_0_DATA_IN_0_PRECISION_1),
    .DATA_IN_0_TENSOR_SIZE_DIM_0(RELU_PL0_0_DATA_IN_0_TENSOR_SIZE_DIM_0),
    .DATA_IN_0_TENSOR_SIZE_DIM_1(RELU_PL0_0_DATA_IN_0_TENSOR_SIZE_DIM_1),
    .DATA_IN_0_PARALLELISM_DIM_0(RELU_PL0_0_DATA_IN_0_PARALLELISM_DIM_0),
    .DATA_IN_0_PARALLELISM_DIM_1(RELU_PL0_0_DATA_IN_0_PARALLELISM_DIM_1),
    .DATA_OUT_0_PRECISION_0(RELU_PL0_0_DATA_OUT_0_PRECISION_0),
    .DATA_OUT_0_PRECISION_1(RELU_PL0_0_DATA_OUT_0_PRECISION_1),
    .DATA_OUT_0_TENSOR_SIZE_DIM_0(RELU_PL0_0_DATA_OUT_0_TENSOR_SIZE_DIM_0),
    .DATA_OUT_0_TENSOR_SIZE_DIM_1(RELU_PL0_0_DATA_OUT_0_TENSOR_SIZE_DIM_1),
    .DATA_OUT_0_PARALLELISM_DIM_0(RELU_PL0_0_DATA_OUT_0_PARALLELISM_DIM_0),
    .DATA_OUT_0_PARALLELISM_DIM_1(RELU_PL0_0_DATA_OUT_0_PARALLELISM_DIM_1)
) relu_pl0_0
(
    .rst(rst),
    .clk(clk),
    .data_in_0(relu_pl0_0_data_in_0),
    .data_out_0(relu_pl0_0_data_out_0),
    .data_in_0_valid(relu_pl0_0_data_in_0_valid),
    .data_in_0_ready(relu_pl0_0_data_in_0_ready),
    .data_out_0_valid(relu_pl0_0_data_out_0_valid),
    .data_out_0_ready(relu_pl0_0_data_out_0_ready)   
);

fixed_linear_programmable #(
    .WEIGHTS_PRE_TRANSPOSED(LINEAR_PL0_1_WEIGHTS_PRE_TRANSPOSED),
    .DATA_IN_0_PRECISION_0(LINEAR_PL0_1_DATA_IN_0_PRECISION_0),
    .DATA_IN_0_PRECISION_1(LINEAR_PL0_1_DATA_IN_0_PRECISION_1),
    .DATA_IN_0_MAX_TENSOR_SIZE_DIM_0(LINEAR_PL0_1_DATA_IN_0_MAX_TENSOR_SIZE_DIM_0),
    .DATA_IN_0_MAX_TENSOR_SIZE_DIM_1(LINEAR_PL0_1_DATA_IN_0_MAX_TENSOR_SIZE_DIM_1),
    .DATA_IN_0_TENSOR_SIZE_DIM_2(LINEAR_PL0_1_DATA_IN_0_TENSOR_SIZE_DIM_2),
    .DATA_IN_0_PARALLELISM_DIM_0(LINEAR_PL0_1_DATA_IN_0_PARALLELISM_DIM_0),  // must equal WEIGHT_PARALLELISM_DIM_1
    .DATA_IN_0_PARALLELISM_DIM_1(LINEAR_PL0_1_DATA_IN_0_PARALLELISM_DIM_1),
    .DATA_IN_0_PARALLELISM_DIM_2(LINEAR_PL0_1_DATA_IN_0_PARALLELISM_DIM_2),
    .WEIGHT_PRECISION_0(LINEAR_PL0_1_WEIGHT_PRECISION_0),
    .WEIGHT_PRECISION_1(LINEAR_PL0_1_WEIGHT_PRECISION_1),
    .WEIGHT_MAX_TENSOR_SIZE_DIM_0(LINEAR_PL0_1_WEIGHT_MAX_TENSOR_SIZE_DIM_0),
    .WEIGHT_MAX_TENSOR_SIZE_DIM_1(LINEAR_PL0_1_WEIGHT_MAX_TENSOR_SIZE_DIM_1),
    .WEIGHT_PARALLELISM_DIM_0(LINEAR_PL0_1_WEIGHT_PARALLELISM_DIM_0),
    .WEIGHT_PARALLELISM_DIM_1(LINEAR_PL0_1_WEIGHT_PARALLELISM_DIM_1),
    .DATA_OUT_0_PRECISION_0(LINEAR_PL0_1_DATA_OUT_0_PRECISION_0),
    .DATA_OUT_0_PRECISION_1(LINEAR_PL0_1_DATA_OUT_0_PRECISION_1),
    .DATA_OUT_0_MAX_TENSOR_SIZE_DIM_0(LINEAR_PL0_1_DATA_OUT_0_MAX_TENSOR_SIZE_DIM_0),
    .DATA_OUT_0_MAX_TENSOR_SIZE_DIM_1(LINEAR_PL0_1_DATA_OUT_0_MAX_TENSOR_SIZE_DIM_1),
    .DATA_OUT_0_MAX_TENSOR_SIZE_DIM_2(LINEAR_PL0_1_DATA_OUT_0_MAX_TENSOR_SIZE_DIM_2),
    .DATA_OUT_0_PARALLELISM_DIM_0(LINEAR_PL0_1_DATA_OUT_0_PARALLELISM_DIM_0),
    .DATA_OUT_0_PARALLELISM_DIM_1(LINEAR_PL0_1_DATA_OUT_0_PARALLELISM_DIM_1),
    .DATA_OUT_0_PARALLELISM_DIM_2(LINEAR_PL0_1_DATA_OUT_0_PARALLELISM_DIM_2),
    .BIAS_PRECISION_0(LINEAR_PL0_1_BIAS_PRECISION_0),
    .BIAS_PRECISION_1(LINEAR_PL0_1_BIAS_PRECISION_1),
    .BIAS_MAX_TENSOR_SIZE_DIM_0(LINEAR_PL0_1_BIAS_MAX_TENSOR_SIZE_DIM_0),
    .BIAS_MAX_TENSOR_SIZE_DIM_1(LINEAR_PL0_1_BIAS_MAX_TENSOR_SIZE_DIM_1),
    .BIAS_PARALLELISM_DIM_0(LINEAR_PL0_1_BIAS_PARALLELISM_DIM_0),
    .BIAS_PARALLELISM_DIM_1(LINEAR_PL0_1_BIAS_PARALLELISM_DIM_1)
) linear_pl0_1 (
    .clk(clk),
    .rst(rst),
    .data_in_0_depth_dim1(linear_pl0_1_data_in_0_depth_dim1),
    .weight_tensor_size_dim0(linear_pl0_1_weight_tensor_size_dim0),
    .weight_depth_dim0(linear_pl0_1_weight_depth_dim0),
    .weight_depth_dim1(linear_pl0_1_weight_depth_dim1),
    .weight_depth_mult(linear_pl0_1_weight_depth_mult),
    .data_in_0(linear_pl0_1_data_in_0),
    .data_in_0_valid(linear_pl0_1_data_in_0_valid),
    .data_in_0_ready(linear_pl0_1_data_in_0_ready),
    .weight(linear_pl0_1_weight),
    .weight_valid(linear_pl0_1_weight_valid),
    .weight_ready(linear_pl0_1_weight_ready),
    .bias(linear_pl0_1_bias),
    .bias_valid(linear_pl0_1_bias_valid),
    .bias_ready(linear_pl0_1_bias_ready),
    .data_out_0(linear_pl0_1_data_out_0),
    .data_out_0_valid(linear_pl0_1_data_out_0_valid),
    .data_out_0_ready(linear_pl0_1_data_out_0_ready)
);

fixed_adder #(
    .DATA_IN_0_PRECISION_0(RESIDUAL_PL0_1_DATA_IN_0_PRECISION_0),
    .DATA_IN_0_PRECISION_1(RESIDUAL_PL0_1_DATA_IN_0_PRECISION_1),
    .DATA_IN_0_TENSOR_SIZE_DIM_0(RESIDUAL_PL0_1_DATA_IN_0_TENSOR_SIZE_DIM_0),
    .DATA_IN_0_TENSOR_SIZE_DIM_1(RESIDUAL_PL0_1_DATA_IN_0_TENSOR_SIZE_DIM_1),
    .DATA_IN_0_TENSOR_SIZE_DIM_2(RESIDUAL_PL0_1_DATA_IN_0_TENSOR_SIZE_DIM_2),
    .DATA_IN_0_PARALLELISM_DIM_0(RESIDUAL_PL0_1_DATA_IN_0_PARALLELISM_DIM_0),
    .DATA_IN_0_PARALLELISM_DIM_1(RESIDUAL_PL0_1_DATA_IN_0_PARALLELISM_DIM_1),
    .DATA_IN_0_PARALLELISM_DIM_2(RESIDUAL_PL0_1_DATA_IN_0_PARALLELISM_DIM_2),
    .DATA_IN_1_PRECISION_0(RESIDUAL_PL0_1_DATA_IN_1_PRECISION_0),
    .DATA_IN_1_PRECISION_1(RESIDUAL_PL0_1_DATA_IN_1_PRECISION_1),
    .DATA_IN_1_TENSOR_SIZE_DIM_0(RESIDUAL_PL0_1_DATA_IN_1_TENSOR_SIZE_DIM_0),
    .DATA_IN_1_TENSOR_SIZE_DIM_1(RESIDUAL_PL0_1_DATA_IN_1_TENSOR_SIZE_DIM_1),
    .DATA_IN_1_TENSOR_SIZE_DIM_2(RESIDUAL_PL0_1_DATA_IN_1_TENSOR_SIZE_DIM_2),
    .DATA_IN_1_PARALLELISM_DIM_0(RESIDUAL_PL0_1_DATA_IN_1_PARALLELISM_DIM_0),
    .DATA_IN_1_PARALLELISM_DIM_1(RESIDUAL_PL0_1_DATA_IN_1_PARALLELISM_DIM_1),
    .DATA_IN_1_PARALLELISM_DIM_2(RESIDUAL_PL0_1_DATA_IN_1_PARALLELISM_DIM_2)
) residual_pl0_1
(
    .clk(clk),
    .rst(rst),
    .data_in_0(residual_pl0_1_data_in_0),
    .data_in_0_valid(residual_pl0_1_data_in_0_valid),
    .data_in_0_ready(residual_pl0_1_data_in_0_ready),
    .data_in_1(residual_pl0_1_data_in_1),
    .data_in_1_valid(residual_pl0_1_data_in_1_valid),
    .data_in_1_ready(residual_pl0_1_data_in_1_ready),
    .data_out_0(residual_pl0_1_data_out_0),
    .data_out_0_valid(residual_pl0_1_data_out_0_valid),
    .data_out_0_ready(residual_pl0_1_data_out_0_ready)
);


fixed_unroll_buffer #(
    .ROLL_MAX_DISTANCE (ROLL_PL0_0_ROLL_MAX_DISTANCE),
    .MAX_BUFFER_SIZE (ROLL_PL0_0_MAX_BUFFER_SIZE),
    .DATA_IN_0_PARALLELISM_DIM_0 (ROLL_PL0_0_DATA_IN_0_PARALLELISM_DIM_0),   
    .DATA_IN_0_PARALLELISM_DIM_1 (ROLL_PL0_0_DATA_IN_0_PARALLELISM_DIM_1),   
    .DATA_IN_0_PRECISION_0 (ROLL_PL0_0_DATA_IN_0_PRECISION_0),  
    .DATA_IN_0_PRECISION_1 (ROLL_PL0_0_DATA_IN_0_PRECISION_1),
    .DATA_OUT_0_PRECISION_0 (ROLL_PL0_0_DATA_OUT_0_PRECISION_0),
    .DATA_OUT_0_PARALLELISM_DIM_0 (ROLL_PL0_0_DATA_OUT_0_PARALLELISM_DIM_0),
    .DATA_OUT_0_PARALLELISM_DIM_1 (ROLL_PL0_0_DATA_OUT_0_PARALLELISM_DIM_1)

) roll_pl0_0
(
    .clk(clk),
    .rst(rst),
    .roll_distance(roll_pl0_0_roll_distance),
    .buffer_size(roll_pl0_0_buffer_size),
    .data_in_0(roll_pl0_0_data_in_0),
    .data_in_0_valid(roll_pl0_0_data_in_0_valid), 
    .data_in_0_ready(roll_pl0_0_data_in_0_ready),
    .data_out_0(roll_pl0_0_data_out_0),
    .data_out_0_valid(roll_pl0_0_data_out_0_valid),
    .data_out_0_ready(roll_pl0_0_data_out_0_ready)    
); 

fixed_unroll_buffer #(
    .ROLL_MAX_DISTANCE (ROLL_PL0_0_ROLL_MAX_DISTANCE),
    .MAX_BUFFER_SIZE (ROLL_PL0_0_MAX_BUFFER_SIZE),
    .DATA_IN_0_PARALLELISM_DIM_0 (ROLL_PL0_0_DATA_IN_0_PARALLELISM_DIM_0),   
    .DATA_IN_0_PARALLELISM_DIM_1 (ROLL_PL0_0_DATA_IN_0_PARALLELISM_DIM_1),   
    .DATA_IN_0_PRECISION_0 (ROLL_PL0_0_DATA_IN_0_PRECISION_0),  
    .DATA_IN_0_PRECISION_1 (ROLL_PL0_0_DATA_IN_0_PRECISION_1),
    .DATA_OUT_0_PRECISION_0 (ROLL_PL0_0_DATA_OUT_0_PRECISION_0),
    .DATA_OUT_0_PARALLELISM_DIM_0 (ROLL_PL0_0_DATA_OUT_0_PARALLELISM_DIM_0),
    .DATA_OUT_0_PARALLELISM_DIM_1 (ROLL_PL0_0_DATA_OUT_0_PARALLELISM_DIM_1)

) roll_pl0_1
(
    .clk(clk),
    .rst(rst),
    .roll_distance(roll_pl0_1_roll_distance),
    .buffer_size(roll_pl0_1_buffer_size),
    .data_in_0(roll_pl0_1_data_in_0),
    .data_in_0_valid(roll_pl0_1_data_in_0_valid), 
    .data_in_0_ready(roll_pl0_1_data_in_0_ready),
    .data_out_0(roll_pl0_1_data_out_0),
    .data_out_0_valid(roll_pl0_1_data_out_0_valid),
    .data_out_0_ready(roll_pl0_1_data_out_0_ready)    
); 



assign adder_fifo_pl0_0_in_data = input_signal;
assign layer_norm_pl0_0_in_data = input_signal;


//layernorm0_0 -> mha0_0
//1st module
assign layer_norm_pl0_0_out_ready = mha_pl0_0_data_in_0_ready;
//2nd module
assign mha_pl0_0_data_in_0_valid = layer_norm_pl0_0_out_valid;
assign mha_pl0_0_data_in_0 = layer_norm_pl0_0_out_data;

//mha0_0 -> adder0_0
//1st module
assign mha_pl0_0_data_out_0_ready = residual_pl0_0_data_in_0_ready;
//2nd module
assign residual_pl0_0_data_in_0_valid = mha_pl0_0_data_out_0_valid;
assign residual_pl0_0_data_in_0 = mha_pl0_0_data_out_0;

//fifo0_0 -> adder0_0
//1st module
assign adder_fifo_pl0_0_out_ready = residual_pl0_0_data_in_1_ready;
//2nd module
assign residual_pl0_0_data_in_1_valid = adder_fifo_pl0_0_out_valid;
assign residual_pl0_0_data_in_1 = adder_fifo_pl0_0_out_data;

//adder0_0 -> layernorm0_1
//control handshakes done by split module
assign layer_norm_pl0_1_in_data = residual_pl0_0_data_out_0;


//adder0_0 -> layernorm0_1
//control handshakes done by split module
assign adder_fifo_pl0_1_in_data = residual_pl0_0_data_out_0;

//layernorm0_1 -> linear0_0
//1st module
assign layer_norm_pl0_1_out_ready = linear_pl0_0_data_in_0_ready;
//2nd module
assign linear_pl0_0_data_in_0_valid = layer_norm_pl0_1_out_valid;
assign linear_pl0_0_data_in_0 = layer_norm_pl0_1_out_data;


//linear0_0 -> relu0_0
//1st module
assign linear_pl0_0_data_out_0_ready = relu_pl0_0_data_in_0_ready;
//2nd module
assign relu_pl0_0_data_in_0_valid = linear_pl0_0_data_out_0_valid;
assign relu_pl0_0_data_in_0 = linear_pl0_0_data_out_0;

//relu0_0 -> linear0_1
//1st module
assign relu_pl0_0_data_out_0_ready = linear_pl0_1_data_in_0_ready;
//2nd module
assign linear_pl0_1_data_in_0_valid = relu_pl0_0_data_out_0_valid;
assign linear_pl0_1_data_in_0 = relu_pl0_0_data_out_0;

//linear0_1 -> adder0_1
//1st module
assign linear_pl0_1_data_out_0_ready = residual_pl0_1_data_in_0_ready;
//2nd module
assign residual_pl0_1_data_in_0_valid = linear_pl0_1_data_out_0_valid;
assign residual_pl0_1_data_in_0 = linear_pl0_1_data_out_0;

//fifo0_1 -> adder0_1
//1st module
assign adder_fifo_pl0_1_out_ready = residual_pl0_1_data_in_1_ready;
//2nd module
assign residual_pl0_1_data_in_1_valid = adder_fifo_pl0_1_out_valid;
assign residual_pl0_1_data_in_1 = adder_fifo_pl0_1_out_data;

//adder0_1 -> roll0_0
assign residual_pl0_1_data_out_0_ready = input_mux_ready;
assign input_mux_valid = residual_pl0_1_data_out_0_valid;
assign input_mux_data = residual_pl0_1_data_out_0;


always_comb begin
    if (load_input_seq) begin
        input_buffer_data_out_0_ready = input_signal_ready;
        input_signal_valid = input_buffer_data_out_0_valid;
        input_signal = input_buffer_data_out_0;
    end else begin
        output_mux_ready = input_signal_ready;
        input_signal_valid = output_mux_valid;
        input_signal = output_mux_data;
    end
end


always_comb begin
    if (input_mux_ctrl) begin
        input_mux_ready = roll_pl0_0_data_in_0_ready;
        roll_pl0_0_data_in_0_valid = input_mux_valid;
        roll_pl0_0_data_in_0 = input_mux_data;
    end else begin
        input_mux_ready = roll_pl0_1_data_in_0_ready;
        roll_pl0_1_data_in_0_valid = input_mux_valid;
        roll_pl0_1_data_in_0 = input_mux_data;
    end
end

always_comb begin
    if(output_mux_ctrl)begin
        roll_pl0_0_data_out_0_ready = output_mux_ready;
        output_mux_valid = roll_pl0_0_data_out_0_valid;
        output_mux_data = roll_pl0_0_data_out_0;
    end else begin
        roll_pl0_1_data_out_0_ready = output_mux_ready;
        output_mux_valid = roll_pl0_0_data_out_0_valid;
        output_mux_data = roll_pl0_0_data_out_0;
    end

end




endmodule