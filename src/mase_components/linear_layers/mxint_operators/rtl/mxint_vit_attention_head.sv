`timescale 1ns / 1ps
module mxint_vit_attention_head #(
    // Input dimensions and parallelism
    parameter IN_DATA_TENSOR_SIZE_DIM_0 = 32,
    parameter IN_DATA_TENSOR_SIZE_DIM_1 = 10,
    parameter IN_DATA_PARALLELISM_DIM_0 = 2,
    parameter IN_DATA_PARALLELISM_DIM_1 = 2,
    parameter IN_DATA_PRECISION_0 = 16,
    parameter IN_DATA_PRECISION_1 = 3,

    // Output dimensions
    parameter OUT_DATA_TENSOR_SIZE_DIM_0 = IN_DATA_TENSOR_SIZE_DIM_0,
    parameter OUT_DATA_TENSOR_SIZE_DIM_1 = IN_DATA_TENSOR_SIZE_DIM_1,
    parameter OUT_DATA_PARALLELISM_DIM_0 = IN_DATA_PARALLELISM_DIM_0,
    parameter OUT_DATA_PARALLELISM_DIM_1 = IN_DATA_PARALLELISM_DIM_1,
    parameter OUT_DATA_PRECISION_0 = 16,
    parameter OUT_DATA_PRECISION_1 = 3
) (
    input logic clk,
    input logic rst,

    // Query inputs
    input logic [IN_DATA_PRECISION_0-1:0] mquery [IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_1-1:0],
    input logic [IN_DATA_PRECISION_1-1:0] equery,
    input logic query_valid,
    output logic query_ready,

    // Key inputs
    input logic [IN_DATA_PRECISION_0-1:0] mkey [IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_1-1:0],
    input logic [IN_DATA_PRECISION_1-1:0] ekey,
    input logic key_valid,
    output logic key_ready,

    // Value inputs
    input logic [IN_DATA_PRECISION_0-1:0] mvalue [IN_DATA_PARALLELISM_DIM_0*IN_DATA_PARALLELISM_DIM_1-1:0],
    input logic [IN_DATA_PRECISION_1-1:0] evalue,
    input logic value_valid,
    output logic value_ready,

    // Outputs
    output logic [OUT_DATA_PRECISION_0-1:0] mout [OUT_DATA_PARALLELISM_DIM_0*OUT_DATA_PARALLELISM_DIM_1-1:0],
    output logic [OUT_DATA_PRECISION_1-1:0] eout,
    output logic out_valid,
    input logic out_ready
);

    // QK matmul signals
    logic [IN_DATA_PRECISION_0-1:0] qk_mout [IN_DATA_PARALLELISM_DIM_1 * IN_DATA_PARALLELISM_DIM_1-1:0];
    logic [IN_DATA_PRECISION_1-1:0] qk_eout;
    logic qk_valid, qk_ready;

    // Softmax signals 
    logic [OUT_DATA_PRECISION_0-1:0] sm_mout [IN_DATA_PARALLELISM_DIM_1 * IN_DATA_PARALLELISM_DIM_1-1:0];
    logic [OUT_DATA_PRECISION_1-1:0] sm_eout;
    logic sm_valid, sm_ready;

    // First compute Q * K^T using mxint_linear
    mxint_linear #(
        .DATA_IN_0_PRECISION_0(IN_DATA_PRECISION_0),
        .DATA_IN_0_PRECISION_1(IN_DATA_PRECISION_1),
        .DATA_IN_0_TENSOR_SIZE_DIM_0(IN_DATA_TENSOR_SIZE_DIM_0),
        .DATA_IN_0_TENSOR_SIZE_DIM_1(IN_DATA_TENSOR_SIZE_DIM_1),
        .DATA_IN_0_PARALLELISM_DIM_0(IN_DATA_PARALLELISM_DIM_0),
        .DATA_IN_0_PARALLELISM_DIM_1(IN_DATA_PARALLELISM_DIM_1),
        
        .WEIGHT_PRECISION_0(IN_DATA_PRECISION_0),
        .WEIGHT_PRECISION_1(IN_DATA_PRECISION_1),
        .WEIGHT_TENSOR_SIZE_DIM_0(IN_DATA_TENSOR_SIZE_DIM_0),
        .WEIGHT_TENSOR_SIZE_DIM_1(IN_DATA_TENSOR_SIZE_DIM_1),
        .WEIGHT_PARALLELISM_DIM_0(IN_DATA_PARALLELISM_DIM_0),
        .WEIGHT_PARALLELISM_DIM_1(IN_DATA_PARALLELISM_DIM_1),
        
        .DATA_OUT_0_PRECISION_0(IN_DATA_PRECISION_0),
        .DATA_OUT_0_PRECISION_1(IN_DATA_PRECISION_1),
        .HAS_BIAS(0)
    ) query_key_linear (
        .clk(clk),
        .rst(rst),
        .mdata_in_0(mquery),
        .edata_in_0(equery),
        .data_in_0_valid(query_valid),
        .data_in_0_ready(query_ready),
        .mweight(mkey),
        .eweight(ekey),
        .weight_valid(key_valid),
        .weight_ready(key_ready),
        .mbias(), // Not used since HAS_BIAS=0
        .ebias(),
        .bias_valid(1'b1),
        .bias_ready(),
        .mdata_out_0(qk_mout),
        .edata_out_0(qk_eout),
        .data_out_0_valid(qk_valid),
        .data_out_0_ready(qk_ready)
    );

    // Apply softmax to QK^T result
    mxint_softmax #(
        .DATA_IN_0_PRECISION_0(IN_DATA_PRECISION_0),
        .DATA_IN_0_PRECISION_1(IN_DATA_PRECISION_1), 
        .DATA_IN_0_DIM(IN_DATA_TENSOR_SIZE_DIM_1),
        .DATA_IN_0_PARALLELISM(IN_DATA_PARALLELISM_DIM_1),
        .DATA_OUT_0_PRECISION_0(OUT_DATA_PRECISION_0),
        .DATA_OUT_0_PRECISION_1(OUT_DATA_PRECISION_1),
        .DATA_OUT_0_DIM(IN_DATA_TENSOR_SIZE_DIM_1),
        .DATA_OUT_0_PARALLELISM(IN_DATA_PARALLELISM_DIM_1)
    ) attention_softmax (
        .clk(clk),
        .rst(rst),
        .mdata_in_0(qk_mout),
        .edata_in_0(qk_eout),
        .data_in_0_valid(qk_valid),
        .data_in_0_ready(qk_ready),
        .mdata_out_0(sm_mout),
        .edata_out_0(sm_eout),
        .data_out_0_valid(sm_valid),
        .data_out_0_ready(sm_ready)
    );

    // Compute softmax(QK^T)V
    mxint_matmul #(
        .A_TOTAL_DIM0(IN_DATA_TENSOR_SIZE_DIM_1),
        .A_TOTAL_DIM1(IN_DATA_TENSOR_SIZE_DIM_1),
        .B_TOTAL_DIM0(IN_DATA_TENSOR_SIZE_DIM_0),
        .B_TOTAL_DIM1(IN_DATA_TENSOR_SIZE_DIM_1),
        .A_COMPUTE_DIM0(IN_DATA_PARALLELISM_DIM_1),
        .A_COMPUTE_DIM1(IN_DATA_PARALLELISM_DIM_1),
        .B_COMPUTE_DIM0(IN_DATA_PARALLELISM_DIM_0),
        .B_COMPUTE_DIM1(IN_DATA_PARALLELISM_DIM_1),
        .A_MAN_WIDTH(OUT_DATA_PRECISION_0),
        .A_EXP_WIDTH(OUT_DATA_PRECISION_1),
        .B_MAN_WIDTH(IN_DATA_PRECISION_0),
        .B_EXP_WIDTH(IN_DATA_PRECISION_1),
        .OUT_MAN_WIDTH(OUT_DATA_PRECISION_0),
        .OUT_EXP_WIDTH(OUT_DATA_PRECISION_1)
    ) attention_value_matmul (
        .clk(clk),
        .rst(rst),
        .ma_data(sm_mout),
        .ea_data(sm_eout),
        .a_valid(sm_valid),
        .a_ready(sm_ready),
        .mb_data(mvalue),
        .eb_data(evalue),
        .b_valid(value_valid),
        .b_ready(value_ready), 
        .mout_data(mout),
        .eout_data(eout),
        .out_valid(out_valid),
        .out_ready(out_ready)
    );

endmodule
