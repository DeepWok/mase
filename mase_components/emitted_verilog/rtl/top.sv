
// =====================================
//     Mase Hardware
//     Model: top
//     21/01/2024 17:38:44
// =====================================
`timescale 1ns/1ps
module top #(
    parameter fc1_DATA_IN_0_PRECISION_0 = 32,
    parameter fc1_DATA_IN_0_TENSOR_SIZE_DIM_0 = 784,
    parameter fc1_DATA_IN_0_PARALLELISM_DIM_0 = 784,
    parameter fc1_DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter fc1_DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter fc1_WEIGHT_PRECISION_0 = 32,
    parameter fc1_WEIGHT_TENSOR_SIZE_DIM_0 = 784,
    parameter fc1_WEIGHT_PARALLELISM_DIM_0 = 784,
    parameter fc1_WEIGHT_TENSOR_SIZE_DIM_1 = 10,
    parameter fc1_WEIGHT_PARALLELISM_DIM_1 = 10,
    parameter fc1_BIAS_PRECISION_0 = 32,
    parameter fc1_BIAS_TENSOR_SIZE_DIM_0 = 10,
    parameter fc1_BIAS_PARALLELISM_DIM_0 = 10,
    parameter fc1_DATA_OUT_0_PRECISION_0 = 32,
    parameter fc1_DATA_OUT_0_TENSOR_SIZE_DIM_0 = 10,
    parameter fc1_DATA_OUT_0_PARALLELISM_DIM_0 = 10,
    parameter fc1_DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,
    parameter fc1_DATA_OUT_0_PARALLELISM_DIM_1 = 1,
    parameter relu_DATA_IN_0_PRECISION_0 = 32,
    parameter relu_DATA_IN_0_TENSOR_SIZE_DIM_0 = 10,
    parameter relu_DATA_IN_0_PARALLELISM_DIM_0 = 10,
    parameter relu_DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter relu_DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter relu_INPLACE = 0,
    parameter relu_DATA_OUT_0_PRECISION_0 = 32,
    parameter relu_DATA_OUT_0_TENSOR_SIZE_DIM_0 = 10,
    parameter relu_DATA_OUT_0_PARALLELISM_DIM_0 = 10,
    parameter relu_DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,
    parameter relu_DATA_OUT_0_PARALLELISM_DIM_1 = 1
) (
    input clk,
    input rst,

    input  [fc1_DATA_IN_0_PRECISION_0-1:0] data_in_0 [fc1_DATA_IN_0_PARALLELISM_DIM_0*fc1_DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input  data_in_0_valid,
    output data_in_0_ready,
    output  [relu_DATA_OUT_0_PRECISION_0-1:0] data_out_0 [relu_DATA_OUT_0_PARALLELISM_DIM_0*relu_DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output  data_out_0_valid,
    input data_out_0_ready
);

// --------------------------
//   fc1 signals
// --------------------------
logic [fc1_DATA_IN_0_PRECISION_0-1:0]  fc1_data_in_0        [fc1_DATA_IN_0_PARALLELISM_DIM_0*fc1_DATA_IN_0_PARALLELISM_DIM_1-1:0];
logic                             fc1_data_in_0_valid;
logic                             fc1_data_in_0_ready;
logic [fc1_WEIGHT_PRECISION_0-1:0]  fc1_weight        [fc1_WEIGHT_PARALLELISM_DIM_0*fc1_WEIGHT_PARALLELISM_DIM_1-1:0];
logic                             fc1_weight_valid;
logic                             fc1_weight_ready;
logic [fc1_BIAS_PRECISION_0-1:0]  fc1_bias        [fc1_BIAS_PARALLELISM_DIM_0-1:0];
logic                             fc1_bias_valid;
logic                             fc1_bias_ready;
logic [fc1_DATA_OUT_0_PRECISION_0-1:0]  fc1_data_out_0        [fc1_DATA_OUT_0_PARALLELISM_DIM_0*fc1_DATA_OUT_0_PARALLELISM_DIM_1-1:0];
logic                             fc1_data_out_0_valid;
logic                             fc1_data_out_0_ready;
// --------------------------
//   relu signals
// --------------------------
logic [relu_DATA_IN_0_PRECISION_0-1:0]  relu_data_in_0        [relu_DATA_IN_0_PARALLELISM_DIM_0*relu_DATA_IN_0_PARALLELISM_DIM_1-1:0];
logic                             relu_data_in_0_valid;
logic                             relu_data_in_0_ready;
logic [relu_DATA_OUT_0_PRECISION_0-1:0]  relu_data_out_0        [relu_DATA_OUT_0_PARALLELISM_DIM_0*relu_DATA_OUT_0_PARALLELISM_DIM_1-1:0];
logic                             relu_data_out_0_valid;
logic                             relu_data_out_0_ready;

// --------------------------
//   Component instantiation
// --------------------------

// fc1
// fixed_linear #(
//     .DATA_IN_0_PRECISION_0(fc1_DATA_IN_0_PRECISION_0), // = 32
//     .DATA_IN_0_TENSOR_SIZE_DIM_0(fc1_DATA_IN_0_TENSOR_SIZE_DIM_0), // = 784
//     .DATA_IN_0_PARALLELISM_DIM_0(fc1_DATA_IN_0_PARALLELISM_DIM_0), // = 784
//     .DATA_IN_0_TENSOR_SIZE_DIM_1(fc1_DATA_IN_0_TENSOR_SIZE_DIM_1), // = 1
//     .DATA_IN_0_PARALLELISM_DIM_1(fc1_DATA_IN_0_PARALLELISM_DIM_1), // = 1
//     .WEIGHT_PRECISION_0(fc1_WEIGHT_PRECISION_0), // = 32
//     .WEIGHT_TENSOR_SIZE_DIM_0(fc1_WEIGHT_TENSOR_SIZE_DIM_0), // = 784
//     .WEIGHT_PARALLELISM_DIM_0(fc1_WEIGHT_PARALLELISM_DIM_0), // = 784
//     .WEIGHT_TENSOR_SIZE_DIM_1(fc1_WEIGHT_TENSOR_SIZE_DIM_1), // = 10
//     .WEIGHT_PARALLELISM_DIM_1(fc1_WEIGHT_PARALLELISM_DIM_1), // = 10
//     .BIAS_PRECISION_0(fc1_BIAS_PRECISION_0), // = 32
//     .BIAS_TENSOR_SIZE_DIM_0(fc1_BIAS_TENSOR_SIZE_DIM_0), // = 10
//     .BIAS_PARALLELISM_DIM_0(fc1_BIAS_PARALLELISM_DIM_0), // = 10
//     .DATA_OUT_0_PRECISION_0(fc1_DATA_OUT_0_PRECISION_0), // = 32
//     .DATA_OUT_0_TENSOR_SIZE_DIM_0(fc1_DATA_OUT_0_TENSOR_SIZE_DIM_0), // = 10
//     .DATA_OUT_0_PARALLELISM_DIM_0(fc1_DATA_OUT_0_PARALLELISM_DIM_0), // = 10
//     .DATA_OUT_0_TENSOR_SIZE_DIM_1(fc1_DATA_OUT_0_TENSOR_SIZE_DIM_1), // = 1
//     .DATA_OUT_0_PARALLELISM_DIM_1(fc1_DATA_OUT_0_PARALLELISM_DIM_1)
// ) fc1_inst (
//     .clk(clk),
//     .rst(rst),

//     .data_in_0(fc1_data_in_0),
//     .data_in_0_valid(fc1_data_in_0_valid),
//     .data_in_0_ready(fc1_data_in_0_ready),

//     .data_out_0(fc1_data_out_0),
//     .data_out_0_valid(fc1_data_out_0_valid),
//     .data_out_0_ready(fc1_data_out_0_ready)
// );

fc1_weight_source #(
.WEIGHT_PRECISION_0(fc1_WEIGHT_PRECISION_0),
.WEIGHT_TENSOR_SIZE_DIM_0(fc1_WEIGHT_TENSOR_SIZE_DIM_0),
.WEIGHT_PARALLELISM_DIM_0(fc1_WEIGHT_PARALLELISM_DIM_0),
.WEIGHT_TENSOR_SIZE_DIM_1(fc1_WEIGHT_TENSOR_SIZE_DIM_1),
.WEIGHT_PARALLELISM_DIM_1(fc1_WEIGHT_PARALLELISM_DIM_1)
) fc1_weight_source_0 (
    .clk(clk),
    .rst(rst),
    .data_out(fc1_weight),
    .data_out_ready(fc1_weight_ready),
    .data_out_valid(fc1_weight_valid)
);

fc1_bias_source #(
.BIAS_PRECISION_0(fc1_BIAS_PRECISION_0),
.BIAS_TENSOR_SIZE_DIM_0(fc1_BIAS_TENSOR_SIZE_DIM_0),
.BIAS_PARALLELISM_DIM_0(fc1_BIAS_PARALLELISM_DIM_0)
) fc1_bias_source_0 (
    .clk(clk),
    .rst(rst),
    .data_out(fc1_bias),
    .data_out_ready(fc1_bias_ready),
    .data_out_valid(fc1_bias_valid)
);

// relu
fixed_relu #(
    .DATA_IN_0_PRECISION_0(relu_DATA_IN_0_PRECISION_0), // = 32
    .DATA_IN_0_TENSOR_SIZE_DIM_0(relu_DATA_IN_0_TENSOR_SIZE_DIM_0), // = 10
    .DATA_IN_0_PARALLELISM_DIM_0(relu_DATA_IN_0_PARALLELISM_DIM_0), // = 10
    .DATA_IN_0_TENSOR_SIZE_DIM_1(relu_DATA_IN_0_TENSOR_SIZE_DIM_1), // = 1
    .DATA_IN_0_PARALLELISM_DIM_1(relu_DATA_IN_0_PARALLELISM_DIM_1), // = 1
    .INPLACE(relu_INPLACE), // = 0
    .DATA_OUT_0_PRECISION_0(relu_DATA_OUT_0_PRECISION_0), // = 32
    .DATA_OUT_0_TENSOR_SIZE_DIM_0(relu_DATA_OUT_0_TENSOR_SIZE_DIM_0), // = 10
    .DATA_OUT_0_PARALLELISM_DIM_0(relu_DATA_OUT_0_PARALLELISM_DIM_0), // = 10
    .DATA_OUT_0_TENSOR_SIZE_DIM_1(relu_DATA_OUT_0_TENSOR_SIZE_DIM_1), // = 1
    .DATA_OUT_0_PARALLELISM_DIM_1(relu_DATA_OUT_0_PARALLELISM_DIM_1)
) relu_inst (
    .clk(clk),
    .rst(rst),

    .data_in_0(relu_data_in_0),
    .data_in_0_valid(relu_data_in_0_valid),
    .data_in_0_ready(relu_data_in_0_ready),
    
    .data_out_0(relu_data_out_0),
    .data_out_0_valid(relu_data_out_0_valid),
    .data_out_0_ready(relu_data_out_0_ready)
);


// --------------------------
//   Interconnections
// --------------------------
    
    assign data_in_0_ready = fc1_data_in_0_ready;
    assign fc1_data_in_0_valid    = data_in_0_valid;
    assign fc1_data_in_0    = data_in_0;

    assign data_out_0_valid = relu_data_out_0_valid;
    assign relu_data_out_0_ready    = data_out_0_ready;
    assign data_out_0 = relu_data_out_0;

    assign fc1_data_out_0_ready  = relu_data_in_0_ready;
    assign relu_data_in_0_valid    = fc1_data_out_0_valid;
    assign relu_data_in_0 = fc1_data_out_0;

endmodule
    