`timescale 1ns / 1ps


module scatter_threshold #(
    parameter DATA_IN_0_PRECISION_0 = 4,
    
    // parameter DATA_IN_0_PRECISION_1 = 3,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 4,
    // parameter DATA_IN_0_PRECISION_0 = 32,
    parameter DATA_OUT_0_PRECISION_0 = 4,
    parameter HIGH_SLOTS = 2,
    parameter LOW_SLOTS = DATA_IN_0_TENSOR_SIZE_DIM_0-HIGH_SLOTS,
    parameter THRESHOLD = 6
    // parameter DATA_IN_0_PARALLELISM_DIM_0 = 16,
    // parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,
    // parameter DATA_OUT_0_PARALLELISM_DIM_0 = 16
    // parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1,
) (
    input clk,
    input rst,
    input  logic signed [DATA_IN_0_PRECISION_0-1:0] data_in[DATA_IN_0_TENSOR_SIZE_DIM_0-1:0],
    output  [DATA_OUT_0_PRECISION_0-1:0] data_out[DATA_IN_0_TENSOR_SIZE_DIM_0-1:0],
    output [DATA_IN_0_PRECISION_0-1:0] o_high_precision [DATA_IN_0_TENSOR_SIZE_DIM_0-1:0], 
    output [DATA_IN_0_PRECISION_0-1:0] o_low_precision [DATA_IN_0_TENSOR_SIZE_DIM_0-1:0] 
);




    //Logic to indentify values larger than x (2MSB check) //Can parameterise
    // logic [DATA_IN_0_PRECISION_0-2:0] mantisssa [DATA_IN_0_TENSOR_SIZE_DIM_0-1:0];
    // logic [DATA_IN_0_TENSOR_SIZE_DIM_0-1:0] sign;

    logic [DATA_IN_0_TENSOR_SIZE_DIM_0-1:0] high_precision_req_vec;
    for (genvar i = 0; i < DATA_IN_0_TENSOR_SIZE_DIM_0; i = i + 1) begin
        //Check if greater than threshold
        assign high_precision_req_vec[i] = ($signed(data_in[i])> +THRESHOLD | $signed(data_in[i])< -THRESHOLD);
        assign data_out[i] = data_in[i];
    end


    
    //Logic to assign indicies based on priority
    //Pack array first
    wire [3:0] output_mask;

    logic [$clog2(DATA_IN_0_TENSOR_SIZE_DIM_0)-1:0] address_outliers[HIGH_SLOTS-1:0];
    priority_encoder #(
        .NUM_INPUT_CHANNELS(DATA_IN_0_TENSOR_SIZE_DIM_0),
        .NUM_OUPUT_CHANNELS(DATA_IN_0_TENSOR_SIZE_DIM_0),
        .NO_INDICIES(HIGH_SLOTS)
    )
    encoder1(
        .input_channels(high_precision_req_vec),
        // .output_channels(address_outliers)
        .mask(output_mask)
    );

    //Logic to turn indicies to mask

    // index_to_mask #(
    //     .NUM_INPUT_CHANNELS(DATA_IN_0_TENSOR_SIZE_DIM_0),
    //     .NUM_OUPUT_CHANNELS(DATA_IN_0_TENSOR_SIZE_DIM_0),
    //     .NO_INDICIES(HIGH_SLOTS),
    //     .OUTPUT_WIDTH(DATA_IN_0_TENSOR_SIZE_DIM_0)
    // )mask_gen(
    //     .indicies(address_outliers),
    //     .output_mask(output_mask)
    // );



    //Logic to apply mask
    array_zero_mask#(
        .NUM_INPUTS(DATA_IN_0_TENSOR_SIZE_DIM_0),
        .PRECISION(DATA_IN_0_PRECISION_0)
    )masker(
        .data(data_in),   // Unpacked array of 4 8-bit vectors
        .mask(output_mask),        // 4-bit mask
        .data_out_0(o_high_precision),
        .data_out_1(o_low_precision) 

    );






endmodule





