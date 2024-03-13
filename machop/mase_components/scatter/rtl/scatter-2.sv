`timescale 1ns / 1ps
module scatter #(
    parameter DATA_IN_0_PRECISION_0 = 4,
    // parameter DATA_IN_0_PRECISION_1 = 3,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 4,
    // parameter DATA_IN_0_PRECISION_0 = 32,
    parameter DATA_OUT_0_PRECISION_0 = 4,
    parameter HIGH_SLOTS = 2,
    parameter MSB_CHECK_DEPTH = 2,
    parameter LOW_SLOTS = DATA_IN_0_TENSOR_SIZE_DIM_0-HIGH_SLOTS
    // parameter DATA_IN_0_PARALLELISM_DIM_0 = 16,
    // parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,
    // parameter DATA_OUT_0_PARALLELISM_DIM_0 = 16
    // parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1,
) (
    input clk,
    input rst,
    input  logic  [DATA_IN_0_PRECISION_0-1:0] data_in[DATA_IN_0_TENSOR_SIZE_DIM_0-1:0],
    output  [DATA_OUT_0_PRECISION_0-1:0] data_out[DATA_IN_0_TENSOR_SIZE_DIM_0-1:0],
    output [DATA_IN_0_PRECISION_0-1:0] o_high_precision [DATA_IN_0_TENSOR_SIZE_DIM_0-1:0], 
    output [DATA_IN_0_PRECISION_0-1:0] o_low_precision [DATA_IN_0_TENSOR_SIZE_DIM_0-1:0] 
);


    logic [MSB_CHECK_DEPTH-1:0] msb_array[DATA_IN_0_TENSOR_SIZE_DIM_0-1:0];

    //Quantize Abs Vector Input(Only compare top MSBS to save power)
    parallel_abs_quantize #(
        .NO_INPUTS(DATA_IN_0_TENSOR_SIZE_DIM_0),
        .INPUT_PRECISION(DATA_IN_0_PRECISION_0),
        .OUTPUT_PRECISION(MSB_CHECK_DEPTH)
    )parallel_abs_quantize_inst0(
        .input_array(data_in),
        .output_array(msb_array)
    );



    //Example
    // 1001,1110,1111,0011
    
    // 11,00,00,01
    // 1,0,0,1


    // 100100,111010,111101,001110
    // 110,001,000,011
    //1,0,0,1

    //Find mask
    logic [DATA_IN_0_TENSOR_SIZE_DIM_0-1:0] mask;
    n_largest_mask #(
        .NUM_INPUTS(DATA_IN_0_TENSOR_SIZE_DIM_0),
        .N(HIGH_SLOTS),
        .PRECISION(MSB_CHECK_DEPTH)

    )n_largest_mask_inst0(
        .input_array(msb_array),
        .mask(mask)
    );  


    // 1001,1110,1111,0011
    // 1,0,0,1
    // 1001,0,0,0011 - high
    // 0,1110,1111,0 - low


    // 1,0,0,1
    // 1001,0,0,0011 - high
    // 0,1110,1111,0 - low
    // Apply mask to input array
    array_zero_mask #(
        .NUM_INPUTS(DATA_IN_0_TENSOR_SIZE_DIM_0),
        .PRECISION(DATA_IN_0_PRECISION_0)
    )masker(
        .data(data_in),   // Unpacked array of 4 8-bit vectors
        .mask(mask),        // 4-bit mask
        .data_out_0(o_high_precision),  // Modified array
        .data_out_1(o_low_precision)  // Modified array
    
    );



    for (genvar i = 0; i < DATA_IN_0_TENSOR_SIZE_DIM_0; i = i + 1) begin
        //Use this to check testbench is working
        assign data_out[i] = data_in[i];
    end


   

endmodule




