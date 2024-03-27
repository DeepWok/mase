`timescale 1ns / 1ps


module scatter_threshold_synthesis(
    input clk,
    input rst,
    input logic  [16-1:0] data_in [16-1:0],
    output logic  [16-1:0] o_high_precision [16-1:0], 
    output logic  [16-1:0] o_low_precision [16-1:0] 
);

    //Logic to indentify values larger than x (2MSB check) //Can parameterise
    // logic [DATA_IN_0_PRECISION_0-2:0] mantisssa [DATA_IN_0_TENSOR_SIZE_DIM_0-1:0];
    // logic [DATA_IN_0_TENSOR_SIZE_DIM_0-1:0] sign;

    logic [16-1:0] high_precision_req_vec;
    for (genvar i = 0; i < 16; i = i + 1) begin
        //Check if greater than threshold
        assign high_precision_req_vec[i] = ($signed(data_in[i])> +6 | $signed(data_in[i])< -6);
    end

    
    //Logic to assign indicies based on priority

    //Pack array first
    wire [16-1:0] output_mask;


    logic [$clog2(16)-1:0] address_outliers[2-1:0]; //High slots

    generate
        if (1 == 1) begin: PE_D1
            priority_encoder #(
                .NUM_INPUT_CHANNELS(16),
                .NUM_OUPUT_CHANNELS(16),
                .NO_INDICIES(2) //High slots
            )
            encoder1(
                .input_channels(high_precision_req_vec),
                // .output_channels(address_outliers)
                .mask(output_mask)
            );

        end
        else if (1 == 2) begin: PE_D2
            priority_encoder #(
                .NUM_INPUT_CHANNELS(16),
                .NUM_OUPUT_CHANNELS(16),
                .NO_INDICIES(2)//high slots
            )
            encoder1(
                .input_channels(high_precision_req_vec),
                // .output_channels(address_outliers)
                .mask(output_mask)
            );

        end

    endgenerate



    //Logic to apply mask
    array_zero_mask#(
        .NUM_INPUTS(16),
        .PRECISION(16)
    )masker(
        .data(data_in),   // Unpacked array of 4 8-bit vectors
        .mask(output_mask),        // 4-bit mask
        .data_out_0(o_high_precision),
        .data_out_1(o_low_precision) 
    );



endmodule