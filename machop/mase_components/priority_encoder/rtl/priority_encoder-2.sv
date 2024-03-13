`timescale 1ns / 1ps
module priority_encoder #(
    parameter NUM_INPUT_CHANNELS = 2,
    parameter NUM_OUPUT_CHANNELS = 1,
    parameter NO_INDICIES = 1

)(
    input [NUM_INPUT_CHANNELS-1:0] input_channels,
    output logic [$clog2(NUM_INPUT_CHANNELS)-1:0] output_channels [NO_INDICIES-1:0],
    output logic [NUM_INPUT_CHANNELS-1:0] output_mask
);  

    //Check what happens when there are input channel is zero, what is default behaviour
    logic [NUM_INPUT_CHANNELS-1:0] mask;
    integer i;
    integer j;
    always @* begin
        mask = {NUM_INPUT_CHANNELS{1'b1}}; 
        
        for (j = 0; j < NO_INDICIES; j = j + 1) begin: PRIORITY_ENCODER
            //LSB Priority
            for (i=NUM_INPUT_CHANNELS-1; i>=0; i=i-1) begin
                // output_channels[j] = 1'b1;
                if (input_channels[i] & mask[i]) begin
                    mask[i] = 0;
                    output_channels[j] = i;
                    output_mask = mask;
                    
                end
            end
        end
    end


    



endmodule


module index_to_mask #(
    parameter NUM_INPUT_CHANNELS = 4,
    parameter NUM_OUPUT_CHANNELS = 1,
    parameter NO_INDICIES = 1,
    parameter OUTPUT_WIDTH = 4

)(
    input [$clog2(NUM_INPUT_CHANNELS)-1:0] indicies [NO_INDICIES-1:0],
    output logic [OUTPUT_WIDTH-1:0] output_mask
);
    integer i;
    always @* begin
        output_mask = 0;
        for (i=0; i< NO_INDICIES; i=i+1)
            output_mask[indicies[i]] = 1'b1;
    end


endmodule



module array_zero_mask#(
    parameter NUM_INPUTS =4,
    parameter PRECISION = 4
    )(
    input [PRECISION-1:0] data[NUM_INPUTS-1:0],   // Unpacked array of 4 8-bit vectors
    input [NUM_INPUTS-1:0] mask,        // 4-bit mask
    output logic [PRECISION-1:0] data_out_0[NUM_INPUTS-1:0],  // Modified array
    output logic [PRECISION-1:0] data_out_1[NUM_INPUTS-1:0]  // Modified array

);

// Always block that updates the output based on the mask
always @(*) begin
    integer i;
    for (i = 0; i < NUM_INPUTS; i = i + 1) begin
        // Check each bit of the mask; if it's 0, copy the original data, else set to zero
        if (mask[i] == 1'b1) begin
            data_out_0[i] = data[i];
            data_out_1[i] = 0; 

        end else begin
            data_out_0[i] = 0; 
            data_out_1[i] = data[i];
        end
    end
end

endmodule


module parallel_abs_quantize#(
    NO_INPUTS = 2,
    INPUT_PRECISION = 16,
    OUTPUT_PRECISION = 1
    )(
    input [INPUT_PRECISION-1:0] input_array [NO_INPUTS-1:0],
    output logic [OUTPUT_PRECISION-1:0] output_array [NO_INPUTS-1:0]
);

    logic sign  [INPUT_PRECISION-1:0];
    logic [OUTPUT_PRECISION-1:0] msbs [INPUT_PRECISION-1:0];

    for (genvar i = 0; i < NO_INPUTS; i = i + 1) begin
        //Extract sign bit
        assign sign[i] = input_array[i][INPUT_PRECISION-1];
        assign msbs[i] = input_array[i][INPUT_PRECISION-2:INPUT_PRECISION-2-OUTPUT_PRECISION];

        assign output_array[i] = (sign[i])? !msbs[i]:msbs[i];
    end

endmodule




`timescale 1ns / 1ps
module n_largest_mask #(
    parameter NUM_INPUTS = 4,
    parameter N = 1,
    parameter PRECISION = 2

)(
    input [PRECISION-1:0] input_array [NUM_INPUTS-1:0],
    output logic [PRECISION-1:0] masked_high_precision_array [NUM_INPUTS-1:0],
    output logic [PRECISION-1:0] masked_low_precision_array [NUM_INPUTS-1:0]
);  

    // Create zero mask of  for N largest values
    logic [NUM_INPUTS-1:0] mask;
    integer i;
    integer j;
    logic [$clog2(NUM_INPUTS)-1:0] largest_idx; 
    always @* begin
        mask = {NUM_INPUTS{1'b1}}; 
        
        for (j = 0; j < N; j = j + 1) begin: COMPARISION
            largest_idx = 0;
            for (i=0; i<NUM_INPUTS; i=i+1) begin
                if (input_array[i]>input_array[largest_idx] & mask[i]) begin
                    largest_idx = i;             
                end
            end
            mask[largest_idx] = 0;

        end
    end

    // Apply to input array
    array_zero_mask #(
        .NUM_INPUTS(NUM_INPUTS),
        .PRECISION(PRECISION)
    )masker(
        .data(input_array),   // Unpacked array of 4 8-bit vectors
        .mask(mask),        // 4-bit mask
        .data_out_0(masked_high_precision_array),  // Modified array
        .data_out_1(masked_low_precision_array)  // Modified array
    
    );

    



endmodule
