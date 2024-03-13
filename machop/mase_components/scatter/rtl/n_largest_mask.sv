
`timescale 1ns / 1ps
module n_largest_mask #(
    parameter NUM_INPUTS = 4,
    parameter N = 1,
    parameter PRECISION = 2

)(
    input [PRECISION-1:0] input_array [NUM_INPUTS-1:0],
    output logic [NUM_INPUTS-1:0] mask
    // output logic [PRECISION-1:0] masked_high_precision_array [NUM_INPUTS-1:0],
    // output logic [PRECISION-1:0] masked_low_precision_array [NUM_INPUTS-1:0]
);  

    // Create zero mask of  for N largest values
    // logic [NUM_INPUTS-1:0] mask;
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

  
    



endmodule


module array_zero_mask#(
    parameter NUM_INPUTS =4,
    parameter PRECISION = 4
    )(
    input [PRECISION-1:0] data [NUM_INPUTS-1:0],   // Unpacked array of 4 8-bit vectors
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
