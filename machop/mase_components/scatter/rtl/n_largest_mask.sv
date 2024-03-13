
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
    logic [31:0] sum; 
    always @* begin
        mask = {NUM_INPUTS{1'b1}}; 
        sum = 0;
        largest_idx = 0;
        for (j = 0; j < N; j = j + 1) begin: COMPARISION
            largest_idx = +1; //Ensures default largest index is not masked twice
            for (i=0; i<NUM_INPUTS; i=i+1) begin
                if (input_array[i]>input_array[largest_idx] & mask[i]) begin
                    largest_idx = i;    
                    sum = sum+input_array[largest_idx];         
                end
            end
            mask[largest_idx] = 0;

        end
    end

  
    



endmodule

