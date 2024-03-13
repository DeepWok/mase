
`timescale 1ns / 1ps
// Change to N larger than threshold

module n_threshold_mask #(
    parameter NUM_INPUTS = 4,
    parameter N = 1,
    parameter PRECISION = 2,
    parameter THRESHOLD

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
    logic set; 
    always @* begin
        mask = {NUM_INPUTS{1'b1}}; 
        for (j = 0; j < N; j = j + 1) begin: COMPARISION
            set = 1'b0;
            for (i=0; i<NUM_INPUTS; i=i+1) begin
                if ((input_array[i] > THRESHOLD) & mask[i] & !set) begin
                    mask[i] = 0;
                    set = 1'b1;
                end
            end

        end
    end

  
    



endmodule

