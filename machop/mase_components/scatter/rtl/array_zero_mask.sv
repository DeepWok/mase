`timescale 1ns / 1ps

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
