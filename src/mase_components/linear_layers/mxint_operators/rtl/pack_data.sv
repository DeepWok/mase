`timescale 1ns / 1ps
module pack_data #(
    parameter IN_WIDTH = 1,
    parameter IN_SIZE = 8
) (
    input logic [IN_WIDTH - 1:0] data_in [IN_SIZE - 1:0],
    output logic [IN_WIDTH*IN_SIZE-1:0] data_out
);

    // Pack the array into a single vector
    for (genvar i = 0; i < IN_SIZE; i++) begin : reshape
        assign data_out[i*IN_WIDTH +: IN_WIDTH] = data_in[i];
    end

endmodule