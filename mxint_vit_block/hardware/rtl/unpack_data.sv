`timescale 1ns / 1ps
module unpack_data #(
    parameter IN_WIDTH = 1,
    parameter IN_SIZE = 8
) (
    input logic [IN_WIDTH*IN_SIZE-1:0] data_in,
    output logic [IN_WIDTH - 1:0] data_out [IN_SIZE - 1:0]
);

    // Unpack the vector into an array
    for (genvar i = 0; i < IN_SIZE; i++) begin : reshape
        assign data_out[i] = data_in[i*IN_WIDTH +: IN_WIDTH];
    end

endmodule