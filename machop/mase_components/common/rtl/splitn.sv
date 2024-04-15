/*
Module      : splitn
Description : This module implements a 1-to-N streaming interface handshake.
*/

`timescale 1ns / 1ps

module splitn #(
    parameter N = 2
) (
    input  logic data_in_valid,
    output logic data_in_ready,
    output logic [N-1:0] data_out_valid,
    input  logic [N-1:0] data_out_ready
);

for (genvar i = 0; i < N; i++) begin
    logic [N-1:0] mask = 1 << i;
    logic all_others_are_ready = &(data_out_ready | mask);
    assign data_out_valid[i] = data_in_valid && all_others_are_ready;
end

assign data_in_ready = &data_out_ready;

endmodule
