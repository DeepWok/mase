/*
Module      : splitn
Description : This module implements a 1-to-N streaming interface handshake.
*/

`timescale 1ns / 1ps

module split_n #(
    parameter N = 2
) (
    input logic data_in_valid,
    output logic data_in_ready,
    output logic [N-1:0] data_out_valid,
    input logic [N-1:0] data_out_ready
);

generate
  if (N == 1) begin : gen_passthrough
    assign data_out_valid[0] = data_in_valid;
    assign data_in_ready = data_out_ready[0];
  end else begin : gen_split_handshake
    assign data_out_valid = {N{data_in_valid && data_in_ready}};
    assign data_in_ready  = &data_out_ready;
  end
endgenerate

endmodule
