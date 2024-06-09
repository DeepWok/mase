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

logic [N-1:0] ready_intermediate;

  for (genvar i = 0; i < N; i++) begin : handshake
    // We should wait to drive the output valid until all other ports are ready, without checking the current port
    // since this leads to a combinatorial loop
    assign ready_intermediate[i] = (i == 0) ? &data_out_ready[N-1:1]
                                  : (i == N-1) ? &data_out_ready[N-2:0]
                                  : &{data_out_ready[N-1:i+1], data_out_ready[i-1:0]};
    
    assign data_out_valid[i] = data_in_valid && ready_intermediate[i];
  end

  // Apply backpressure until all output ports are ready
  assign data_in_ready  = &data_out_ready;

endmodule
