`timescale 1ns / 1ps
// Join2 synchronises two sets of input handshake signals with a set of output handshaked signals 
module join2 #(
) (
    input logic [1:0] data_in_valid,
    output logic [1:0] data_in_ready,
    output logic data_out_valid,
    input logic data_out_ready
);

  // If only one of the inputs is valid - we need to stall that input and wait
  // for the other input by setting one of the ready bit to 0.
  // +-----------+-----------+------------+------------+------------+
  // | data_out_ready | invalid_0 | data_in_valid_1 | data_in_ready_0 | data_in_ready_1 |
  // +-----------+-----------+------------+------------+------------+
  // |         0 |         0 |          0 |          0 |          0 |
  // +-----------+-----------+------------+------------+------------+
  // |         0 |         0 |          1 |          0 |          0 |
  // +-----------+-----------+------------+------------+------------+
  // |         0 |         1 |          0 |          0 |          0 |
  // +-----------+-----------+------------+------------+------------+
  // |         0 |         1 |          1 |          0 |          0 |
  // +-----------+-----------+------------+------------+------------+
  // |         1 |         0 |          0 |          1 |          1 |
  // +-----------+-----------+------------+------------+------------+
  // |         1 |         0 |          1 |          1 |          0 |
  // +-----------+-----------+------------+------------+------------+
  // |         1 |         1 |          0 |          0 |          1 |
  // +-----------+-----------+------------+------------+------------+
  // |         1 |         1 |          1 |          1 |          1 |
  // +-----------+-----------+------------+------------+------------+
  assign data_in_ready[0] = data_out_ready & (!data_in_valid[0] | data_in_valid[1]);
  assign data_in_ready[1] = data_out_ready & (!data_in_valid[1] | data_in_valid[0]);
  assign data_out_valid   = &data_in_valid;

endmodule
